import torch
import torch.nn as nn
import torch.nn.functional as F
from gvp import GVP, GVPConvLayer, LayerNorm
from torch_geometric.nn import global_mean_pool, GlobalAttention
from torch_scatter import scatter_mean

def tuple_sum(*args):
    """
    对多个(s, V)特征的元组进行逐元素求和
    参数:
        args: 多个(s, V)元组
    返回:
        (s, V): 求和后的特征
    """
    s = sum([x[0] for x in args])
    v = sum([x[1] for x in args])
    return (s, v)

def tuple_cat(*args, dim=-1):
    """
    对多个(s, V)特征的元组进行拼接
    参数:
        args: 多个(s, V)元组
        dim: 拼接维度
    返回:
        (s, V): 拼接后的特征
    """
    s = torch.cat([x[0] for x in args], dim=dim)
    v = torch.cat([x[1] for x in args], dim=dim)
    return (s, v)

class GVP_encoder(nn.Module):
    def __init__(
        self,
        node_in_dim=(1043, 3),
        node_h_dim=(256, 32),  # 修改：增加向量通道数
        edge_in_dim=(32, 1),
        edge_h_dim=(32, 1),
        num_layers=4,
        drop_rate=0.1,
        device='cuda:0'  # [v2-9 patch]
    ):
        super().__init__()
        self.device = device
        self.node_vector_norm = nn.LayerNorm(node_in_dim[1]).to(device)
        
        # 边向量处理增强
        self.edge_vector_transform = nn.Sequential(
            nn.Linear(3, 3),
            nn.LayerNorm(3)
        ).to(device)
        
        # 节点/边处理层
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True)
        ).to(device)
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True)
        ).to(device)
        
        # 带残差的GVP层
        self.layers = nn.ModuleList([
            GVPConvLayer(
                node_h_dim, edge_h_dim,
                activations=(nn.ELU(), None),
                vector_gate=True,
                drop_rate=drop_rate
            ).to(device) for _ in range(num_layers)
        ])
        self.res_conn = nn.ModuleList([  # 新增残差连接
            nn.Sequential(
                GVP(node_h_dim, node_h_dim),
                LayerNorm(node_h_dim)
            ).to(device) for _ in range(num_layers)
        ])

        # 输出层
        self.output_transform = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (node_h_dim[0], 0), activations=(None, None), vector_gate=True),
            nn.Linear(node_h_dim[0], 128),
            nn.ELU(),
            nn.Dropout(drop_rate)
        ).to(device)

    def forward(self, h_V, edge_index, h_E):
        # 节点向量处理
        scalar, vectors = h_V
        vectors = self.node_vector_norm(vectors.to(self.device))
        h_V = (scalar.to(self.device), vectors)
        h_V = self.W_v(h_V)
        
        # 边向量处理
        edge_scalar, edge_vectors = h_E
        edge_vectors = edge_vectors.squeeze(1)
        edge_vectors = self.edge_vector_transform(edge_vectors)
        edge_vectors = edge_vectors.unsqueeze(1)
        h_E = (edge_scalar.to(self.device), edge_vectors)
        h_E = self.W_e(h_E)
        
        # 带残差的GVP层
        for i, layer in enumerate(self.layers):
            h_V_new = layer(h_V, edge_index.to(self.device), h_E)
            h_V = tuple_sum(h_V_new, self.res_conn[i](h_V))  # 残差连接
            
        return self.output_transform(h_V)

class AMPRegressor(nn.Module):
    def __init__(self, input_dim=1043, hidden_dim=256, drop_rate=0.2, device='cuda:0'):  # [v2-9 patch]
        super().__init__()
        self.device = device
        
        # GVP编码器
        self.encoder = GVP_encoder(
            node_in_dim=(input_dim, 3),
            node_h_dim=(hidden_dim, 32),
            edge_in_dim=(32, 1),
            edge_h_dim=(32, 1),
            num_layers=4,
            drop_rate=drop_rate,
            device=device
        )
        
        # 池化层
        self.attention_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 1)
            )
        )
        self.pool_fc = nn.Linear(128 * 2, 128)
        
        # 回归器（保持128+553维度）
        self.regressor = nn.Sequential(
            nn.Linear(128 + 553, 64),  # 确保输入维度匹配
            nn.ELU(),
            nn.Dropout(drop_rate),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, data):
        data = data.to(self.device)
        
        # ==================== 节点特征处理 ====================
        # 确保节点特征维度正确
        if data.node_s.size(1) < 1043:
            padding = torch.zeros(data.node_s.size(0), 1043 - data.node_s.size(1), 
                                device=self.device)
            data.node_s = torch.cat([data.node_s, padding], dim=1)
        
        prottrans = data.node_s[:, :1024]
        dssp = data.node_s[:, 1024:1037]
        dihedrals = data.node_s[:, 1037:1043]
        
        scaled_features = torch.cat([
            F.layer_norm(prottrans, [1024]),
            F.layer_norm(dssp, [13]),
            F.layer_norm(dihedrals, [6])
        ], dim=1)

        # ==================== GVP编码 ====================
        encoded = self.encoder(
            (scaled_features, data.node_v),
            data.edge_index,
            (data.edge_s, data.edge_v)
        )
        
        # ==================== 池化处理 ====================
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(len(data.x), 
                                                                dtype=torch.long, 
                                                                device=self.device)
        
        pooled_mean = global_mean_pool(encoded, batch)
        pooled_attn = self.attention_pool(encoded, batch)
        
        pooled = torch.cat([pooled_mean, pooled_attn], dim=1)
        pooled = self.pool_fc(pooled)

        # ==================== 特征融合 ====================
        # 确保肽特征维度正确
        if not hasattr(data, 'peptide_feat'):
            # 创建全零肽特征作为后备
            data.peptide_feat = torch.zeros(len(data.x), 553, device=self.device)
        
        if data.peptide_feat.dim() == 1:
            data.peptide_feat = data.peptide_feat.unsqueeze(0)
        elif data.peptide_feat.dim() > 2:
            data.peptide_feat = data.peptide_feat.view(-1, 553)
        
        # 检查维度匹配
        if pooled.size(0) != data.peptide_feat.size(0):
            # 调整肽特征维度以匹配批次大小
            data.peptide_feat = data.peptide_feat[:pooled.size(0)]
        
        combined = torch.cat([pooled, data.peptide_feat], dim=1)

        # ==================== 回归预测 ====================
        output = self.regressor(combined)
        return output.squeeze(-1), encoded
    
    def configure_optimizers(self, lr=1e-3):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n,p in self.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n,p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        return torch.optim.AdamW(params, lr=lr)
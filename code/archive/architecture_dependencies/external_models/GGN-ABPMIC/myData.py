import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_cluster
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
import random
import re
from Bio.PDB import PDBParser
import hashlib
from torch.utils.data.sampler import Sampler
import traceback

class ProteinGraphDataset(Dataset):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.pdb_dir = args.pdb_dir
        self.prottrans_dir = args.prottrans_dir
        self.dssp_dir = args.dssp_dir
        self.mode = mode
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        self.num_rbf = getattr(args, 'num_rbf', 20)          # 默认20个RBF特征
        self.num_positional_embeddings = getattr(args, 'num_positional_embeddings', 16)  # 默认16个位置编码
        self._validate_config()

        self.sequences = []
        self.values = []
        self.node_counts = []

        self.peptide_features = []  # 新增肽特征列表

        try:
            self.csv_path = args.train_csv if mode == 'train' else args.val_csv
            self.df = pd.read_csv(self.csv_path)
            
            if 'sequence' not in self.df.columns or 'value' not in self.df.columns:
                raise ValueError("CSV必须包含'sequence'和'value'两列")
                
            for idx, row in self.df.iterrows():
                seq = str(row['sequence']).strip().upper()
                val = row['value']
                if self._is_valid_sequence(seq) and self._is_valid_value(val):
                    self.sequences.append(seq)
                    self.values.append(float(val))
                    # 提取肽特征：排除'sequence'和'value'列
                    peptide_feat = row.drop(['sequence', 'value']).values.astype(np.float32)
                    self.peptide_features.append(peptide_feat)

            self.node_counts = [len(seq) for seq in self.sequences]
            print(f"节点数预计算完成: 共 {len(self.node_counts)} 个有效样本")

            # 转换为张量
            self.peptide_features = torch.tensor(np.array(self.peptide_features), dtype=torch.float32)

            # [v2-9 patch] optional prebuilt graph cache (GGN_GRAPH_CACHE=path)
            self._gcache = {}
            import os as _os
            _gc = _os.environ.get('GGN_GRAPH_CACHE')
            if _gc and _os.path.exists(_gc):
                self._gcache = torch.load(_gc, map_location='cpu')
                print(f"[v2-9] graph cache loaded: {len(self._gcache)} seqs")

        except Exception as e:
            print(f"数据集初始化失败: {str(e)}")
            self.sequences = []
            self.values = []
            self.node_counts = []

        self.letter_to_num = {
            'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9,
            'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 
            'V':19, 'X':20
        }

    def _validate_config(self):
        """验证config参数完整性"""
        required_args = ['pdb_dir', 'prottrans_dir', 'dssp_dir', 'train_csv']
        for arg in required_args:
            if not hasattr(self.args, arg):
                raise ValueError(f"Config缺少必要参数: {arg}")
        
        # 设置默认值
        if not hasattr(self.args, 'top_k'):
            print("警告: 使用默认top_k=30")
            self.args.top_k = 30

    def _validate_dataset(self):
        """数据完整性校验（移除id列依赖）"""
        for idx in tqdm(range(len(self)), desc="数据校验"):
            try:
                data = self[idx]
                assert data.node_s.shape[0] == len(self.sequences[idx]), "节点数不匹配"
            except Exception as e:
                # 使用序列哈希作为标识
                seq_hash = hashlib.md5(self.sequences[idx].encode()).hexdigest()
                print(f"数据校验失败: {seq_hash} - {str(e)}")
                raise
    
    def _load_prottrans(self, seq):
        """加载ProtTrans特征并转换为Tensor"""
        safe_seq = re.sub(r'[\\/*?"<>|]', "_", seq)
        path = os.path.join(self.prottrans_dir, f"{safe_seq}_prottrans.npy")
        try:
            prottrans = np.load(path)
            return torch.tensor(prottrans, 
                            dtype=torch.float32,
                            device=self.device)
        except Exception as e:
            print(f"ProtTrans加载失败: {safe_seq}")  # 修改seq_id为safe_seq
            return torch.zeros((len(seq), 1024),  # 使用参数seq而不是seq_id
                            device=self.device)

    def __getitem__(self, idx):
        # 安全初始化
        if not isinstance(idx, int):
            raise ValueError("索引必须是整数")
        
        seq = self.sequences[idx] if 0 <= idx < len(self.sequences) else ""
        seq_length = len(seq) if seq else 0
        seq_id = f"invalid_{idx}"
        
        try:
            # 生成唯一标识
            if seq:
                seq_hash = hashlib.md5(seq.encode()).hexdigest()
                seq_id = f"{seq_hash}_{idx}"

            # [v2-9 patch] fast path from prebuilt cache
            if seq in self._gcache:
                g = self._gcache[seq]
                peptide_feat = self.peptide_features[idx]
                if peptide_feat.dim() == 1:
                    peptide_feat = peptide_feat.unsqueeze(0)
                return Data(
                    x=g['x'], seq=g['seq'], node_s=g['node_s'], node_v=g['node_v'],
                    edge_index=g['edge_index'], edge_s=g['edge_s'], edge_v=g['edge_v'],
                    y=torch.tensor([self.values[idx]], device=self.device),
                    name=seq,
                    peptide_feat=peptide_feat.to(self.device))

            # 加载特征
            coords = self._load_pdb_coordinates(seq)
            prottrans = self._load_prottrans(seq)
            dssp = self._load_dssp_features(seq)
            
            # 检查维度
            if len(coords['CA']) != len(seq):
                raise ValueError(f"坐标({len(coords['CA'])})与序列({len(seq)})长度不匹配")
                
            # 计算节点特征
            ca = coords['CA']
            dihedrals = self._calculate_dihedrals(coords)
            orientations = self._calculate_orientations(coords)
            sidechains = self._calculate_sidechains(coords)
            
            # 构建图 - 确保至少有1个邻居
            top_k = min(max(1, getattr(self.args, 'top_k', 30)), len(ca)-1)
            edge_index = torch_cluster.knn_graph(ca, k=top_k, loop=False)
            
            # 特征拼接 - 确保维度正确
            node_s = torch.cat([
                dihedrals[:, :6],
                prottrans[:, :min(1024, prottrans.size(1))],
                dssp[:, :min(13, dssp.size(1))]
            ], dim=1)
            
            # 节点向量特征 - 确保是[N,3,3]格式
            node_v = torch.cat([
                orientations.reshape(len(seq), -1, 3),
                sidechains.reshape(len(seq), -1, 3)
            ], dim=1)
            node_v = node_v.permute(0, 2, 1)  # [N,3,3]
            
            # 边向量特征 - 确保是[E,1,3]格式
            edge_vec = ca[edge_index[0]] - ca[edge_index[1]]  # [E,3]
            edge_v = edge_vec.unsqueeze(1)  # [E,1,3]
            edge_v = edge_v / (torch.norm(edge_v, dim=-1, keepdim=True) + 1e-7)  # 归一化
            
            # 边标量特征 - 确保是[E,32]格式
            edge_dist = torch.norm(ca[edge_index[0]] - ca[edge_index[1]], dim=-1)
            edge_s = torch.cat([
                self._rbf(edge_dist),
                self._positional_embeddings(edge_index)
            ], dim=-1)
            
            # 确保最终维度正确
            if node_s.size(1) < 1043:
                padding = torch.zeros(node_s.size(0), 1043 - node_s.size(1), 
                                    device=self.device)
                node_s = torch.cat([node_s, padding], dim=1)
            
            # 处理肽特征 - 确保始终是2D [1,553]
            peptide_feat = self.peptide_features[idx]
            if peptide_feat.dim() == 1:
                peptide_feat = peptide_feat.unsqueeze(0)  # [553] -> [1,553]
            elif peptide_feat.dim() > 2:
                peptide_feat = peptide_feat.view(1, -1)[:, :553]  # 处理异常维度
            
            _d = Data(
                x=ca,
                seq=torch.tensor([self.letter_to_num.get(aa, 20) for aa in seq], 
                        device=self.device),
                node_s=node_s[:, :1043],  # 确保是[N,1043]
                node_v=node_v[:, :, :3],  # 确保是[N,3,3]
                edge_index=edge_index,
                edge_s=edge_s[:, :32],    # 确保是[E,32]
                edge_v=edge_v[:, :, :3],  # 确保是[E,1,3]
                y=torch.tensor([self.values[idx]], device=self.device),
                name=seq,
                peptide_feat=peptide_feat.to(self.device)  # 确保是[1,553]
            )
            # [v2-9 patch] store to cache for reuse
            self._gcache[seq] = {k: getattr(_d, k).cpu() for k in
                                 ('x', 'seq', 'node_s', 'node_v',
                                  'edge_index', 'edge_s', 'edge_v')}
            return _d

        except Exception as e:
            print(f"\n样本处理错误: {seq_id}\n{str(e)}")
            traceback.print_exc()
            return self._get_safe_template(seq_length)

    def _get_empty_data(self, seq):
        """修正空数据模板中的属性引用"""
        return Data(
            x=torch.zeros(0, 3, device=self.device),
            seq=torch.zeros(len(seq), dtype=torch.long, device=self.device),
            node_s=torch.zeros(0, 1043, device=self.device),
            node_v=torch.zeros(0, 3, 3, device=self.device),
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=self.device),
            edge_s=torch.zeros(0, self.num_rbf + self.num_positional_embeddings, device=self.device),
            edge_v=torch.zeros(0, 1, 3, device=self.device),
            y=torch.tensor([0.0], device=self.device),
            name="invalid_" + seq
        )

    # 新增必须的辅助方法
    def _load_pdb_coordinates(self, seq):
        """增强的PDB坐标加载方法"""
        try:
            safe_seq = re.sub(r'[\\/*?"<>|]', "_", seq)
            pdb_path = os.path.join(self.pdb_dir, f"{safe_seq}.pdb")
            
            if not os.path.exists(pdb_path):
                raise FileNotFoundError(pdb_path)
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', pdb_path)
            
            # 初始化坐标字典
            coords = {
                'N': [], 'CA': [], 'C': [], 'CB': []
            }
            
            for residue in structure.get_residues():
                # 必须存在的原子
                try:
                    n = residue['N'].coord
                    ca = residue['CA'].coord
                    c = residue['C'].coord
                except KeyError as e:
                    print(f"残基{residue}缺少必需原子: {str(e)}")
                    continue
                    
                # 处理CB原子
                try:
                    cb = residue['CB'].coord
                except KeyError:
                    # 生成虚拟CB坐标
                    cb = self._generate_virtual_cb(n, ca, c)
                    if isinstance(cb, torch.Tensor):
                        cb = cb.cpu().numpy()
                    
                # 存储坐标
                coords['N'].append(n)
                coords['CA'].append(ca)
                coords['C'].append(c)
                coords['CB'].append(cb)
            
            # 转换为张量
            tensor_coords = {}
            for key in coords:
                if len(coords[key]) != len(seq):
                    # 填充缺失的坐标
                    coords[key] = [np.zeros(3) for _ in range(len(seq))]
                tensor_coords[key] = torch.tensor(np.array(coords[key]), 
                                            dtype=torch.float32,
                                            device=self.device)
                
            return tensor_coords
            
        except Exception as e:
            print(f"PDB加载失败: {str(e)}")
            # 返回全零坐标
            return {
                'N': torch.zeros(len(seq), 3, device=self.device),
                'CA': torch.zeros(len(seq), 3, device=self.device),
                'C': torch.zeros(len(seq), 3, device=self.device),
                'CB': torch.zeros(len(seq), 3, device=self.device)
            }

    def _load_dssp_features(self, seq):
        """支持.npy格式的DSSP特征加载"""
        safe_seq = re.sub(r'[\\/*?"<>|]', "_", seq)
        # 尝试两种文件格式
        for ext in ['.npy', '.csv']:
            path = os.path.join(self.dssp_dir, f"{safe_seq}_dssp{ext}")
            if os.path.exists(path):
                try:
                    if ext == '.npy':
                        dssp = np.load(path)
                    else:
                        dssp = pd.read_csv(path, header=None).values
                    
                    if len(dssp) != len(seq):
                        raise ValueError(f"DSSP特征数({len(dssp)})与序列长度({len(seq)})不匹配")
                    
                    return torch.tensor(dssp, dtype=torch.float32, device=self.device)
                except Exception as e:
                    print(f"DSSP加载警告: {path} - {str(e)}")
                    continue
        
        print(f"DSSP加载失败: 未找到{safe_seq}的.npy或.csv文件")
        return torch.zeros((len(seq), 13), dtype=torch.float32, device=self.device)


    def _is_valid_sequence(self, seq):
        """严格验证氨基酸序列（支持扩展字符）"""
        return bool(re.fullmatch("^[ARNDCQEGHILKMFPSTWYVUOBZX]+$", seq))

    def _is_valid_value(self, val):
        """验证目标值是否为有效浮点数"""
        try:
            float(val)
            return True
        except:
            return False

    def __len__(self):
        """返回数据集中的样本总数"""
        return len(self.sequences)  # 直接返回序列列表的长度
    
    def _load_data(self):
        """严格的数据加载流程"""
        try:
            # 读取CSV文件
            self.df = pd.read_csv(self.csv_path, header=None, names=['sequence', 'value'])
            # 清洗序列并验证
            self.sequences = self._clean_sequences(self.df['sequence'])
            # 强制类型检查
            if not isinstance(self.sequences, list):
                raise ValueError("sequences 必须为列表类型")
            print(f"成功加载 {len(self.sequences)} 个样本")
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            self.sequences = []

    # ======================= 核心修改部分 ======================= #
    def _parse_pdb_coords(self, pdb_path):
        """更健壮的PDB坐标解析方法"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        
        coords = {'N': [], 'CA': [], 'C': [], 'CB': []}
        
        for residue in structure.get_residues():
            # 必须存在的原子
            try:
                n = residue['N'].coord
                ca = residue['CA'].coord
                c = residue['C'].coord
            except KeyError as e:
                print(f"残基{residue}缺少必需原子: {str(e)}")
                continue
                
            # 处理CB原子
            try:
                cb = residue['CB'].coord
            except KeyError:
                # 甘氨酸等特殊情况，生成虚拟CB
                cb = self._generate_virtual_cb(n, ca, c)
                
            coords['N'].append(n)
            coords['CA'].append(ca)
            coords['C'].append(c)
            coords['CB'].append(cb)
        
        # 转换为张量
        tensor_coords = {}
        for key in coords:
            tensor_coords[key] = torch.tensor(np.array(coords[key]), 
                                            dtype=torch.float32,
                                            device=self.device)
        return tensor_coords

    def _generate_virtual_cb(self, n, ca, c):
        """改进的虚拟CB生成方法，正确处理CUDA张量"""
        # 确保输入是numpy数组
        if isinstance(n, torch.Tensor):
            n = n.detach().cpu().numpy()
        if isinstance(ca, torch.Tensor):
            ca = ca.detach().cpu().numpy()
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        
        # 计算平面法向量
        v_n = n - ca
        v_c = c - ca
        normal = np.cross(v_n, v_c)
        norm = np.linalg.norm(normal) + 1e-7
        
        # 生成CB坐标 (1.54Å是典型C-C键长)
        cb = ca + (normal / norm) * 1.54
        
        # 转换回tensor并保持设备一致
        if hasattr(self, 'device'):
            return torch.tensor(cb, device=self.device)
        return torch.tensor(cb)

    def _calculate_dihedrals(self, coords):
        """完全重写的二面角计算方法"""
        try:
            ca = coords['CA']
            seq_len = len(ca)
            
            if seq_len < 4:
                return torch.zeros(seq_len, 6, device=self.device)
                
            # 计算所有可能的二面角
            p0 = ca[:-3]  # 第i个原子
            p1 = ca[1:-2] # 第i+1个原子
            p2 = ca[2:-1] # 第i+2个原子
            p3 = ca[3:]   # 第i+3个原子
            
            # 计算向量
            u1 = p1 - p0
            u2 = p2 - p1
            u3 = p3 - p2
            
            # 计算法向量
            n1 = torch.linalg.cross(u1, u2, dim=-1)
            n2 = torch.linalg.cross(u2, u3, dim=-1)
            
            # 计算角度
            norm_prod = torch.norm(n1, dim=-1) * torch.norm(n2, dim=-1) + 1e-7
            cos_theta = (n1 * n2).sum(-1) / norm_prod
            sin_theta = torch.linalg.cross(n1, n2, dim=-1).sum(-1) / norm_prod
            
            # 初始化结果张量
            dihedrals = torch.zeros(seq_len, 2, device=self.device)
            
            # 计算有效索引范围
            valid_len = seq_len - 3
            if valid_len != len(cos_theta):
                raise ValueError(f"有效长度不匹配: {valid_len} vs {len(cos_theta)}")
            
            # 填充有效二面角
            dihedrals[1:1+valid_len] = torch.stack([cos_theta, sin_theta], dim=-1)
            
            # 生成完整特征
            full_dihedrals = torch.cat([
                dihedrals,
                dihedrals.roll(1, dims=0),
                dihedrals.roll(2, dims=0)
            ], dim=-1)
            
            return full_dihedrals
            
        except Exception as e:
            print(f"二面角计算失败: {str(e)}")
            return torch.zeros(len(coords['CA']), 6, device=self.device)

    def _calculate_orientations(self, coords):
        """修正后的方向特征计算方法"""
        try:
            ca = coords['CA']
            seq_len = len(ca)
            
            if seq_len < 2:
                return torch.zeros(seq_len, 2, 3, device=self.device)
            
            # 使用torch.narrow代替切片操作
            forward = ca.narrow(0, 1, seq_len-1) - ca.narrow(0, 0, seq_len-1)
            backward = ca.narrow(0, 0, seq_len-1) - ca.narrow(0, 1, seq_len-1)
            
            # 使用pad进行填充
            forward = F.pad(forward, (0,0,0,1))  # 在第二维(序列维度)末尾填充1个零
            backward = F.pad(backward, (0,0,1,0))  # 在第二维开头填充1个零
            
            # 归一化并堆叠
            return torch.stack([
                F.normalize(forward, dim=-1),
                F.normalize(backward, dim=-1)
            ], dim=1)
            
        except Exception as e:
            print(f"方向计算失败: {str(e)}")
            return torch.zeros(len(coords['CA']), 2, 3, device=self.device)

    def _calculate_sidechains(self, coords):
        """增强的侧链方向计算方法"""
        try:
            # 确保CB坐标存在
            if 'CB' not in coords:
                raise KeyError("坐标中缺少CB键")
                
            ca = coords['CA']
            cb = coords['CB']
            
            # 检查维度一致性
            if len(ca) != len(cb):
                raise ValueError(f"CA({len(ca)})和CB({len(cb)})数量不匹配")
                
            # 计算方向向量
            vectors = cb - ca
            
            # 处理零向量情况
            norm = torch.norm(vectors, dim=-1, keepdim=True) + 1e-7
            normalized = vectors / norm
            
            return normalized.unsqueeze(1)  # 保持维度兼容
            
        except Exception as e:
            print(f"侧链计算失败: {str(e)}")
            # 返回合理的默认值
            return torch.zeros(len(coords['CA']), 1, 3, device=self.device)

    # ===================== 修改加载逻辑 ===================== #
    def _load_features(self, seq):
        """统一特征加载入口"""
        # 生成安全文件名
        safe_seq = re.sub(r'[\\/*?"<>|]', "_", seq)
        
        # 加载各特征
        coords = self._load_pdb_coordinates(seq)
        prottrans = self._load_prottrans(seq)
        dssp = self._load_dssp_features(seq)
        
        # 添加维度校验
        if len(coords['CA']) != len(seq):
            raise ValueError(f"PDB坐标长度({len(coords['CA'])})与序列长度({len(seq)})不符")
        
        return coords, prottrans, dssp


    def _get_safe_template(self, seq_length=0):
        return Data(
            x=torch.zeros(seq_length, 3, device=self.device),
            seq=torch.zeros(seq_length, dtype=torch.long, device=self.device),
            node_s=torch.zeros(seq_length, 1043, device=self.device),
            node_v=torch.zeros(seq_length, 3, 3, device=self.device),
            edge_index=torch.zeros(2, 0, dtype=torch.long, device=self.device),
            edge_s=torch.zeros(0, 32, device=self.device),
            edge_v=torch.zeros(0, 1, 3, device=self.device),
            y=torch.tensor([0.0], device=self.device),
            peptide_feat=torch.zeros(1, 553),  # 确保虚拟数据也是2D
            name="invalid"
        )

    # ===================== 辅助方法 ===================== #
    def _clean_sequences(self, sequences):
        """带严格校验的序列清洗"""
        valid_seqs = []
        for seq in sequences:
            # 移除空格并转为大写
            cleaned = seq.replace(" ", "").upper().strip()
            # 验证氨基酸字符
            if not re.match("^[ACDEFGHIKLMNPQRSTVWY]*$", cleaned):
                print(f"无效序列: {seq}")
                continue
            valid_seqs.append(cleaned)
        return valid_seqs

    def _precompute_node_counts(self):
        return [len(seq) for seq in self.sequences]

    def _rbf(self, distances):
        """径向基函数特征（添加属性检查）"""
        if not hasattr(self, 'num_rbf'):
            self.num_rbf = 20  # 默认值
            
        centers = torch.linspace(0, 20, self.num_rbf, device=self.device)
        width = 20 / self.num_rbf
        return torch.exp(-((distances.unsqueeze(-1) - centers) / width)**2)

    def _positional_embeddings(self, edge_index):
        """位置编码（添加属性检查）"""
        if not hasattr(self, 'num_positional_embeddings'):
            self.num_positional_embeddings = 16  # 默认值
            
        delta = edge_index[0] - edge_index[1]
        frequencies = torch.exp(
            torch.arange(0, self.num_positional_embeddings, 2, device=self.device) * 
            -torch.log(torch.tensor(10000.0)) / self.num_positional_embeddings
        )
        angles = delta.unsqueeze(-1) * frequencies
        return torch.cat([torch.cos(angles), torch.sin(angles)], -1)

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, node_counts, max_nodes=3000, shuffle=True):
        """
        修正后的批次采样器，移除了device参数
        
        参数:
            node_counts: 每个样本的节点数列表
            max_nodes: 单个批次的最大节点数
            shuffle: 是否打乱顺序
        """
        super().__init__(None)
        self.node_counts = node_counts
        self.max_nodes = max_nodes
        self.shuffle = shuffle
        self._form_batches()

    def _form_batches(self):
        """形成批次"""
        indices = list(range(len(self.node_counts)))
        if self.shuffle:
            random.shuffle(indices)
            
        self.batches = []
        current_batch = []
        current_count = 0
        
        for idx in indices:
            if current_count + self.node_counts[idx] > self.max_nodes:
                if current_batch:
                    self.batches.append(current_batch)
                current_batch = [idx]
                current_count = self.node_counts[idx]
            else:
                current_batch.append(idx)
                current_count += self.node_counts[idx]
                
        if current_batch:
            self.batches.append(current_batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
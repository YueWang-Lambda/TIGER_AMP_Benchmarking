import torch
import torch.nn as nn
import torch.nn.functional as F

class MICContrastiveLoss(nn.Module):
    """适用于抗菌肽MIC回归任务的对比学习损失"""
    def __init__(self, temperature=0.1, margin=1.0):
        """
        :param temperature: 温度系数，控制分布尖锐程度
        :param margin: 边界值，用于构建相似样本对的阈值
        """
        super(MICContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, embeddings, mic_values):
        """
        :param embeddings: 模型输出的特征嵌入 [batch_size, embed_dim] 
        :param mic_values: MIC真实值 [batch_size]
        """
        batch_size = mic_values.size(0)
        
        # 确保嵌入向量是二维的
        if embeddings.dim() > 2:
            embeddings = embeddings.view(batch_size, -1)
        
        # 计算MIC值的绝对差异矩阵
        mic_diff = torch.abs(mic_values.unsqueeze(0) - mic_values.unsqueeze(1))  # [batch, batch]
        
        # 构建相似性掩码（MIC差异小于阈值的视为相似样本）
        similar_mask = (mic_diff < self.margin).float()  # [batch, batch]
        diag_mask = torch.eye(batch_size, device=mic_values.device)  # 排除自身
        similar_mask = similar_mask * (1 - diag_mask)
        
        # 计算特征相似度
        embeddings_norm = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.T)  # [batch, batch]
        
        # 确保相似度矩阵和掩码矩阵维度一致
        if similarity.size() != similar_mask.size():
            min_dim = min(similarity.size(0), similar_mask.size(0))
            similarity = similarity[:min_dim, :min_dim]
            similar_mask = similar_mask[:min_dim, :min_dim]
        
        # 分离正负样本
        pos_samples = similarity * similar_mask
        neg_samples = similarity * (1 - similar_mask)
        
        # 对比损失计算
        numerator = torch.sum(torch.exp(pos_samples / self.temperature))
        denominator = torch.sum(torch.exp(neg_samples / self.temperature))
        
        # 添加小常数防止除零
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        return loss / batch_size
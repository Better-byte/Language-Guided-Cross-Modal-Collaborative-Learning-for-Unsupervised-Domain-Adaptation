import torch
import torch.nn.functional as F

class CosimAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(CosimAlignmentLoss, self).__init__()
    
    def forward(self, text_seq, vision_seq):
        """
        计算余弦相似度对齐损失
        
        参数:
            text_seq: 文本序列特征 [batch_size, seq_len, feature_dim]
            vision_seq: 视觉序列特征 [batch_size, seq_len, feature_dim]
        
        返回:
            余弦相似度对齐损失值
        """
        text_vec = text_seq.mean(dim=1)
        vision_vec = vision_seq.mean(dim=1)
        
        text_vec = F.normalize(text_vec, p=2, dim=1)
        vision_vec = F.normalize(vision_vec, p=2, dim=1)
        
        cosine_sim = (text_vec * vision_vec).sum(dim=1)
        return 1 - cosine_sim.mean()
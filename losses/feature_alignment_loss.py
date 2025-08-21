import torch
import torch.nn.functional as F

class FeatureAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
    
    def forward(self, text_feat, vision_feat):
        """
        计算特征对齐损失（均值和方差对齐）
        
        参数:
            text_feat: 文本特征 [batch_size, feature_dim]
            vision_feat: 视觉特征 [batch_size, feature_dim]
        
        返回:
            特征对齐损失值
        """
        mean_text = text_feat.mean(dim=0)
        var_text = text_feat.var(dim=0)
        mean_vision = vision_feat.mean(dim=0)
        var_vision = vision_feat.var(dim=0)
        
        mean_loss = F.mse_loss(mean_text, mean_vision)
        var_loss = F.mse_loss(var_text, var_vision)
        return mean_loss + var_loss
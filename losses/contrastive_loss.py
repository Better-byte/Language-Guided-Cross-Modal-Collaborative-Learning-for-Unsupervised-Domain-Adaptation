import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones([]) * temperature)
    
    def forward(self, text_feat, vision_feat, labels=None):
        """
        计算对比损失（InfoNCE损失）
        
        参数:
            text_feat: 文本特征 [batch_size, feature_dim]
            vision_feat: 视觉特征 [batch_size, feature_dim]
            labels: 标签（可选），用于有监督对比学习
        
        返回:
            对比损失值
        """
        batch_size = text_feat.size(0)
        
        text_feat = F.normalize(text_feat, p=2, dim=1)
        vision_feat = F.normalize(vision_feat, p=2, dim=1)
        
        sim_matrix = torch.matmul(text_feat, vision_feat.T) / self.temperature
        
        if labels is not None:
            # 有监督对比学习
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(text_feat.device)
            
            # 计算logits
            logits = torch.exp(sim_matrix)
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=text_feat.device)
            
            # 正样本对
            positives = (logits * mask * logits_mask).sum(dim=1)
            
            # 负样本对
            negatives = (logits * logits_mask).sum(dim=1)
            
            loss = -torch.log(positives / negatives).mean()
        else:
            # 无监督对比学习
            labels_contrastive = torch.arange(batch_size).to(text_feat.device)
            loss = F.cross_entropy(sim_matrix, labels_contrastive) + \
                   F.cross_entropy(sim_matrix.T, labels_contrastive)
            loss = loss / 2
        
        return loss
import torch
import torch.nn.functional as F

class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    
    def forward(self, logits):
        """
        计算熵损失，用于无监督学习中的不确定性最小化
        
        参数:
            logits: 模型输出的logits [batch_size, num_classes]
        
        返回:
            熵损失值
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        return -(probs * log_probs).sum(dim=1).mean()
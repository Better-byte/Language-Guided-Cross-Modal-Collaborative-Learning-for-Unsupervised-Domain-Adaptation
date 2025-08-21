import torch
import torch.nn.functional as F

class ConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
    
    def js_div(self, p, q):
        """计算Jensen-Shannon散度"""
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') +
                      F.kl_div(q.log(), m, reduction='batchmean'))
    
    def forward(self, text_logits, vision_logits, final_logits):
        """
        计算模态间预测一致性损失
        
        参数:
            text_logits: 文本分支的logits [batch_size, num_classes]
            vision_logits: 视觉分支的logits [batch_size, num_classes]
            final_logits: 融合分支的logits [batch_size, num_classes]
        
        返回:
            一致性损失值
        """
        text_probs = F.softmax(text_logits, dim=1)
        vision_probs = F.softmax(vision_logits, dim=1)
        final_probs = F.softmax(final_logits, dim=1)
        
        loss_t_v = self.js_div(text_probs, vision_probs)
        loss_t_f = self.js_div(text_probs, final_probs)
        loss_v_f = self.js_div(vision_probs, final_probs)
        
        return (loss_t_v + loss_t_f + loss_v_f) / 3
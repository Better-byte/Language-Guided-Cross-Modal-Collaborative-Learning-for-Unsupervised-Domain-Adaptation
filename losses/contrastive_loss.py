import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.ones([]) * temperature)
    
    def forward(self, text_feat, vision_feat, labels=None):

        batch_size = text_feat.size(0)
        
        text_feat = F.normalize(text_feat, p=2, dim=1)
        vision_feat = F.normalize(vision_feat, p=2, dim=1)
        
        sim_matrix = torch.matmul(text_feat, vision_feat.T) / self.temperature
        
        if labels is not None:

            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(text_feat.device)
            

            logits = torch.exp(sim_matrix)
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=text_feat.device)
            

            positives = (logits * mask * logits_mask).sum(dim=1)
            

            negatives = (logits * logits_mask).sum(dim=1)
            
            loss = -torch.log(positives / negatives).mean()
        else:

            labels_contrastive = torch.arange(batch_size).to(text_feat.device)
            loss = F.cross_entropy(sim_matrix, labels_contrastive) + \
                   F.cross_entropy(sim_matrix.T, labels_contrastive)
            loss = loss / 2
        

        return loss

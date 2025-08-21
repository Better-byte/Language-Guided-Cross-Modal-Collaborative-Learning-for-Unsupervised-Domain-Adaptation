import torch
import torch.nn.functional as F

class CosimAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(CosimAlignmentLoss, self).__init__()
    
    def forward(self, text_seq, vision_seq):

        text_vec = text_seq.mean(dim=1)
        vision_vec = vision_seq.mean(dim=1)
        
        text_vec = F.normalize(text_vec, p=2, dim=1)
        vision_vec = F.normalize(vision_vec, p=2, dim=1)
        
        cosine_sim = (text_vec * vision_vec).sum(dim=1)

        return 1 - cosine_sim.mean()

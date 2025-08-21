import torch
import torch.nn.functional as F

class FeatureAlignmentLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
    
    def forward(self, text_feat, vision_feat):

        mean_text = text_feat.mean(dim=0)
        var_text = text_feat.var(dim=0)
        mean_vision = vision_feat.mean(dim=0)
        var_vision = vision_feat.var(dim=0)
        
        mean_loss = F.mse_loss(mean_text, mean_vision)
        var_loss = F.mse_loss(var_text, var_vision)

        return mean_loss + var_loss

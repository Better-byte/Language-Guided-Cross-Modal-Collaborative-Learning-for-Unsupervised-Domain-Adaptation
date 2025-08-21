import torch
import torch.nn.functional as F

class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()
    
    def forward(self, logits):

        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        return -(probs * log_probs).sum(dim=1).mean()

import torch.nn as nn
import clip

class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model
        
    def forward(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
        return features
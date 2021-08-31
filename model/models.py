import torch.nn as nn

from model.resnet50 import Resnet50
from model.vit import ViT

class MainModel(nn.Module):
    def __init__(self, num_classes=18, pretrained=True):
        super().__init__()
        self.vit = ViT(num_classes, pretrained=pretrained)
        
    def forward(self, x):
        x = self.vit(x)
        return x
from torchvision import models
import torch.nn as nn
import math

class Resnet50(nn.Module):
    def __init__(self, num_classes=18, pretrained=True):
        super().__init__()
        self.model_ft = models.resnet50(pretrained=pretrained)

        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs,num_classes)
        nn.init.xavier_uniform_(self.model_ft.fc.weight)
        stdv = 1. / math.sqrt(self.model_ft.fc.weight.size(1))
        self.model_ft.fc.bias.data.uniform_(-stdv, stdv)

        # all layer freeze
        for param in self.model_ft.parameters() :
            param.requires_grad = False

        # FC unfreeze
        for param in self.model_ft.fc.parameters() :
            param.requires_grad = True

        # 학습시킬 레이어 unfreeze 해주시면 됩니다.
        # for param in model_ft.layer4.parameters() :
        #     param.requires_grad = True

        return

    def forward(self, x):
        x = self.model_ft(x)
        return x
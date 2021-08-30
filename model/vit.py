import timm
import torch.nn as nn
import math

class ViT(nn.Module):
    def __init__(self, num_classes=18, pretrained=True):
        super().__init__()
        self.model_ft = timm.create_model('vit_base_patch16_224', pretrained=True)

        num_ftrs = self.model_ft.head.in_features
        self.model_ft.head = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(self.model_ft.head.weight)
        stdv = 1. / math.sqrt(self.model_ft.head.weight.size(1))
        self.model_ft.head.bias.data.uniform_(-stdv, stdv)

        # all layer freeze
        for param in self.model_ft.parameters():
            param.requires_grad = False

        # FC unfreeze
        for param in self.model_ft.head.parameters():
            param.requires_grad = True

        # 학습시킬 레이어 unfreeze 해주시면 됩니다.
        # for param in model_ft.blocks.parameters() :
        #     param.requires_grad = True
        return

    def forward(self, x):
        x = self.model_ft(x)
        return x
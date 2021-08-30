import timm
import torch.nn as nn
import math

class ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(64, 3, kernel_size=(1,1), stride=(1,1))        
        nn.init.xavier_uniform_(self.conv1.weight)
        stdv = 1. / math.sqrt(self.conv1.weight.size(1))
        self.conv1.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.conv2.weight)
        stdv = 1. / math.sqrt(self.conv2.weight.size(1))
        self.conv2.bias.data.uniform_(-stdv, stdv)

        self.model_ft = timm.create_model('vit_base_patch16_224', pretrained=True)

        num_ftrs = self.model_ft.head.in_features
        self.model_ft.head = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(self.model_ft.head.weight)
        stdv = 1. / math.sqrt(self.model_ft.head.weight.size(1))
        self.model_ft.head.bias.data.uniform_(-stdv, stdv)

        for param in self.model_ft.parameters():
            param.requires_grad = False

        for param in self.conv1.parameters():
            param.requires_grad = True

        for param in self.conv2.parameters():
            param.requires_grad = True

        for param in self.model_ft.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.model_ft(x)
        return x
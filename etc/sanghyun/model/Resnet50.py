import torch.nn as nn
from torchvision import models
import math

def Resnet50(num_classes):
    model_ft = models.resnet50(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    nn.init.xavier_uniform_(model_ft.fc.weight)
    stdv = 1. / math.sqrt(model_ft.fc.weight.size(1))
    model_ft.fc.bias.data.uniform_(-stdv, stdv)

    # all layer freeze
    for param in model_ft.parameters() :
        param.requires_grad = False

    # FC unfreeze
    for param in model_ft.fc.parameters() :
        param.requires_grad = True

    # 학습시킬 레이어 unfreeze 해주시면 됩니다.
    for param in model_ft.layer4.parameters() :
        param.requires_grad = True

    return model_ft
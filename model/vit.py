import timm
import torch.nn as nn
import math

def ViT(pretrained=False) :
    model_ft = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

    num_ftrs = model_ft.head.in_features
    model_ft.head = nn.Linear(num_ftrs,18)
    nn.init.xavier_uniform_(model_ft.head.weight)
    stdv = 1. / math.sqrt(model_ft.head.weight.size(1))
    model_ft.head.bias.data.uniform_(-stdv, stdv)

    # all layer freeze
    for param in model_ft.parameters() :
        param.requires_grad = False

    # FC unfreeze
    for param in model_ft.head.parameters() :
        param.requires_grad = True

    # 학습시킬 레이어 unfreeze 해주시면 됩니다.
    # for param in model_ft.blocks.parameters() :
    #     param.requires_grad = True

    return model_ft
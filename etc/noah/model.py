import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm

class Resnet_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.res50 = torchvision.models.resnet50(pretrained = True)
        self.linear_layers = nn.Linear(1000,num_classes, bias = True)
    
    def forward(self,x):
        x = self.res50(x)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim = 1)


class MobilenetV2_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mobilev2 = torchvision.models.mobilenet_v2(pretrained = True)
        self.linear_layers = nn.Linear(1000,num_classes, bias = True)
    
    def forward(self,x):
        x = self.mobilev2(x)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim = 1)


class Effnet_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.effnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.linear_layers1 = nn.Linear(1000,2, bias = True)
        self.linear_layers2 = nn.Linear(1000,3, bias = True)
        self.linear_layers3 = nn.Linear(1000,3, bias = True)
        # self.output_num = [2,3,3] # gender, mask, age
    # 따로 지정해줘야만 되네..
    def forward(self,x):
        x = self.effnet(x)
        x1 = self.linear_layers1(x)
        x2 = self.linear_layers2(x)
        x3 = self.linear_layers3(x)
        x1 = F.log_softmax(x1, dim = 1)
        x2 = F.log_softmax(x2, dim = 1)
        x3 = F.log_softmax(x3, dim = 1)
        return x1,x2,x3


# 이게 안돌아가요 ㅜㅜ 
class Resnet_Multi(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained = True)
        self.mutual = nn.Sequential(*(list(self.resnet.children())[:-3])) # 여기까지는 공통
        self.sep_conv1 = nn.Sequential(*(list(self.resnet.children())[-3:-1])) # 마지막 conv block + Adaptiveavgpool
        self.sep_conv2 = nn.Sequential(*(list(self.resnet.children())[-3:-1]))
        self.sep_conv3 = nn.Sequential(*(list(self.resnet.children())[-3:-1]))
        self.linear_layers1 = nn.Linear(2048,2, bias = True)
        self.linear_layers2 = nn.Linear(2048,3, bias = True)
        self.linear_layers3 = nn.Linear(2048,3, bias = True)
        # self.output_num = [2,3,3] # gender, mask, age
    # 따로 지정해줘야만 되네..
    def forward(self,x):
        x = self.mutual(x)
        x1 = self.sep_conv1(x)
        x2 = self.sep_conv2(x)
        x3 = self.sep_conv3(x)
        x1 = self.linear_layers1(x1)
        x2 = self.linear_layers2(x2)
        x3 = self.linear_layers3(x3)
        x1 = F.log_softmax(x1, dim = 1)
        x2 = F.log_softmax(x2, dim = 1)
        x3 = F.log_softmax(x3, dim = 1)
        # output = [F.log_softmax(x1, dim = 1),F.log_softmax(x2, dim = 1),F.log_softmax(x3, dim = 1)]
        return x1, x2, x3

class Resnet_Multi(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained = True)
        # self.mutual = nn.Sequential(*(list(self.resnet.children())[:-1])) # 여기까지는 공통
        self.linear_layers1 = nn.Linear(1000,2, bias = True)
        self.linear_layers2 = nn.Linear(1000,3, bias = True)
        self.linear_layers3 = nn.Linear(1000,3, bias = True)
        # self.output_num = [2,3,3] # gender, mask, age
    def forward(self,x):
        x = self.resnet(x)
        x1 = self.linear_layers1(x)
        x2 = self.linear_layers2(x)
        x3 = self.linear_layers3(x)
        x1 = F.log_softmax(x1, dim = 1)
        x2 = F.log_softmax(x2, dim = 1)
        x3 = F.log_softmax(x3, dim = 1)
        # output = [F.log_softmax(x1, dim = 1),F.log_softmax(x2, dim = 1),F.log_softmax(x3, dim = 1)]
        return x1,x2,x3

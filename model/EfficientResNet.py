import torch 
import torch.nn as nn
import math
from efficientnet_pytorch import EfficientNet
from torchvision import models

class CustomEfficientNet(nn.Module):
    def __init__(self, device, num_features = 18, dropout_p = 0.6):
        super().__init__()
        
        if device is None:
            self.effnet = EfficientNet.from_pretrained("efficientnet-b2")
        else:
            self.effnet = EfficientNet.from_pretrained("efficientnet-b2").to(device)
        self.pool   = nn.AdaptiveAvgPool2d(output_size = 1)
        self.drop   = nn.Dropout2d(dropout_p)
        self.dense  = nn.Linear(1408, num_features)

        for param in self.effnet.parameters():
            param.require_grads = False
            # freezing all params
        
    def forward(self, x):
        # x.shape == torch.Size([B, 3, 224, 224])
        x = self.effnet.extract_features(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.flatten(start_dim = 1)
        # print(x.shape)
        x = self.dense(x)
        return x

class Customresnet50(nn.Module) :
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = models.resnet50(pretrained=True)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,num_classes)
        
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1. / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
    
    def forward(self,x) :
        output = self.model(x)
        
        return output

# Use two pre-trained model to ensemble
class EfficientResnet(nn.Module):
    def __init__(self, efficientnet_path, resnet_path , num_features = 18, dropout_p = 0.6 , device = None, ):
        super(EfficientResnet, self).__init__()

        self.effnet   = CustomEfficientNet(device, num_features, dropout_p)
        self.effnet.load_state_dict(torch.load(efficientnet_path))
        self.resnet = Customresnet50(num_features)
        self.resnet.load_state_dict(torch.load(resnet_path))
    def forward(self, x):

        self.effnet_output = self.effnet(x.clone())
        self.resnet_output = self.resnet(x.clone())
        
        pred = torch.add(self.effnet_output,self.resnet_output)
        # label 만 받고 싶으면 아래 주석 해제                   
        # pred = pred.argmax(dim=-1) 
        return pred

if __name__ == "__main__" :
    EFF_PATH = ""
    RES_PATH = ""

    effresnet = EfficientResnet(EFF_PATH,RES_PATH)

    effresnet.eval()
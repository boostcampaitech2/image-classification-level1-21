import torch.nn as nn
from torchvision import models

class PretrainedResnet(nn.Module):
    def __init__(self, xdim, ksize, cdims, hdims, ydim, name='cnn', USE_BATCHNORM=False):
        super(PretrainedResnet, self).__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM

        self.layers = []
        prev_cdim = self.xdim[0]
        self.layers.append(models.resnet18(pretrained=True))
        for cdim in self.cdims:
            self.layers.append(
                nn.Conv2d(in_channels=prev_cdim,
                          out_channels=cdim,
                          kernel_size=self.ksize,
                          stride=(1,1),
                          padding=(0,0)))
            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim))
            self.layers.append(nn.ReLU(True))
            prev_cdim = cdim

        self.layers.append(nn.Flatten())
        prev_hdim = 1000
        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim, bias=True))
            self.layers.append(nn.ReLU(True))
            prev_hdim = hdim

        self.layers.append(nn.Linear(prev_hdim, self.ydim, bias=True))
        
        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)
    
    def forward(self, x):
        return self.net(x)
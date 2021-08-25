from model.pretrained_resnet import *
from resnet50 import Resnet50
from vit import ViT

class ModelList():
    def __init__(self):
        return

    @classmethod
    def parse_model(self, model_name):
        if model_name == "PretrainedResnet":
            return PretrainedResnet

        if model_name == "Resnet50" :
            return Resnet50

        if model_name == "ViT" :
            return ViT
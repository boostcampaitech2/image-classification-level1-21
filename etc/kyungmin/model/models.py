from model.resnet50 import *
from model.vit import *

class ModelList():
    def __init__(self):
        return

    @classmethod
    def parse_model(self, model_name):
        if model_name == "Resnet50":
            return Resnet50
        
        elif model_name == "ViT":
            return ViT

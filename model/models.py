from model.pretrained_resnet import *

class ModelList():
    def __init__(self):
        return

    @classmethod
    def parse_model(self, model_name):
        if model_name == "PretrainedResnet":
            return PretrainedResnet
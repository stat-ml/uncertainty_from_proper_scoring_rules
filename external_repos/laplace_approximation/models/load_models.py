import torch.nn

from .constants import *
from .resnet import *
from .vgg import *


def get_model(model_name: str, n_classes: int = 10) -> torch.nn.Module:
    if model_name == ModelName.RESNET18:
        return ResNet18(n_classes=n_classes)
    else:
        raise ValueError(
        f"{model_name} --  no such neural network is available. ",
        f"Available options are: {[element.value for element in ModelName]}")

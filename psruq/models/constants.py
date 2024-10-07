from enum import Enum


class ModelName(Enum):
    RESNET18 = "resnet18"
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"


class ModelSource(Enum):
    OUR_MODELS = "our_models"
    TORCH_UNCERTAINTY = "torch_uncertainty"

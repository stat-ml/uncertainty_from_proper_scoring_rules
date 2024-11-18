from enum import Enum


class ModelName(Enum):
    RESNET18 = "resnet18"
    RESNET18_DROPOUT = "resnet18_dropout"
    RESNET18_FLIPOUT = "resnet18_flipout"
    RESNET18_DUQ = "resnet18_duq"
    VGG11 = "vgg11"
    VGG13 = "vgg13"
    VGG16 = "vgg16"
    VGG19 = "vgg19"

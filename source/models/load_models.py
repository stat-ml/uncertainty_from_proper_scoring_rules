import source.models.constants
import source.models.resnet
import source.models.vgg
import torch.nn


def get_model(model_name: str, n_classes: int = 10) -> torch.nn.Module:
    match source.models.constants.ModelName(model_name):
        case source.models.constants.ModelName.RESNET18:
            return source.models.resnet.ResNet18(n_classes=n_classes)
        case source.models.constants.ModelName.VGG11:
            return source.models.vgg.VGG(vgg_name="VGG11", n_classes=n_classes)
        case source.models.constants.ModelName.VGG13:
            return source.models.vgg.VGG(vgg_name="VGG13", n_classes=n_classes)
        case source.models.constants.ModelName.VGG16:
            return source.models.vgg.VGG(vgg_name="VGG16", n_classes=n_classes)
        case source.models.constants.ModelName.VGG19:
            return source.models.vgg.VGG(vgg_name="VGG19", n_classes=n_classes)
        case _:
            raise ValueError(
                f"{model_name} --  no such neural network is available. ",
                f"Available options are: {[element.value for element in source.models.constants.ModelName]}",
            )

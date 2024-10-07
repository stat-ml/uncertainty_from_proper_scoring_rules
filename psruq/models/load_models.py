import os

import torch.nn

import psruq.models.constants
import psruq.models.resnet
import psruq.models.vgg
from torch_uncertainty.models.resnet import resnet
from safetensors.torch import load_file
from psruq.models.constants import ModelSource
from psruq.source.path_utils import (
    make_load_path,
    make_logits_path,
    make_model_load_path,
)


def get_model(model_name: str, n_classes: int = 10) -> torch.nn.Module:
    match psruq.models.constants.ModelName(model_name):
        case psruq.models.constants.ModelName.RESNET18:
            return psruq.models.resnet.ResNet18(n_classes=n_classes)
        case psruq.models.constants.ModelName.VGG11:
            return psruq.models.vgg.VGG(vgg_name="VGG11", n_classes=n_classes)
        case psruq.models.constants.ModelName.VGG13:
            return psruq.models.vgg.VGG(vgg_name="VGG13", n_classes=n_classes)
        case psruq.models.constants.ModelName.VGG16:
            return psruq.models.vgg.VGG(vgg_name="VGG16", n_classes=n_classes)
        case psruq.models.constants.ModelName.VGG19:
            return psruq.models.vgg.VGG(vgg_name="VGG19", n_classes=n_classes)
        case _:
            raise ValueError(
                f"{model_name} --  no such neural network is available. ",
                f"Available options are: {[element.value for element in psruq.models.constants.ModelName]}",
            )


def load_model_checkpoint(
    architecture: str, path: str, n_classes: int, device
) -> torch.nn.Module:
    """Load trained model checkpoint

    Args:
        architecture (str): _description_
        path (str): _description_
        n_classes (int): _description_
        device (_type_): _description_

    Returns:
        nn.Module: _description_
    """
    checkpoint = torch.load(path, map_location=device)
    model = get_model(model_name=architecture, n_classes=n_classes)
    model.load_state_dict(checkpoint["net"])
    return model


def load_model_from_source(
    model_source: ModelSource,
    architecture: str,
    training_dataset_name: str,
    extraction_dataset_name: str,
    loss_function_name: str,
    n_classes: int,
    model_id: int,
    device: str,
    severity: int | None,
):
    match model_source:
        case ModelSource.OUR_MODELS.value:
            load_path = make_load_path(
                architecture=architecture,
                dataset_name=training_dataset_name,
                loss_function_name=loss_function_name,
                model_id=model_id,
            )
            checkpoint_path = os.path.join(load_path, "ckpt.pth")
            logits_path = make_logits_path(
                model_id=model_id,
                training_dataset_name=training_dataset_name,
                extraction_dataset_name=extraction_dataset_name,
                severity=severity,
                model_source=model_source,
                architecture=architecture,
                loss_function_name=loss_function_name,
            )

            # if os.path.exists(logits_path):
            #     print("Embeddings are already extracted! Skipping...")
            #     return
            model = load_model_checkpoint(
                architecture=architecture,
                path=checkpoint_path,
                device=device,
                n_classes=n_classes,
            )
        case ModelSource.TORCH_UNCERTAINTY.value:
            model_path = make_model_load_path(
                version=model_id, training_dataset=training_dataset_name
            )
            model = resnet(
                num_classes=n_classes,
                in_channels=3,
                arch=18,
                style="cifar",
                conv_bias=False,
            )
            state_dict = load_file(model_path)
            model.load_state_dict(state_dict=state_dict)

    return model

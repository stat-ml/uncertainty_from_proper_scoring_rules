import os

import torch.nn

import source.models.constants
import source.models.resnet
import source.models.vgg
from torch_uncertainty_models.source.std_loading import cust_load_model
from source.models.constants import ModelSource
from source.source.path_utils import (
    make_load_path,
    make_logits_path,
    make_model_load_path,
)


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

            if os.path.exists(logits_path):
                print("Embeddings are already extracted! Skipping...")
                return
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
            model = cust_load_model(
                style=(
                    "cifar" if training_dataset_name.startswith("cifar") else "imagenet"
                ),
                num_classes=n_classes,
                arch=18,
                path=model_path,
                conv_bias=False,
            )
    return model

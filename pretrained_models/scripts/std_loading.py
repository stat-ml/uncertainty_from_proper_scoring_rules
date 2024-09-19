import os
from pathlib import Path

from safetensors.torch import load_file

from scripts.resnet import resnet


def cust_load_model(
    version: int,
    arch: int,
    num_classes: int,
    path_to_folder_with_models: str,
    style: str,
    conv_bias: bool,
):
    """Load the model corresponding to the given version."""
    model = resnet(
        num_classes=num_classes,
        in_channels=3,
        arch=arch,
        style=style,
        conv_bias=conv_bias,
    )
    path = os.path.join(path_to_folder_with_models, f"version_{version}.safetensors")

    if not os.path.exists(path):
        raise ValueError("File does not exist")

    state_dict = load_file(path)
    model.load_state_dict(state_dict=state_dict)
    return model

import os
from safetensors.torch import load_file
from torch_uncertainty_models.source.resnet import resnet


def cust_load_model(
    arch: int,
    num_classes: int,
    path: str,
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

    if not os.path.exists(path):
        raise ValueError("File does not exist")

    state_dict = load_file(path)
    model.load_state_dict(state_dict=state_dict)
    return model

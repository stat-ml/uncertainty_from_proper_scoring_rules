import os
from pathlib import Path

from scripts.std_loading import cust_load_model

ROOT_PATH = Path(__file__).resolve().parent

version = 10

model_cifar10 = cust_load_model(
    style="cifar",
    num_classes=10,
    arch=18,
    path_to_folder_with_models=os.path.join(
        ROOT_PATH, "models", "cifar10-resnet18-0-1023"
    ),
    version=version,
    conv_bias=False,
)

model_cifar100 = cust_load_model(
    style="cifar",
    num_classes=100,
    arch=18,
    path_to_folder_with_models=os.path.join(
        ROOT_PATH, "models", "cifar100-resnet18-0-1023"
    ),
    version=version,
    conv_bias=False,
)

model_tiny_imagenet = cust_load_model(
    style="imagenet",
    num_classes=200,
    arch=18,
    path_to_folder_with_models=os.path.join(
        ROOT_PATH, "models", "tiny-imagenet-resnet18-0-1023"
    ),
    version=version,
    conv_bias=False,
)

print("SUCCESS!")

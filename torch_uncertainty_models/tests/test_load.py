import os
from pathlib import Path

from torch_uncertainty_models.source.std_loading import cust_load_model
from torch_uncertainty_models.source.utils import ROOT_PATH, make_model_load_path


version = 10

path = make_model_load_path(version=version, training_dataset="cifar10")

model_cifar10 = cust_load_model(
    style="cifar",
    num_classes=10,
    arch=18,
    path=path,
    conv_bias=False,
)

path = make_model_load_path(version=version, training_dataset="cifar100")
model_cifar100 = cust_load_model(
    style="cifar",
    num_classes=100,
    arch=18,
    path=path,
    conv_bias=False,
)

path = make_model_load_path(version=version, training_dataset="tiny_imagenet")
model_tiny_imagenet = cust_load_model(
    style="imagenet",
    num_classes=200,
    arch=18,
    path=path,
    conv_bias=False,
)

print("SUCCESS!")

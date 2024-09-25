from source.datasets.constants import DatasetName
from source.datasets.datasets import get_dataset_class_instance
from source.datasets.loaders import get_dataloaders
from source.datasets.transforms import (
    get_cifar10_transforms,
    get_cifar100_transforms,
    get_transforms,
)

__all__ = [
    "DatasetName",
    "get_dataloaders",
    "get_dataset_class_instance",
    "get_transforms",
    "get_cifar100_transforms",
    "get_cifar10_transforms",
]


def show_progress(block_num, block_size, total_size):
    print(round(block_num * block_size / total_size * 100, 2), end="\r")

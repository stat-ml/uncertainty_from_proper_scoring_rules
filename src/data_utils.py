from external_repos.pytorch_cifar10.utils import (
    get_transforms as get_cifar10_transforms,
)

from external_repos.pytorch_cifar100.utils import (
    get_transforms as get_cifar100_transforms,
)
import os
import torch
import torchvision
from torchvision import transforms


def load_dataloader_for_extraction(
        training_dataset_name: str,
        extraction_dataset_name: str,
) -> torch.utils.data.DataLoader:
    """The function returns dataloader for extracting embeddings.
    It takes into account proper transformations from training dataset,
    and performs corresponding normalization.

    Args:
        training_dataset_name (str): name of the dataset, used in training
        extraction_dataset_name (str): name of the dataset,
                            we want extract embeddings from

    Returns:
        torch.utils.data.DataLoader: correspinding test loader
    """
    if training_dataset_name == 'cifar10':
        _, ind_transforms = get_cifar10_transforms()
    elif training_dataset_name == 'cifar100':
        _, ind_transforms = get_cifar100_transforms()
    else:
        ValueError("No such dataset available")

    if extraction_dataset_name == 'lsun':
        if training_dataset_name in ['cifar10', 'cifar100', 'svhn']:
            ind_transforms = transforms.Compose(
                [transforms.Resize((32, 32))] + ind_transforms.transforms)
        dataset = torchvision.datasets.LSUN(
            root='./data',
            classes='test', transform=ind_transforms
        )

    elif extraction_dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=ind_transforms
        )

    elif extraction_dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=ind_transforms
        )
    elif extraction_dataset_name == 'svhn':
        dataset = torchvision.datasets.SVHN(
            root='./data',
            split='test',
            download=True,
            transform=ind_transforms
        )
    else:
        ValueError("No such dataset available")

    testloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=100
    )
    return testloader

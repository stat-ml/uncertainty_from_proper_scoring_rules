import logging
from typing import Optional

import torch.utils.data
import torchvision

from .constants import DatasetName
from .transforms import get_cifar10_transforms, get_cifar100_transforms

LOGGER = logging.getLogger(__name__)

def get_dataloaders(dataset: str, missed_label: Optional[int] = None):
    # Data
    LOGGER.info(f'Preparing dataset {dataset.__str__()}')

    if dataset == DatasetName.CIFAR10_ONE_BATCH:
        transform_train, transform_test = get_cifar10_transforms()
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transform_train)
        
        trainset = torch.utils.data.Subset(
            trainset, list(range(128)))

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transform_test)
        
        testset = torch.utils.data.Subset(
            testset, list(range(128)))

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False)
        
    elif dataset == DatasetName.CIFAR10:
        transform_train, transform_test = get_cifar10_transforms()
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transform_train)
    
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transform_test)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False)

    elif dataset == DatasetName.CIFAR100:
        transform_train, transform_test = get_cifar100_transforms()
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,
            transform=transform_train)
        
        trainset = torch.utils.data.Subset(
            trainset, list(range(128)))

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,
            transform=transform_test)
        
        testset = torch.utils.data.Subset(
            testset, list(range(128)))

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False)
    else:
        raise ValueError(
            f"{dataset} --  no such dataset available. ",
            f"Available options are: {[element.value for element in DatasetName]}")

    return trainloader, testloader

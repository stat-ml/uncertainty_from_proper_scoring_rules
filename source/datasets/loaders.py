import logging
from ctypes import ArgumentError
from typing import Optional

import torch.utils.data

import source.datasets.constants
import source.datasets.datasets
import source.datasets.transforms

LOGGER = logging.getLogger(__name__)


def get_dataloaders(
    dataset: str,
    *dataloader_args,
    missed_label: Optional[int] = None,
    **dataloader_kwargs,
):
    # Data
    LOGGER.info(f"Preparing dataset {dataset.__str__()}")
    dataset_class = source.datasets.datasets.get_dataset_class_instance(
        dataset=dataset, missed_label=missed_label
    )
    transform_train, transform_test = source.datasets.transforms.get_transforms(
        dataset=dataset,
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_class(
            *dataloader_args,
            root="./data",
            train=True,
            download=True,
            transform=transform_train,
            **dataloader_kwargs,
        ),
        batch_size=128,
        shuffle=False,
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_class(
            *dataloader_args,
            root="./data",
            train=False,
            download=True,
            transform=transform_test,
            **dataloader_kwargs,
        ),
        batch_size=128,
        shuffle=False,
    )

    return trainloader, testloader

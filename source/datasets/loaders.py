import logging
import os
from typing import Optional

import torch.utils.data
import torchvision

import source.datasets.constants
import source.datasets.datasets
import source.datasets.transforms
from source.source.path_config import REPOSITORY_ROOT

LOGGER = logging.getLogger(__name__)


def get_dataloaders(
    dataset: str,
    missed_label: Optional[int] = None,
    severity: Optional[int] = None,
    transform_train: Optional[torchvision.transforms.Compose] = None,
    transform_test: Optional[torchvision.transforms.Compose] = None,
):
    # Data
    LOGGER.info(f"Preparing dataset {dataset.__str__()}")
    root_path = os.path.join(REPOSITORY_ROOT, "datasets")
    dataset_class = source.datasets.datasets.get_dataset_class_instance(
        dataset=dataset, missed_label=missed_label, severity=severity
    )
    if transform_train is None or transform_test is None:
        transform_train, transform_test = source.datasets.transforms.get_transforms(
            dataset=dataset
        )

    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_class(
            root=root_path,
            train=True,
            download=True,
            transform=transform_train,
        ),
        batch_size=128,
        shuffle=True,
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_class(
            root=root_path,
            train=False,
            download=True,
            transform=transform_test,
        ),
        batch_size=128,
        shuffle=True,
    )

    return trainloader, testloader

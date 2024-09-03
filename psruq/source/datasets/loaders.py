import logging
from typing import Optional

import source.datasets.constants
import source.datasets.datasets
import source.datasets.transforms
import torch.utils.data

LOGGER = logging.getLogger(__name__)

def get_dataloaders(dataset: str, missed_label: Optional[int] = None):
    # Data
    LOGGER.info(f'Preparing dataset {dataset.__str__()}')
    dataset_class = source.datasets.datasets.get_dataset_class_instance(
        dataset=dataset, missed_label=missed_label)
    transform_train, transform_test = source.datasets.transforms.get_transforms(dataset=dataset)
    
    trainloader = torch.utils.data.DataLoader(
        dataset=dataset_class(
            root='./data',
            train=True,
            download=True,
            transform=transform_train
        ),
        batch_size=128,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        dataset=dataset_class(
            root='./data',
            train=False,
            download=True,
            transform=transform_test
        ),
        batch_size=128,
        shuffle=True
    )
    
    return trainloader, testloader

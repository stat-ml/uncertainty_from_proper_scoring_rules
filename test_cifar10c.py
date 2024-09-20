import os

import torch_uncertainty.datasets.classification as torch_uncertainty_datasets
from torch.utils.data import DataLoader

from source.datasets.constants import DatasetName
from source.datasets.transforms import get_transforms
from source.source.path_config import REPOSITORY_ROOT

for i in range(1, 6):
    dataset = torch_uncertainty_datasets.CIFAR10C(
        root=os.path.join(REPOSITORY_ROOT, "datasets"),
        transform=get_transforms(dataset=DatasetName.CIFAR10C.value)[1],
        subset="all",
        severity=i,
        download=True,
    )

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
    )

    print(next(iter(trainloader))[0].shape)

print("hey!")

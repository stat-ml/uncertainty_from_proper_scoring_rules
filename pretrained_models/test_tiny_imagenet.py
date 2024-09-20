import os

import torch_uncertainty.datasets.classification as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from source.source.path_config import REPOSITORY_ROOT

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


ti_dataset = datasets.TinyImageNet(
    root=os.path.join(REPOSITORY_ROOT, "datasets"),
    split="train",
    transform=transform_test,
)

dl = DataLoader(ti_dataset, batch_size=64, shuffle=False)

print(next(iter(dl))[0].shape, next(iter(dl))[1].shape)

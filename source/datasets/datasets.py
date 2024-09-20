import logging
import random

import numpy as np
import torch.utils.data
import torch_uncertainty.datasets.classification as torch_uncertainty_datasets
import torchvision

import source.datasets.constants
import source.datasets.transforms

LOGGER = logging.getLogger(__name__)


def get_dataset_class_instance(dataset: str, missed_label: int | None = None):
    match source.datasets.constants.DatasetName(dataset):
        case source.datasets.constants.DatasetName.CIFAR10_ONE_BATCH:
            return lambda *args, **kwargs: torch.utils.data.Subset(
                dataset=torchvision.datasets.CIFAR10(*args, **kwargs),
                indices=list(range(128)),
            )

        case source.datasets.constants.DatasetName.CIFAR10:
            return torchvision.datasets.CIFAR10

        case source.datasets.constants.DatasetName.CIFAR10_BLURRED:
            return torchvision.datasets.CIFAR10

        case source.datasets.constants.DatasetName.CIFAR100:
            return torchvision.datasets.CIFAR100

        case source.datasets.constants.DatasetName.CIFAR100_BLURRED:
            return torchvision.datasets.CIFAR100

        case source.datasets.constants.DatasetName.CIFAR10_MISSED_LABEL:
            if missed_label is None:
                error_message = (
                    f"For {source.datasets.constants.DatasetName.CIFAR10_MISSED_LABEL}"
                    " missed label should be precised."
                )
                raise RuntimeError(error_message)
            return lambda *args, **kwargs: CIFAR10MissedLabels(
                *args,
                **kwargs,
                missed_label=missed_label,
            )

        case source.datasets.constants.DatasetName.CIFAR10_NOISY_LABEL:
            return CIFAR10NoisyLabels

        case source.datasets.constants.DatasetName.SVHN:
            return lambda root, train, download, transform: torchvision.datasets.SVHN(
                split="train" if train else "test",
                root=root,
                download=download,
                transform=transform,
            )

        case source.datasets.constants.DatasetName.TINY_IMAGENET:
            return lambda root, train, download, transform: torch_uncertainty_datasets.TinyImageNet(
                root=root,
                split="train" if train else "val",
                transform=transform,
            )

        case _:
            raise ValueError(
                f"{dataset} --  no such dataset available. ",
                f"Available options are: {[element.value for element in source.datasets.constants.DatasetName]}",
            )


class CIFAR10MissedLabels(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        missed_label: int,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.target_transform = target_transform
        self.missed_label = missed_label

    def __getitem__(self, index):
        image, label = self.dataset[index]

        while label == self.missed_label:
            new_index = np.random.randint(low=0, high=len(self.dataset))
            image, label = self.dataset[new_index]

        # Apply any target transformations (if any)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        # Return the length of the original CIFAR-10 dataset
        return len(self.dataset)


class CIFAR10NoisyLabels(torch.utils.data.Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=transform
        )
        self.target_transform = target_transform
        # Pairs of labels to be swapped randomly
        self.label_pairs = {1: 7, 7: 1, 3: 8, 8: 3, 2: 5, 5: 2}

    def __getitem__(self, index):
        # Get an item from the original CIFAR-10 dataset
        image, label = self.dataset[index]

        # If the label is part of a pair,
        # randomly assign one of the two paired labels
        if label in self.label_pairs:
            label = random.choice([label, self.label_pairs[label]])

        # Apply any target transformations (if any)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        # Return the length of the original CIFAR-10 dataset
        return len(self.dataset)

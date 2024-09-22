"""Some helper functions for PyTorch, including:
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
"""

import os
import sys
import time

import torch.nn as nn
import torch.nn.init as init

from .models import VGG as VGG19_Cifar10, ResNet18
from typing import Optional
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


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


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    return transform_train, transform_test


def get_model(
    architecture: str,
    n_classes: int,
) -> nn.Module:
    match architecture:
        case "vgg":
            if n_classes == 10:
                net = VGG19_Cifar10("VGG19", n_classes)
            elif n_classes == 100:
                sys.path.insert(0, "external_repos/pytorch_cifar100/models")
                from .vgg import vgg19_bn as VGG19_Cifar100

                net = VGG19_Cifar100()
            else:
                raise ValueError(f"Wrong number of classes: {n_classes}")
        case "resnet18":
            net = ResNet18(n_classes)
        case _:
            print("No such architecture")
            raise ValueError()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        # net = SimpleDLA()
    return net


def get_dataloaders(dataset: str, missed_label: Optional[int] = None):
    # Data
    print("==> Preparing data..")
    transform_train, transform_test = get_transforms()

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    elif dataset == "noisy_cifar10":
        trainset = CIFAR10NoisyLabels(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    elif dataset == "svhn":
        trainset = torchvision.datasets.SVHN(
            split="train", root="./data", download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True
        )

        testset = torchvision.datasets.SVHN(
            split="test", root="./data", download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    elif dataset == "missed_class_cifar10":
        trainset = CIFAR10MissedLabels(
            root="./data",
            train=True,
            download=True,
            transform=transform_train,
            missed_label=missed_label,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=128,
            shuffle=True,
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    else:
        raise ValueError(f"{dataset} --  no such dataset available.")

    return trainloader, testloader


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)
except:
    term_width = 167

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f

"""helper function

author baiyu
"""

import os
import sys
import re
import datetime

import numpy
import random
import torch
import numpy as np

from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from conf import settings


class CIFAR100MissedLabels(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        missed_label: int,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.dataset = torchvision.datasets.CIFAR100(
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


class CIFAR100NoisyLabels(torch.utils.data.Dataset):
    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=transform
        )
        self.target_transform = target_transform
        # Pairs of labels to be swapped randomly
        self.label_pairs = {
            1: 7,
            7: 1,
            3: 8,
            8: 3,
            2: 5,
            5: 2,
            10: 20,
            20: 10,
            40: 50,
            50: 40,
            90: 99,
            99: 90,
            25: 75,
            75: 25,
            17: 71,
            71: 17,
            13: 31,
            31: 13,
            24: 42,
            42: 24,
        }

    def __getitem__(self, index):
        # Get an item from the original CIFAR-100 dataset
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
        # Return the length of the original CIFAR-100 dataset
        return len(self.dataset)


def get_network(args):
    """return given network"""

    if args.architecture == "vgg16":
        from models.vgg import vgg16_bn

        net = vgg16_bn()
    elif args.architecture == "vgg13":
        from models.vgg import vgg13_bn

        net = vgg13_bn()
    elif args.architecture == "vgg11":
        from models.vgg import vgg11_bn

        net = vgg11_bn()
    elif args.architecture == "vgg":
        from models.vgg import vgg19_bn

        net = vgg19_bn()
    elif args.architecture == "densenet121":
        from models.densenet import densenet121

        net = densenet121()
    elif args.architecture == "densenet161":
        from models.densenet import densenet161

        net = densenet161()
    elif args.architecture == "densenet169":
        from models.densenet import densenet169

        net = densenet169()
    elif args.architecture == "densenet201":
        from models.densenet import densenet201

        net = densenet201()
    elif args.architecture == "googlenet":
        from models.googlenet import googlenet

        net = googlenet()
    elif args.architecture == "inceptionv3":
        from models.inceptionv3 import inceptionv3

        net = inceptionv3()
    elif args.architecture == "inceptionv4":
        from models.inceptionv4 import inceptionv4

        net = inceptionv4()
    elif args.architecture == "inceptionresnetv2":
        from models.inceptionv4 import inception_resnet_v2

        net = inception_resnet_v2()
    elif args.architecture == "xception":
        from models.xception import xception

        net = xception()
    elif args.architecture == "resnet18":
        from models.resnet import resnet18

        net = resnet18()
    elif args.architecture == "resnet34":
        from models.resnet import resnet34

        net = resnet34()
    elif args.architecture == "resnet50":
        from models.resnet import resnet50

        net = resnet50()
    elif args.architecture == "resnet101":
        from models.resnet import resnet101

        net = resnet101()
    elif args.architecture == "resnet152":
        from models.resnet import resnet152

        net = resnet152()
    elif args.architecture == "preactresnet18":
        from models.preactresnet import preactresnet18

        net = preactresnet18()
    elif args.architecture == "preactresnet34":
        from models.preactresnet import preactresnet34

        net = preactresnet34()
    elif args.architecture == "preactresnet50":
        from models.preactresnet import preactresnet50

        net = preactresnet50()
    elif args.architecture == "preactresnet101":
        from models.preactresnet import preactresnet101

        net = preactresnet101()
    elif args.architecture == "preactresnet152":
        from models.preactresnet import preactresnet152

        net = preactresnet152()
    elif args.architecture == "resnext50":
        from models.resnext import resnext50

        net = resnext50()
    elif args.architecture == "resnext101":
        from models.resnext import resnext101

        net = resnext101()
    elif args.architecture == "resnext152":
        from models.resnext import resnext152

        net = resnext152()
    elif args.architecture == "shufflenet":
        from models.shufflenet import shufflenet

        net = shufflenet()
    elif args.architecture == "shufflenetv2":
        from models.shufflenetv2 import shufflenetv2

        net = shufflenetv2()
    elif args.architecture == "squeezenet":
        from models.squeezenet import squeezenet

        net = squeezenet()
    elif args.architecture == "mobilenet":
        from models.mobilenet import mobilenet

        net = mobilenet()
    elif args.architecture == "mobilenetv2":
        from models.mobilenetv2 import mobilenetv2

        net = mobilenetv2()
    elif args.architecture == "nasnet":
        from models.nasnet import nasnet

        net = nasnet()
    elif args.architecture == "attention56":
        from models.attention import attention56

        net = attention56()
    elif args.architecture == "attention92":
        from models.attention import attention92

        net = attention92()
    elif args.architecture == "seresnet18":
        from models.senet import seresnet18

        net = seresnet18()
    elif args.architecture == "seresnet34":
        from models.senet import seresnet34

        net = seresnet34()
    elif args.architecture == "seresnet50":
        from models.senet import seresnet50

        net = seresnet50()
    elif args.architecture == "seresnet101":
        from models.senet import seresnet101

        net = seresnet101()
    elif args.architecture == "seresnet152":
        from models.senet import seresnet152

        net = seresnet152()
    elif args.architecture == "wideresnet":
        from models.wideresidual import wideresnet

        net = wideresnet()
    elif args.architecture == "stochasticdepth18":
        from models.stochasticdepth import stochastic_depth_resnet18

        net = stochastic_depth_resnet18()
    elif args.architecture == "stochasticdepth34":
        from models.stochasticdepth import stochastic_depth_resnet34

        net = stochastic_depth_resnet34()
    elif args.architecture == "stochasticdepth50":
        from models.stochasticdepth import stochastic_depth_resnet50

        net = stochastic_depth_resnet50()
    elif args.architecture == "stochasticdepth101":
        from models.stochasticdepth import stochastic_depth_resnet101

        net = stochastic_depth_resnet101()

    else:
        print("the network name you have entered is not supported yet")
        sys.exit()

    if args.gpu:  # use_gpu
        net = net.cuda()

    return net


def get_transforms(
    mean=settings.CIFAR100_TRAIN_MEAN,
    std=settings.CIFAR100_TRAIN_STD,
) -> tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    return transform_train, transform_test


def get_training_dataloader(
    mean, std, dataset: str, batch_size=16, num_workers=2, shuffle=True
):
    """return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train, _ = get_transforms(mean=mean, std=std)

    if dataset == "cifar100":
        training_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset == "noisy_cifar100":
        training_dataset = CIFAR100NoisyLabels(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset == "missed_class_cifar100":
        training_dataset = CIFAR100MissedLabels(
            root="./data", train=True, download=True, transform=transform_train
        )
    else:
        raise ValueError("No such dataset!")

    training_loader = DataLoader(
        training_dataset,
        shuffle=shuffle,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    return training_loader


def get_test_dataloader(
    mean, std, dataset: str, batch_size=16, num_workers=2, shuffle=True
):
    """return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    _, transform_test = get_transforms(mean=mean, std=std)

    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size
    )

    return test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))]
    )
    data_g = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))]
    )
    data_b = numpy.dstack(
        [cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))]
    )
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def most_recent_folder(net_weights, fmt):
    """
    return most recent created folder under net_weights
    if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ""

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
    return most recent created weights file
    if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ""

    regex_str = r"([A-Za-z0-9]+)-([0-9]+)-(regular|best)"

    # sort files by epoch
    weight_files = sorted(
        weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1])
    )

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception("no recent weights were found")
    resume_epoch = int(weight_file.split("-")[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
    return the best acc .pth file in given folder, if no
    best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ""

    regex_str = r"([A-Za-z0-9]+)-([0-9]+)-(regular|best)"
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == "best"]
    if len(best_files) == 0:
        return ""

    best_files = sorted(
        best_files, key=lambda w: int(re.search(regex_str, w).groups()[1])
    )
    return best_files[-1]

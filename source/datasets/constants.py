from enum import Enum


class DatasetName(Enum):
    CIFAR10_ONE_BATCH = "cifar10_one_batch"
    CIFAR10 = "cifar10"
    CIFAR10C = "cifar10c"
    CIFAR10_BLURRED = "blurred_cifar10"
    CIFAR100 = "cifar100"
    CIFAR100C = "cifar100c"
    CIFAR100_BLURRED = "blurred_cifar100"
    CIFAR100_NOISY_LABEL = "cifar100_noisy_label"
    CIFAR10_MISSED_LABEL = "cifar10_missed_label"
    CIFAR10_NOISY_LABEL = "cifar10_noisy_label"
    SVHN = "svhn"
    TINY_IMAGENET = "tiny_imagenet"
    IMAGENET_R = "imagenet_r"
    IMAGENET_C = "imagenet_c"
    IMAGENET_A = "imagenet_a"
    IMAGENET_O = "imagenet_o"

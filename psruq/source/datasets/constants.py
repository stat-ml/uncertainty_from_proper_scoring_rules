from enum import Enum


class DatasetName(Enum):
    CIFAR10_ONE_BATCH = "cifar10_one_batch"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"
    CIFAR10_MISSED_LABEL = "cifar10_missed_label"
    CIFAR10_NOISY_LABEL = "cifar10_noisy_label"
    SVHN = "svhn"
    
    
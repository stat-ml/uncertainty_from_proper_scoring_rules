from enum import StrEnum


class DatasetName(StrEnum):
    CIFAR10 = 'cifar10'
    NOISY_CIFAR10 = 'noisy_cifar10'
    NOISY_CIFAR100 = 'noisy_cifar100'
    MISSED_CLASS_CIFAR10 = 'missed_class_cifar10'
    SVHN = 'svhn'
    CIFAR100 ='cifar100'
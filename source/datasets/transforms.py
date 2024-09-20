import source.datasets.constants
import torchvision.transforms as transforms


def get_transforms(dataset: str):
    match source.datasets.constants.DatasetName(dataset):
        case source.datasets.constants.DatasetName.CIFAR10_ONE_BATCH:
            return get_cifar10_transforms()
        case source.datasets.constants.DatasetName.CIFAR10:
            return get_cifar10_transforms()
        case source.datasets.constants.DatasetName.CIFAR100:
            return get_cifar100_transforms()
        case source.datasets.constants.DatasetName.CIFAR10_NOISY_LABEL:
            return get_cifar10_transforms()
        case source.datasets.constants.DatasetName.CIFAR10_MISSED_LABEL:
            return get_cifar10_transforms()
        case source.datasets.constants.DatasetName.SVHN:
            return get_cifar10_transforms()
        case source.datasets.constants.DatasetName.TINY_IMAGENET:
            return get_tiny_imagenet_transforms()
        case _:
            raise ValueError(
                f"{dataset} --  no such dataset available. ",
                f"Available options are: {[element.value for element in source.datasets.constants.DatasetName]}",
            )


def get_cifar100_transforms() -> tuple[transforms.Compose, transforms.Compose]:
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


def get_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
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


def get_tiny_imagenet_transforms() -> tuple[transforms.Compose, transforms.Compose]:
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


def get_svhn_transforms() -> tuple[transforms.Compose, transforms.Compose]:
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

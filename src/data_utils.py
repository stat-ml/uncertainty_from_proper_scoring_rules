import sys
sys.path.insert(0, './')
from external_repos.pytorch_cifar10.utils import get_model
import pickle
from torchvision import transforms
import torchvision
import torch.nn as nn
import torch
import os
from external_repos.pytorch_cifar100.utils import (
    get_transforms as get_cifar100_transforms,
)
from external_repos.pytorch_cifar10.utils import (
    get_transforms as get_cifar10_transforms,
)


def make_load_path(
        architecture: str,
        loss_function_name: str,
        dataset_name: str,
        model_id: int,
):
    """Create load path for specific model

    Args:
        architecture (str): _description_
        loss_function_name (str): _description_
        dataset_name (str): _description_
        model_id (int): _description_

    Returns:
        _type_: _description_
    """
    if dataset_name == 'cifar10':
        return (f'./external_repos/'
                'pytorch_cifar10/'
                'checkpoints/'
                f'{architecture}/{loss_function_name}/{model_id}/')
    elif dataset_name == 'cifar100':
        return (f'./external_repos/'
                'pytorch_cifar100/'
                'checkpoints/'
                f'{architecture}/{loss_function_name}/{model_id}/')
    else:
        raise ValueError('No such dataset name supported.')


def load_dataloader_for_extraction(
        training_dataset_name: str,
        extraction_dataset_name: str,
) -> torch.utils.data.DataLoader:
    """The function returns dataloader for extracting embeddings.
    It takes into account proper transformations from training dataset,
    and performs corresponding normalization.

    Args:
        training_dataset_name (str): name of the dataset, used in training
        extraction_dataset_name (str): name of the dataset,
                            we want extract embeddings from

    Returns:
        torch.utils.data.DataLoader: correspinding test loader
    """
    if training_dataset_name == 'cifar10':
        _, ind_transforms = get_cifar10_transforms()
    elif training_dataset_name == 'cifar100':
        _, ind_transforms = get_cifar100_transforms()
    else:
        ValueError("No such dataset available")

    if extraction_dataset_name in ['stl10', 'lsun']:
        if training_dataset_name in ['cifar10', 'cifar100', 'svhn']:
            ind_transforms = transforms.Compose(
                [transforms.Resize((32, 32))] + ind_transforms.transforms)

    if extraction_dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root='./datasets',
            train=False,
            download=True,
            transform=ind_transforms
        )
    elif extraction_dataset_name == 'blurred_cifar100':
        ind_transforms = transforms.Compose(
            [transforms.GaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 2.0))
             ] + ind_transforms.transforms)
        dataset = torchvision.datasets.CIFAR100(
            root='./datasets',
            train=False,
            download=True,
            transform=ind_transforms
        )

    elif extraction_dataset_name == 'stl10':
        dataset = torchvision.datasets.STL10(
            root='./datasets',
            split='test',
            download=True,
            transform=ind_transforms
        )

    elif extraction_dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./datasets',
            train=False,
            download=True,
            transform=ind_transforms
        )

    elif extraction_dataset_name == 'blurred_cifar10':
        ind_transforms = transforms.Compose(
            [transforms.GaussianBlur(
                kernel_size=(3, 3), sigma=(0.1, 2.0))
             ] + ind_transforms.transforms)
        dataset = torchvision.datasets.CIFAR10(
            root='./datasets',
            train=False,
            download=True,
            transform=ind_transforms
        )

    elif extraction_dataset_name == 'svhn':
        dataset = torchvision.datasets.SVHN(
            root='./datasets',
            split='test',
            download=True,
            transform=ind_transforms
        )
    else:
        ValueError("No such dataset available")

    testloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=100
    )
    return testloader


def save_dict(save_path: str, dict_to_save: dict) -> None:
    """The function saves dict to a specific file

    Args:
        save_path (str): _description_
        dict_to_save (dict): _description_
    """
    with open(save_path, 'wb') as file:
        pickle.dump(dict_to_save, file)


def load_dict(load_path: str) -> dict:
    """The function loads dict from a specific file

    Args:
        load_path (str): _description_

    Returns:
        dict: _description_
    """
    with open(load_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def load_embeddings_dict(
    architecture: str,
    dataset_name: str,
    model_id: int,
    loss_function_name: str,
):
    """The function loads dict with extracted embeddings

    Args:
        architecture (str): _description_
        dataset_name (str): _description_
        model_id (int): _description_
        loss_function_name (str): _description_

    Returns:
        _type_: _description_
    """
    file_path = make_load_path(
        architecture=architecture,
        dataset_name=dataset_name,
        model_id=model_id,
        loss_function_name=loss_function_name
    )

    # Loading the dictionary from the file
    loaded_dict = load_dict(
        load_path=os.path.join(file_path, f'embeddings_{dataset_name}.pkl'))

    return loaded_dict


def load_model_checkpoint(
    architecture: str,
        path: str,
        n_classes: int,
        device
) -> nn.Module:
    """Load trained model checkpoint

    Args:
        architecture (str): _description_
        path (str): _description_
        n_classes (int): _description_
        device (_type_): _description_

    Returns:
        nn.Module: _description_
    """
    checkpoint = torch.load(path, map_location=device)
    model = get_model(architecture=architecture, n_classes=n_classes)
    model.load_state_dict(checkpoint['net'])
    return model

import os
import pickle


from torchvision import transforms

from psruq.datasets.constants import DatasetName
from psruq.datasets.loaders import get_dataloaders
from psruq.datasets.transforms import get_transforms
from psruq.source.path_utils import make_load_path


NOT_CIFAR_LIKE_DATASETS = [
    DatasetName.TINY_IMAGENET.value,
]
BLURRED_DATASETS = [
    DatasetName.CIFAR10_BLURRED.value,
    DatasetName.CIFAR100_BLURRED.value,
]


def load_dataloader_for_extraction(
    training_dataset_name: str,
    extraction_dataset_name: str,
    severity: int | None = None,
):
    _, joint_transforms = get_transforms(training_dataset_name)

    if (extraction_dataset_name in NOT_CIFAR_LIKE_DATASETS) and (
        training_dataset_name not in NOT_CIFAR_LIKE_DATASETS
    ):
        joint_transforms = transforms.Compose(
            [transforms.Resize((32, 32))] + joint_transforms.transforms
        )

    if (training_dataset_name not in NOT_CIFAR_LIKE_DATASETS) and (
        extraction_dataset_name in NOT_CIFAR_LIKE_DATASETS
    ):
        joint_transforms = transforms.Compose(
            [transforms.Resize((64, 64))] + joint_transforms.transforms
        )
    if extraction_dataset_name in BLURRED_DATASETS:
        joint_transforms = transforms.Compose(
            [transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))]
            + joint_transforms.transforms
        )

    _, testloader = get_dataloaders(
        dataset=extraction_dataset_name,
        transform_train=joint_transforms,
        transform_test=joint_transforms,
        severity=severity,
    )

    return testloader


def save_dict(save_path: str, dict_to_save: dict) -> None:
    """The function saves dict to a specific file

    Args:
        save_path (str): _description_
        dict_to_save (dict): _description_
    """
    with open(save_path, "wb") as file:
        pickle.dump(dict_to_save, file)


def load_dict(load_path: str) -> dict:
    """The function loads dict from a specific file

    Args:
        load_path (str): _description_

    Returns:
        dict: _description_
    """
    with open(load_path, "rb") as file:
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
        loss_function_name=loss_function_name,
    )

    # Loading the dictionary from the file
    loaded_dict = load_dict(load_path=os.path.join(file_path, f"{dataset_name}.pkl"))

    return loaded_dict

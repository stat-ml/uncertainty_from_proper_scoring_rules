import os

from psruq.models.constants import ModelSource
from psruq.source.path_config import REPOSITORY_ROOT


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
    if dataset_name == "cifar10":
        code_folder = "pytorch_cifar10"
        checkpoint_folder = "checkpoints"

    elif dataset_name == "cifar10_noisy_label":
        code_folder = "pytorch_cifar10"
        checkpoint_folder = "checkpoints_noisy_cifar10"

    elif dataset_name == "cifar100_noisy_label":
        code_folder = "pytorch_cifar100"
        checkpoint_folder = "checkpoints_noisy_cifar100"

    elif dataset_name == "missed_class_cifar10":
        code_folder = "pytorch_cifar10"
        checkpoint_folder = "checkpoints_missed_class_cifar10"

    elif dataset_name == "svhn":
        code_folder = "pytorch_cifar10"
        checkpoint_folder = "checkpoints_svhn"

    elif dataset_name == "cifar100":
        code_folder = "pytorch_cifar100"
        checkpoint_folder = "checkpoints"

    else:
        raise ValueError("No such dataset name supported.")

    return (
        f"{REPOSITORY_ROOT}/external_repos/"
        f"{code_folder}/{checkpoint_folder}/"
        f"{architecture}/{loss_function_name}/{model_id}/"
    )


def get_model_folder(dataset_name: str):
    match dataset_name:
        case "cifar100":
            return "cifar100-resnet18-0-1023"
        case "cifar10":
            return "cifar10-resnet18-0-1023"
        case "tiny_imagenet":
            return "tiny-imagenet-resnet18-0-1023"
        case _:
            raise ValueError(f"No such dataset: {dataset_name}")


def make_model_load_path(
    version: int,
    training_dataset: str,
):
    path_to_folder_with_models = os.path.join(
        REPOSITORY_ROOT,
        "torch_uncertainty_models",
        "models",
        get_model_folder(dataset_name=training_dataset),
    )
    path = os.path.join(path_to_folder_with_models, f"version_{version}.safetensors")
    return path


def make_logits_path(
    model_id: int,
    training_dataset_name: str,
    extraction_dataset_name: str,
    severity: int | None,
    model_source: str,
    architecture: str,
    loss_function_name: str,
) -> str:
    match model_source:
        case ModelSource.TORCH_UNCERTAINTY.value:
            logits_dir = os.path.join(
                REPOSITORY_ROOT,
                "torch_uncertainty_models",
                "logits",
                training_dataset_name,
                str(model_id),
            )
        case ModelSource.OUR_MODELS.value:
            logits_dir = make_load_path(
                architecture, loss_function_name, training_dataset_name, model_id
            )
        case _:
            raise ValueError("No such option for ModelSource yet")
    os.makedirs(logits_dir, exist_ok=True)
    if severity is None:
        logits_path = os.path.join(logits_dir, f"{extraction_dataset_name}.pkl")
    else:
        logits_path = os.path.join(
            logits_dir, f"{extraction_dataset_name}_{str(severity)}.pkl"
        )
    return logits_path

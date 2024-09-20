import os
from pathlib import Path


def get_model_folder(dataset_name: str):
    match dataset_name:
        case "cifar100":
            return "cifar100-resnet18-0-1023"
        case "cifar10":
            return "cifar10-resnet18-0-1023"
        case "tiny_imagenet":
            return "tiny-imagenet-resnet18-0-1023"
        case _:
            raise ValueError("No such dataset")


def make_model_load_path(
    version: int,
    training_dataset: str,
):
    path_to_folder_with_models = os.path.join(
        ROOT_PATH, "models", get_model_folder(dataset_name=training_dataset)
    )
    path = os.path.join(path_to_folder_with_models, f"version_{version}.safetensors")
    return path


def make_logits_path(
    version: int,
    training_dataset_name: str,
    extraction_dataset_name: str,
    severity: int | None = None,
):
    path_to_folder_with_logits = os.path.join(
        ROOT_PATH, "logits", training_dataset_name
    )
    logits_dir = os.path.join(path_to_folder_with_logits, str(version))
    os.makedirs(logits_dir, exist_ok=True)
    if severity is None:
        logits_path = os.path.join(logits_dir, f"{extraction_dataset_name}.pkl")
    else:
        logits_path = os.path.join(
            logits_dir, f"{extraction_dataset_name}_{str(severity)}.pkl"
        )
    return logits_path


def get_root_path(start_point_absolute_path: str) -> str:
    root_path = Path(start_point_absolute_path).resolve().parent.parent
    return root_path


ROOT_PATH = get_root_path(__file__)

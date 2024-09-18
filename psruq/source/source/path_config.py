from pathlib import Path


repo_root = Path(__file__).resolve().parent
while not (repo_root / ".git").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent


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

    elif dataset_name == "noisy_cifar10":
        code_folder = "pytorch_cifar10"
        checkpoint_folder = "checkpoints_noisy_cifar10"

    elif dataset_name == "noisy_cifar100":
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
        f"{repo_root}/external_repos/"
        f"{code_folder}/{checkpoint_folder}/"
        f"{architecture}/{loss_function_name}/{model_id}/"
    )

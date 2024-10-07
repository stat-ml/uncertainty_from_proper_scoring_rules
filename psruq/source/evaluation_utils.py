import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report

from psruq.models.constants import ModelSource
from psruq.source.data_utils import load_dict, load_embeddings_dict
from psruq.source.path_utils import make_load_path, make_logits_path


def get_additional_evaluation_metrics(embeddings_dict: Dict) -> Dict | str:
    embeddings = embeddings_dict["embeddings"]
    y_true = embeddings_dict["labels"]
    y_pred = np.argmax(embeddings, axis=-1)
    results_dict = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    return results_dict


def save_additional_stats(
    architecture: str,
    dataset_name: str,
    loss_function_name: str,
    model_id: int,
    model_source: str,
):
    match model_source:
        case ModelSource.OUR_MODELS.value:
            load_path = make_load_path(
                dataset_name=dataset_name,
                architecture=architecture,
                loss_function_name=loss_function_name,
                model_id=model_id,
            )

            embeddings_dict = load_embeddings_dict(
                architecture=architecture,
                loss_function_name=loss_function_name,
                dataset_name=dataset_name,
                model_id=model_id,
            )

            checkpoint_path = os.path.join(load_path, "ckpt.pth")
            last_acc = torch.load(checkpoint_path, map_location="cpu")["acc"]
            if isinstance(last_acc, torch.Tensor):
                last_acc = last_acc.cpu().detach().numpy()
            actual_acc = get_additional_evaluation_metrics(
                embeddings_dict=embeddings_dict
            )
            actual_acc.update({"last_acc": last_acc / 100})

        case ModelSource.TORCH_UNCERTAINTY.value:
            logits_path = make_logits_path(
                model_id=model_id,
                training_dataset_name=dataset_name,
                extraction_dataset_name=dataset_name,
                severity=None,
                model_source=model_source,
                architecture=architecture,
                loss_function_name=loss_function_name,
            )

            # Loading the dictionary from the file
            loaded_dict = load_dict(load_path=logits_path)

            actual_acc = get_additional_evaluation_metrics(embeddings_dict=loaded_dict)
            load_path = Path(logits_path).resolve().parent
            print(load_path)
    try:
        with open(os.path.join(load_path, "results_dict.json"), "w") as file:
            json.dump(
                fp=file,
                obj=actual_acc,
                indent=4,
            )
    except OSError:
        import pdb

        pdb.set_trace()
        print("oh")


def collect_embeddings(
    model_ids: list | np.ndarray,
    architecture: str,
    loss_function_name: str,
    training_dataset_name: str,
    model_source: str,
    list_extraction_datasets: list = [
        "cifar10",
        "cifar100",
        "svhn",
        "blurred_cifar100",
        "blurred_cifar10",
    ],
    temperature: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The function collects embeddings for different members of ensembles
      and different datasets

    Args:
        model_ids (list | np.ndarray): IDs of ensemble memebers we take
        into account
        architecture (str): model architecture name
        loss_function_name (str): loss function name
        training_dataset_name (str): dataset name used in training
        list_extraction_datasets (list, optional): datasets for which
        we will used embeddings. Defaults to ['cifar10', 'cifar100', 'svhn'].
        temperature: (float,): Temperature to scale logits. Default 1.0

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        Key -- dataset name; Value -- stacked logits.
        Key -- dataset name; Value -- stacked targets.
    """
    embeddings_per_dataset = defaultdict(list)
    targets_per_dataset = defaultdict(list)
    for extraction_dataset_name in list_extraction_datasets:
        for model_id in model_ids:
            loaded_dict = load_dict(
                make_logits_path(
                    model_id=model_id,
                    training_dataset_name=training_dataset_name,
                    extraction_dataset_name=extraction_dataset_name,
                    severity=None,
                    model_source=model_source,
                    architecture=architecture,
                    loss_function_name=loss_function_name,
                )
            )

            loaded_embeddings = loaded_dict["embeddings"] / temperature
            loaded_targets = loaded_dict["labels"]

            embeddings_per_dataset[extraction_dataset_name].append(
                loaded_embeddings[None]
            )
            targets_per_dataset[extraction_dataset_name].append(loaded_targets)

        embeddings_per_dataset[extraction_dataset_name] = np.vstack(
            embeddings_per_dataset[extraction_dataset_name]
        )
        targets_per_dataset[extraction_dataset_name] = np.hstack(
            targets_per_dataset[extraction_dataset_name]
        )

    return embeddings_per_dataset, targets_per_dataset


def collect_stats(
    dataset_name: str,
    architecture: str,
    loss_function_name,
    model_source: str,
    model_ids: list | np.ndarray,
) -> dict:
    stats_dict = defaultdict(list)
    for model_id in model_ids:
        if model_source == ModelSource.OUR_MODELS.value:
            load_path = make_load_path(
                architecture=architecture,
                loss_function_name=loss_function_name,
                dataset_name=dataset_name,
                model_id=model_id,
            )
        elif model_source == ModelSource.TORCH_UNCERTAINTY.value:
            logits_path = make_logits_path(
                model_id=model_id,
                training_dataset_name=dataset_name,
                extraction_dataset_name=dataset_name,
                severity=None,
                model_source=model_source,
                architecture=architecture,
                loss_function_name=loss_function_name,
            )
            load_path = Path(logits_path).resolve().parent
        else:
            raise ValueError(f"No such model source available!: {model_source}")

        with open(os.path.join(load_path, "results_dict.json"), "r") as file:
            current_dict_ = json.load(file)
            stats_dict["accuracy"].append(current_dict_["accuracy"])

            stats_dict["macro_avg_precision"].append(
                current_dict_["macro avg"]["precision"]
            )

            stats_dict["macro_avg_recall"].append(current_dict_["macro avg"]["recall"])

            stats_dict["macro_avg_f1-score"].append(
                current_dict_["macro avg"]["f1-score"]
            )
    return stats_dict

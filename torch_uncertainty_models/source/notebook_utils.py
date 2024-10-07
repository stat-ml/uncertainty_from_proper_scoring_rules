import os
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

from psruq.datasets.constants import DatasetName
from psruq.losses.constants import LossName
from psruq.models.constants import ModelName
from psruq.source.evaluation_utils import load_dict, save_dict
from psruq.source.path_config import ModelSource, make_logits_path
from psruq.source.postprocessing_utils import (
    ENSEMBLE_COMBINATIONS,
    UQ_FUNCS_WITH_NAMES,
)


def collect_embeddings(
    model_ids: list | np.ndarray,
    training_dataset_name: str,
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

    list_extraction_datasets = remove_and_expand_list(list_extraction_datasets)

    for extraction_dataset_name in list_extraction_datasets:
        for model_id in model_ids:
            load_logits_path = make_logits_path(
                extraction_dataset_name=extraction_dataset_name,
                training_dataset_name=training_dataset_name,
                model_id=model_id,
                severity=None,
                model_source=ModelSource.TORCH_UNCERTAINTY.value,
                architecture=ModelName.RESNET18.value,
                loss_function_name=LossName.CROSS_ENTROPY.value,
            )
            loaded_dict = load_dict(load_logits_path)

            loaded_embeddings = loaded_dict["embeddings"] / temperature
            loaded_targets = loaded_dict["labels"]

            embeddings_per_dataset[extraction_dataset_name].append(
                loaded_embeddings[None]
            )
            targets_per_dataset[extraction_dataset_name].append(loaded_targets)
        # import pdb
        # pdb.set_trace()
        embeddings_per_dataset[extraction_dataset_name] = np.vstack(
            embeddings_per_dataset[extraction_dataset_name]
        )
        targets_per_dataset[extraction_dataset_name] = np.hstack(
            targets_per_dataset[extraction_dataset_name]
        )

    return embeddings_per_dataset, targets_per_dataset


def get_new_models_sampled_combinations_uncertainty_scores(
    loss_function_names: list[LossName],
    training_dataset_name: str,
    model_ids: np.ndarray,
    list_extraction_datasets: list[str],
    temperature: float = 1.0,
    use_cached: bool = True,
) -> tuple[dict, dict, dict]:
    load_logits_path = make_logits_path(
        extraction_dataset_name="NaN",
        training_dataset_name=training_dataset_name,
        model_id="NaN",
        severity=None,
        model_source=ModelSource.TORCH_UNCERTAINTY.value,
        architecture=ModelName.RESNET18.value,
        loss_function_name=LossName.CROSS_ENTROPY.value,
    )
    extracted_uq_measures_file_path = os.path.join(
        "/".join(load_logits_path.split("/")[:-3]),
        f"{training_dataset_name}_CENTRAL_extracted_information_for_notebook_combinations.pkl",
    )

    if use_cached and os.path.exists(extracted_uq_measures_file_path):
        res_dict = load_dict(extracted_uq_measures_file_path)
        uq_results, embeddings_per_dataset, targets_per_dataset = (
            res_dict["uq_results"],
            res_dict["embeddings_per_dataset"],
            res_dict["targets_per_dataset"],
        )
        return uq_results, embeddings_per_dataset, targets_per_dataset

    uq_results = {}
    embeddings_per_dataset_all = {}

    for uq_name, uncertainty_func in tqdm(UQ_FUNCS_WITH_NAMES):
        uq_results[uq_name] = {}
        for loss_function_name in loss_function_names:
            embeddings_per_dataset, targets_per_dataset = collect_embeddings(
                model_ids=model_ids,
                training_dataset_name=training_dataset_name,
                list_extraction_datasets=list_extraction_datasets,
                temperature=temperature,
            )

            uq_results[uq_name][loss_function_name.value] = {}
            embeddings_per_dataset_all[loss_function_name.value] = (
                embeddings_per_dataset
            )

            for dataset_ in list_extraction_datasets:
                logits = embeddings_per_dataset[dataset_]
                uq_results[uq_name][loss_function_name.value][dataset_] = []

                for comb in ENSEMBLE_COMBINATIONS:
                    uq_results[uq_name][loss_function_name.value][dataset_].append(
                        uncertainty_func(
                            logits=logits[list(comb)],
                            T=temperature,
                        )
                    )

    res_dict = {
        "uq_results": uq_results,
        "embeddings_per_dataset": embeddings_per_dataset_all,
        "targets_per_dataset": targets_per_dataset,
    }
    save_dict(dict_to_save=res_dict, save_path=extracted_uq_measures_file_path)
    return uq_results, embeddings_per_dataset_all, targets_per_dataset

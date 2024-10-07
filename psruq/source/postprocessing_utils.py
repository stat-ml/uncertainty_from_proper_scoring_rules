import os
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from psruq.datasets.constants import DatasetName
from psruq.losses.constants import LossName
from psruq.metrics import (
    ApproximationType,
    GName,
    RiskType,
    get_energy_inner,
    get_energy_outer,
    get_risk_approximation,
    posterior_predictive,
)
from psruq.models.constants import ModelName, ModelSource
from psruq.source.data_utils import load_dict, save_dict
from psruq.source.evaluation_utils import collect_embeddings, collect_stats
from psruq.source.path_utils import make_load_path, make_logits_path


def remove_and_expand_list(list_extraction_datasets: list[str]) -> list[str]:
    if DatasetName.CIFAR10C.value in list_extraction_datasets:
        list_extraction_datasets.remove(DatasetName.CIFAR10C.value)
        list_extraction_datasets.extend(
            [DatasetName.CIFAR10C.value + f"_{i}" for i in range(1, 6)]
        )
    if DatasetName.CIFAR100C.value in list_extraction_datasets:
        list_extraction_datasets.remove(DatasetName.CIFAR100C.value)
        list_extraction_datasets.extend(
            [DatasetName.CIFAR100C.value + f"_{i}" for i in range(1, 6)]
        )
    return list_extraction_datasets


ENSEMBLE_COMBINATIONS = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (8, 9, 10, 11),
    (12, 13, 14, 15),
    (16, 17, 18, 19),
]


def get_uq_funcs_with_names() -> list[tuple[str, callable]]:
    uq_funcs_with_names = []
    for g_name in [el for el in GName]:
        for risk_type in [el for el in RiskType]:
            for gt_approx in [el for el in ApproximationType]:
                for pred_approx in [el for el in ApproximationType]:
                    specific_instance = partial(
                        get_risk_approximation,
                        g_name=g_name,
                        risk_type=risk_type,
                        gt_approx=gt_approx,
                        pred_approx=pred_approx,
                    )
                    if risk_type == RiskType.BAYES_RISK:
                        uq_funcs_with_names.append(
                            (
                                f"{g_name.value} {risk_type.value} {gt_approx.value}",
                                specific_instance,
                            )
                        )
                    else:
                        uq_funcs_with_names.append(
                            (
                                f"{g_name.value} {risk_type.value} {gt_approx.value} {pred_approx.value}",
                                specific_instance,
                            )
                        )
    uq_funcs_with_names.append(
        (f"{GName.LOG_SCORE.value} energy outer", get_energy_outer)
    )
    uq_funcs_with_names.append(
        (f"{GName.LOG_SCORE.value} energy inner", get_energy_inner)
    )
    return uq_funcs_with_names


UQ_FUNCS_WITH_NAMES = get_uq_funcs_with_names()


pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True


def get_metrics_results(
    loss_function_names: list[LossName],
    training_dataset_name: str,
    architecture: str,
    model_ids: np.ndarray,
    model_source: str,
):
    """
    The function reads all the stats from the path and creates
    a pandas dataframe with metrics.
    """
    loss_to_stats = {}

    for loss_function_name in loss_function_names:
        loss_to_stats[loss_function_name] = collect_stats(
            dataset_name=training_dataset_name,
            architecture=architecture,
            loss_function_name=loss_function_name.value,
            model_ids=model_ids,
            model_source=model_source,
        )

    # Convert the nested dictionary to a pandas DataFrame for easier plotting
    data_flat = []
    for loss_function, metrics in loss_to_stats.items():
        for metric, values in metrics.items():
            for value in values:
                data_flat.append(
                    {
                        "Loss Function": loss_function.value,
                        "Metric": metric,
                        "Value": value,
                    }
                )

    df = pd.DataFrame(data_flat)

    return df


def get_predicted_labels(
    embeddings_per_dataset: dict,
    training_dataset_name: str,
):
    """
    The function returns predicted labels given embeddings for a given dataset
    """
    pred_labels_dict = {}
    for loss in embeddings_per_dataset:
        pred_labels_dict[loss] = []
        for comb in ENSEMBLE_COMBINATIONS:
            pred_labels = np.argmax(
                posterior_predictive(
                    embeddings_per_dataset[loss][training_dataset_name][list(comb)]
                )[0],
                axis=-1,
            )
            pred_labels_dict[loss].append(pred_labels)
    return pred_labels_dict


def get_ood_detection_dataframe(
    ind_dataset: str,
    uq_results: dict,
    list_ood_datasets: list[str],
) -> pd.DataFrame:
    """
    The function transforms uq_results dict into pd.Dataframe
    with ROC AUC scores of OOD detection.
    """
    roc_auc_dict = {}

    for uq_name, _ in UQ_FUNCS_WITH_NAMES:
        roc_auc_dict[uq_name] = {}

        for ood_dataset in list_ood_datasets:
            roc_auc_dict[uq_name][ood_dataset] = {}
            for loss_ in uq_results[uq_name].keys():
                roc_auc_dict[uq_name][ood_dataset][loss_] = []
                for it_ in range(len(uq_results[uq_name][loss_][ood_dataset])):
                    y_true = np.hstack(
                        [
                            np.ones(
                                uq_results[uq_name][loss_][ood_dataset][it_].shape[0]
                            ),
                            np.zeros(
                                uq_results[uq_name][loss_][ind_dataset][it_].shape[0]
                            ),
                        ]
                    )
                    y_score = np.hstack(
                        [
                            uq_results[uq_name][loss_][ood_dataset][it_],
                            uq_results[uq_name][loss_][ind_dataset][it_],
                        ]
                    )
                    score = roc_auc_score(y_true=y_true, y_score=y_score)
                    roc_auc_dict[uq_name][ood_dataset][loss_].append(score)

    data_list = []
    for metric_name, datasets in roc_auc_dict.items():
        for dataset_name, loss_functions in datasets.items():
            for loss_function_name, values in loss_functions.items():
                data_list.append(
                    (metric_name, dataset_name, loss_function_name, values)
                )

    # Create a DataFrame
    df = pd.DataFrame(
        data_list, columns=["UQMetric", "Dataset", "LossFunction", "RocAucScores_array"]
    )

    return df


def get_missclassification_dataframe(
    ind_dataset: str,
    uq_results: dict,
    true_labels: np.ndarray,
    pred_labels: list[np.ndarray],
) -> pd.DataFrame:
    """
    The function transforms uq_results dict into pd.Dataframe
    with ROC AUC scores of misclassification detection.
    """
    roc_auc_dict = {}

    for uq_name, _ in UQ_FUNCS_WITH_NAMES:
        roc_auc_dict[uq_name] = {}

        for loss_ in uq_results[uq_name].keys():
            roc_auc_dict[uq_name][loss_] = []
            for it_ in range(len(uq_results[uq_name][loss_][ind_dataset])):
                y_true = (true_labels != pred_labels[loss_][it_]).astype(np.int32)
                y_score = uq_results[uq_name][loss_][ind_dataset][it_]

                score = roc_auc_score(y_true=y_true, y_score=y_score)
                roc_auc_dict[uq_name][loss_].append(score)

    data_list_misclassification = []
    for metric_name, loss_function in roc_auc_dict.items():
        for loss_function_name, values in loss_function.items():
            data_list_misclassification.append(
                (metric_name, loss_function_name, values)
            )

    # Create a DataFrame
    df_misclassification = pd.DataFrame(
        data_list_misclassification,
        columns=[
            "UQMetric",
            "LossFunction",
            "RocAucScores_array",
        ],
    )
    return df_misclassification


def get_raw_scores_dataframe(
    uq_results: dict,
):
    # Convert the nested dictionary into a DataFrame
    data = []
    for uq_name, loss_funcs in uq_results.items():
        for loss_name, datasets in loss_funcs.items():
            for dataset_name, scores in datasets.items():
                data.append(
                    {
                        "UQMetric": uq_name,
                        "LossFunction": loss_name,
                        "Dataset": dataset_name,
                        "Scores": list(scores),
                    }
                )

    df = pd.DataFrame(data)
    return df


def get_sampled_combinations_uncertainty_scores(
    loss_function_names: list[LossName],
    training_dataset_name: str,
    architecture: ModelName,
    model_ids: np.ndarray,
    list_extraction_datasets: list[str],
    model_source: str,
    temperature: float = 1.0,
    use_cached: bool = True,
) -> tuple[dict, dict, dict]:
    folder_path = make_logits_path(
        model_id="NaN",
        extraction_dataset_name="NaN",
        training_dataset_name=training_dataset_name,
        model_source=model_source,
        severity=None,
        architecture=architecture.value,
        loss_function_name="NaN",
    )

    extracted_embeddings_file_path = os.path.join(
        "/".join(folder_path.split("/")[:-3]),
        f"{training_dataset_name}_CENTRAL_extracted_information_for_notebook_combinations.pkl",
    )

    if use_cached and os.path.exists(extracted_embeddings_file_path):
        res_dict = load_dict(extracted_embeddings_file_path)
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
                architecture=architecture.value,
                loss_function_name=loss_function_name.value,
                training_dataset_name=training_dataset_name,
                list_extraction_datasets=list_extraction_datasets,
                temperature=temperature,
                model_source=model_source,
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
    save_dict(dict_to_save=res_dict, save_path=extracted_embeddings_file_path)
    return uq_results, embeddings_per_dataset_all, targets_per_dataset

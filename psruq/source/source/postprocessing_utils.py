import os
import re
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from source.datasets.constants import DatasetName
from source.losses.constants import LossName
from source.metrics import (
    ApproximationType,
    GName,
    RiskType,
    get_risk_approximation,
    posterior_predictive,
)
from source.models.constants import ModelName
from source.source.data_utils import load_dict, make_load_path, save_dict
from source.source.evaluation_utils import collect_embeddings, collect_stats
from tqdm.auto import tqdm


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
    return uq_funcs_with_names


UQ_FUNCS_WITH_NAMES = get_uq_funcs_with_names()


pd.set_option("mode.copy_on_write", True)
pd.options.mode.copy_on_write = True


def get_metrics_results(
    loss_function_names: list[LossName],
    training_dataset_name: str,
    architecture: str,
    model_ids: np.ndarray,
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
        pred_labels = np.argmax(
            posterior_predictive(embeddings_per_dataset[loss][training_dataset_name])[
                0
            ],
            axis=-1,
        )
        pred_labels_dict[loss] = pred_labels
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
        # print(f'OOD computed via {uq_name}')

        for ood_dataset in list_ood_datasets:
            roc_auc_dict[uq_name][ood_dataset] = {}
            for loss_ in uq_results[uq_name].keys():
                y_true = np.hstack(
                    [
                        np.ones(uq_results[uq_name][loss_][ood_dataset].shape[0]),
                        np.zeros(uq_results[uq_name][loss_][ind_dataset].shape[0]),
                    ]
                )
                y_score = np.hstack(
                    [
                        uq_results[uq_name][loss_][ood_dataset],
                        uq_results[uq_name][loss_][ind_dataset],
                    ]
                )
                score = roc_auc_score(y_true=y_true, y_score=y_score)
                roc_auc_dict[uq_name][ood_dataset][loss_] = score

                # print(
                #     (
                #         f'InD: {ind_dataset} \t '
                #         f'OOD: {ood_dataset} \t '
                #         f'loss: {loss_} \t '
                #         f'roc_auc: {score}'
                #     )
                # )

    data_list = []
    for metric_name, datasets in roc_auc_dict.items():
        for dataset_name, loss_functions in datasets.items():
            for loss_function_name, value in loss_functions.items():
                data_list.append((metric_name, dataset_name, loss_function_name, value))

    # Create a DataFrame
    df = pd.DataFrame(
        data_list, columns=["UQMetric", "Dataset", "LossFunction", "RocAucScore"]
    )

    return df


def get_missclassification_dataframe(
    ind_dataset: str,
    uq_results: dict,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> pd.DataFrame:
    """
    The function transforms uq_results dict into pd.Dataframe
    with ROC AUC scores of misclassification detection.
    """
    roc_auc_dict = {}

    for uq_name, _ in UQ_FUNCS_WITH_NAMES:
        roc_auc_dict[uq_name] = {}
        # print(f'Misclassification computed via {uq_name}')

        for loss_ in uq_results[uq_name].keys():
            y_true = (true_labels != pred_labels[loss_]).astype(np.int32)
            y_score = uq_results[uq_name][loss_][ind_dataset]

            score = roc_auc_score(y_true=y_true, y_score=y_score)
            roc_auc_dict[uq_name][loss_] = score

            # print(
            #     f'InD: {ind_dataset} \t loss: {loss_} \t roc_auc: {score}')

    data_list_misclassification = []
    for metric_name, loss_function in roc_auc_dict.items():
        for loss_function_name, value in loss_function.items():
            data_list_misclassification.append((metric_name, loss_function_name, value))

    # Create a DataFrame
    df_misclassification = pd.DataFrame(
        data_list_misclassification,
        columns=[
            "UQMetric",
            "LossFunction",
            "RocAucScore",
        ],
    )
    return df_misclassification


def make_aggregation(
    df_: pd.DataFrame,
    prefix_: str,
    suffix_: Optional[str],
    dataset_: Optional[str] = None,
    by_loss_function_: bool = False,
) -> pd.DataFrame:
    """
    The function aggregates given pd.DataFrame in different ways
    prefix_ used to select a proper UQ name (Total, Excess, Bayes, Bregman..)
    suffix_ used to specify Inner or Outer. If None, both are aggregated
    dataset_ used to specify a concrete dataset
    by_loss_function_ if True, aggregation by Loss function is used
    """
    df = pd.DataFrame.copy(df_)
    if dataset_ is None:
        dataset_mask = np.ones(len(df))
    else:
        dataset_mask = df["Dataset"].str.fullmatch(dataset_)
    if suffix_ is not None:
        filtered_df = df[
            dataset_mask
            & df["UQMetric"].str.startswith(prefix_)
            & df["UQMetric"].str.endswith(suffix_)
        ]
        if by_loss_function_:
            roc_auc_scores = filtered_df.groupby(["UQMetric", "LossFunction"]).agg(
                {"RocAucScore": ["mean", "std", "count"]}
            )
        else:
            roc_auc_scores = filtered_df.groupby(
                [
                    "UQMetric",
                ]
            ).agg({"RocAucScore": ["mean", "std", "count"]})
    else:
        filtered_df = df[dataset_mask & df["UQMetric"].str.startswith(prefix_)]
        filtered_df.loc[:, "GeneralizedMetric"] = filtered_df["UQMetric"].apply(
            lambda x: re.sub(r"\s+(Inner|Outer)$", "", x)
        )

        # Group by 'GeneralizedMetric' and calculate mean ROC AUC score
        if by_loss_function_:
            roc_auc_scores = filtered_df.groupby(
                ["GeneralizedMetric", "LossFunction"]
            ).agg({"RocAucScore": ["mean", "std", "count"]})
        else:
            roc_auc_scores = filtered_df.groupby(
                [
                    "GeneralizedMetric",
                ]
            ).agg({"RocAucScore": ["mean", "std", "count"]})
    # roc_auc_scores = \
    # filtered_df.groupby(['UQMetric', 'LossFunction'])['RocAucScore'].mean()

    roc_auc_df = roc_auc_scores.reset_index()
    roc_auc_df.rename(columns={"RocAucScore": "MeanRocAucScore"}, inplace=True)
    return roc_auc_df


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


def ravel_df(
    df: pd.DataFrame,
    loss_function: str,
    dataset: str,
    score: str,
) -> pd.DataFrame:
    """
    The function takes dataframe where score value is np.ndarray
    and transforms it in a way that it will be a column.
    It also selects a specific score, dataset and loss_function
    """
    tot = df[
        (
            df["Loss Function"].str.startswith(loss_function)
            & df["Dataset"].str.startswith(dataset)
            & df["UQ Method"].str.startswith(f"Total {score}")
        )
    ].Scores.values[0]

    exc_o = df[
        (
            df["Loss Function"].str.startswith(loss_function)
            & df["Dataset"].str.startswith(dataset)
            & df["UQ Method"].str.startswith(f"Excess {score} Outer")
        )
    ].Scores.values[0]
    exc_i = df[
        (
            df["Loss Function"].str.startswith(loss_function)
            & df["Dataset"].str.startswith(dataset)
            & df["UQ Method"].str.startswith(f"Excess {score} Inner")
        )
    ].Scores.values[0]

    bay_o = df[
        (
            df["Loss Function"].str.startswith(loss_function)
            & df["Dataset"].str.startswith(dataset)
            & df["UQ Method"].str.startswith(f"Bayes {score} Outer")
        )
    ].Scores.values[0]
    bay_i = df[
        (
            df["Loss Function"].str.startswith(loss_function)
            & df["Dataset"].str.startswith(dataset)
            & df["UQ Method"].str.startswith(f"Bayes {score} Inner")
        )
    ].Scores.values[0]

    bregman = df[
        (
            df["Loss Function"].str.startswith(loss_function)
            & df["Dataset"].str.startswith(dataset)
            & df["UQ Method"].str.startswith(f"Bregman Information {score}")
        )
    ].Scores.values[0]

    scores_dict_ = {
        "Total": tot,
        "Excess outer": exc_o,
        "Bayes outer": bay_o,
        "Excess inner": exc_i,
        "Bayes inner": bay_i,
        "Bregman Information": bregman,
        "Total outer": exc_o + bay_o,
        "Total inner": exc_i + bay_i,
    }
    scores_df = pd.DataFrame.from_dict(scores_dict_)

    return scores_df


def get_uncertainty_scores(
    loss_function_names: list[LossName],
    training_dataset_name: LossName,
    architecture: ModelName,
    model_ids: np.ndarray,
    list_extraction_datasets: list[str],
    temperature: float = 1.0,
    use_cached: bool = True,
) -> tuple[dict, dict, dict]:
    """
    The function extracts uncertainty scores from a list of datasets.
    It returns a structured dict: [uq score][loss_function][dataset] -> scores
    It also returns a dict of embeddings for each dataset
    and a dict of GT labels for each dataset
    """

    folder_path = make_load_path(
        architecture=architecture.value,
        dataset_name=training_dataset_name,
        loss_function_name="NaN",
        model_id="NaN",
    )
    extracted_embeddings_file_path = os.path.join(
        *folder_path.split("/")[:-3], "extracted_information_for_notebook.pkl"
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
            )

            uq_results[uq_name][loss_function_name.value] = {}
            embeddings_per_dataset_all[loss_function_name.value] = (
                embeddings_per_dataset
            )

            for dataset_ in list_extraction_datasets:
                logits = embeddings_per_dataset[dataset_]

                uq_results[uq_name][loss_function_name.value][dataset_] = (
                    uncertainty_func(
                        logits=logits,
                        T=temperature,
                    )
                )

    res_dict = {
        "uq_results": uq_results,
        "embeddings_per_dataset": embeddings_per_dataset_all,
        "targets_per_dataset": targets_per_dataset,
    }
    # save_dict(dict_to_save=res_dict, save_path=extracted_embeddings_file_path)
    return uq_results, embeddings_per_dataset_all, targets_per_dataset

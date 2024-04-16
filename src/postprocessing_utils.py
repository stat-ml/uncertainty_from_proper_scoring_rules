import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from collections import defaultdict
from src.evaluation_utils import collect_stats
from evaluation_utils import make_load_path, load_dict, collect_embeddings
from uncertainty_scores import (
    total_brier,
    total_logscore,
    total_neglog,
    total_maxprob,
    total_spherical,
    bayes_brier_inner,
    bayes_brier_outer,
    bayes_logscore_inner,
    bayes_logscore_outer,
    bayes_maxprob_inner,
    bayes_maxprob_outer,
    bayes_neglog_inner,
    bayes_neglog_outer,
    bayes_spherical_inner,
    bayes_spherical_outer,
    excess_brier_inner,
    excess_brier_outer,
    excess_logscore_inner,
    excess_logscore_outer,
    excess_maxprob_inner,
    excess_maxprob_outer,
    excess_neglog_inner,
    excess_neglog_outer,
    excess_spherical_inner,
    excess_spherical_outer,
    bi_brier,
    bi_logscore,
    bi_maxprob,
    bi_neglog,
    bi_spherical,
    mutual_information_avg_kl,
    logscore_bias,
    logscore_model_variance,
    logscore_model_variance_plus_mi,
    logscore_bias_plus_mi,
    posterior_predictive
)


uq_funcs_with_names = [
    ("Total Brier", total_brier),
    ("Total Logscore", total_logscore),
    ("Total Neglog", total_neglog),
    ("Total Maxprob", total_maxprob),
    ("Total Spherical", total_spherical),
    ("Bayes Brier Inner", bayes_brier_inner),
    ("Bayes Brier Outer", bayes_brier_outer),
    ("Bayes Logscore Inner", bayes_logscore_inner),
    ("Bayes Logscore Outer", bayes_logscore_outer),
    ("Bayes Maxprob Inner", bayes_maxprob_inner),
    ("Bayes Maxprob Outer", bayes_maxprob_outer),
    ("Bayes Neglog Inner", bayes_neglog_inner),
    ("Bayes Neglog Outer", bayes_neglog_outer),
    ("Bayes Spherical Inner", bayes_spherical_inner),
    ("Bayes Spherical Outer", bayes_spherical_outer),
    ("Excess Brier Inner", excess_brier_inner),
    ("Excess Brier Outer", excess_brier_outer),
    ("Excess Logscore Inner", excess_logscore_inner),
    ("Excess Logscore Outer", excess_logscore_outer),
    ("Excess Maxprob Inner", excess_maxprob_inner),
    ("Excess Maxprob Outer", excess_maxprob_outer),
    ("Excess Neglog Inner", excess_neglog_inner),
    ("Excess Neglog Outer", excess_neglog_outer),
    ("Excess Spherical Inner", excess_spherical_inner),
    ("Excess Spherical Outer", excess_spherical_outer),
    ("Bregman Information Brier", bi_brier),
    ("Bregman Information Logscore", bi_logscore),
    ("Bregman Information Maxprob", bi_maxprob),
    ("Bregman Information Neglog", bi_neglog),
    ("Bregman Information Spherical", bi_spherical),
    ("Mutual Information", mutual_information_avg_kl),
    ("Logscore Bias term", logscore_bias),
    ("Logscore Model Variance term", logscore_model_variance),
    ("Logscore Model Variance + MI", logscore_model_variance_plus_mi),
    ("Logscore Bias Term + MI", logscore_bias_plus_mi)
]


def get_metrics_results(
        loss_function_names: list[str],
        training_dataset_name: str,
        architecture: str,
        model_ids: np.ndarray
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
            loss_function_name=loss_function_name,
            model_ids=model_ids
        )

    # Convert the nested dictionary to a pandas DataFrame for easier plotting
    data_flat = []
    for loss_function, metrics in loss_to_stats.items():
        for metric, values in metrics.items():
            for value in values:
                data_flat.append(
                    {
                        "Loss Function": loss_function,
                        "Metric": metric,
                        "Value": value
                    }
                )

    df = pd.DataFrame(data_flat)

    return df


def get_uncertainty_scores(
        loss_function_names: list[str],
        training_dataset_name: str,
        architecture: str,
        model_ids: np.ndarray,
        list_extraction_datasets: list[str],
        temperature: float = 1.0,
        use_cheating_approximation: bool = False,
        gt_prob_approx: str = 'same',
) -> tuple[dict, dict, dict]:
    """
    The function extracts uncertainty scores from a list of datasets.
    It returns a structured dict: [uq score][loss_function][dataset] -> scores
    It also returns a dict of embeddings for each dataset
    and a dict of GT labels for each dataset
    """
    uq_results = {}

    for uq_name, uncertainty_func in tqdm(uq_funcs_with_names):
        uq_results[uq_name] = {}
        for loss_function_name in loss_function_names:
            embeddings_per_dataset, targets_per_dataset = collect_embeddings(
                model_ids=model_ids,
                architecture=architecture,
                loss_function_name=loss_function_name,
                training_dataset_name=training_dataset_name,
                list_extraction_datasets=list_extraction_datasets,
                temperature=temperature,
            )

            uq_results[uq_name][loss_function_name] = {}

            for dataset_ in list_extraction_datasets:
                ########
                if use_cheating_approximation:
                    if dataset_ == training_dataset_name:
                        gt_prob_approx = 'same'
                    else:
                        gt_prob_approx = 'random'
                ########

                ground_truth_embeddings = create_gt_embeddings(
                    gt_prob_approx=gt_prob_approx,
                    embeddings_per_dataset=embeddings_per_dataset,
                    dataset_=dataset_,
                )

                uq_results[uq_name][loss_function_name][dataset_] = uncertainty_func(
                    logits_gt=ground_truth_embeddings,
                    logits_pred=embeddings_per_dataset[dataset_]
                )
    return uq_results, embeddings_per_dataset, targets_per_dataset


def create_gt_embeddings(
        gt_prob_approx: str,
        embeddings_per_dataset: dict,
        dataset_: str,
) -> np.ndarray:
    """
    The function returns Ground Truth embeddings, depending on the strategy.
    If 'same' -- GT will be approximated the same way as prediction ensemble
    If 'flat' -- GT will be always flat with different evidence
    If 'diract' -- GT will be random samples of diracs
    If 'random' -- GT is random samples from simplex.
    """
    n_members, n_objects, n_classes = embeddings_per_dataset[dataset_].shape
    if gt_prob_approx == 'same':
        ground_truth_embeddings = embeddings_per_dataset[dataset_]
    elif gt_prob_approx == 'flat':
        ground_truth_embeddings = (
            np.ones_like(embeddings_per_dataset[dataset_]) *
            np.random.randn(
                *embeddings_per_dataset[dataset_].shape[:-1], 1)
        )
    elif gt_prob_approx == 'diracs':
        epsilon = 1e-3
        ground_truth_embeddings = np.ones_like(
            embeddings_per_dataset[dataset_]) * epsilon
        random_indices = np.random.randint(
            n_classes, size=(n_members, n_objects))
        ground_truth_embeddings[np.arange(n_members)[:, None], np.arange(
            n_objects), random_indices] = 1 - epsilon * (n_classes - 1)
        ground_truth_embeddings = np.log(ground_truth_embeddings)
    elif gt_prob_approx == 'random':
        alpha = np.ones(n_classes)
        ground_truth_embeddings = np.log(
            np.random.dirichlet(alpha, size=(n_members, n_objects)))
    else:
        raise ValueError('No such gt_prob_approx')

    return ground_truth_embeddings


def get_predicted_labels(
    embeddings_per_dataset: dict,
    training_dataset_name: str,
):
    """
    The function returns predicted labels given embeddings for a given dataset
    """
    pred_labels = np.argmax(
        posterior_predictive(
            embeddings_per_dataset[training_dataset_name])[0],
        axis=-1
    )
    return pred_labels

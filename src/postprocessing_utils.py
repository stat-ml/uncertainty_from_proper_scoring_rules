import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Optional
from sklearn.metrics import roc_auc_score
from src.evaluation_utils import collect_stats
from evaluation_utils import collect_embeddings
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

                uq_results[uq_name][loss_function_name][dataset_] = \
                    uncertainty_func(
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

    for uq_name, _ in uq_funcs_with_names:
        roc_auc_dict[uq_name] = {}
        print(f'OOD computed via {uq_name}')

        for ood_dataset in list_ood_datasets:
            roc_auc_dict[uq_name][ood_dataset] = {}
            for loss_ in uq_results[uq_name].keys():
                y_true = np.hstack(
                    [
                        np.ones(uq_results[uq_name][loss_]
                                [ood_dataset].shape[0]),
                        np.zeros(uq_results[uq_name][loss_]
                                 [ind_dataset].shape[0]),
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

                print(
                    (
                        f'InD: {ind_dataset} \t '
                        f'OOD: {ood_dataset} \t '
                        f'loss: {loss_} \t '
                        f'roc_auc: {score}'
                    )
                )

    data_list = []
    for metric_name, datasets in roc_auc_dict.items():
        for dataset_name, loss_functions in datasets.items():
            for loss_function_name, value in loss_functions.items():
                data_list.append((metric_name, dataset_name,
                                 loss_function_name, value))

    # Create a DataFrame
    df = pd.DataFrame(data_list, columns=[
                      "UQMetric", "Dataset", "LossFunction", "RocAucScore"])

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

    for uq_name, _ in uq_funcs_with_names:
        roc_auc_dict[uq_name] = {}
        print(f'Misclassification computed via {uq_name}')

        for loss_ in uq_results[uq_name].keys():
            y_true = (true_labels != pred_labels).astype(np.int32)
            y_score = uq_results[uq_name][loss_][ind_dataset]

            score = roc_auc_score(y_true=y_true, y_score=y_score)
            roc_auc_dict[uq_name][loss_] = score

            print(
                f'InD: {ind_dataset} \t loss: {loss_} \t roc_auc: {score}')

    data_list_misclassification = []
    for metric_name, loss_function in roc_auc_dict.items():
        for loss_function_name, value in loss_function.items():
            data_list_misclassification.append(
                (metric_name, loss_function_name, value))

    # Create a DataFrame
    df_misclassification = pd.DataFrame(
        data_list_misclassification,
        columns=[
            "UQMetric",
            "LossFunction",
            "RocAucScore",
        ]
    )
    return df_misclassification


def make_aggregation(
    df_: pd.DataFrame,
    prefix_: str,
    suffix_: Optional[str],
    dataset_: Optional[str] = None,
    by_loss_function_: bool = False
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
        dataset_mask = df['Dataset'].str.fullmatch(dataset_)
    if suffix_ is not None:
        filtered_df = df[dataset_mask & df['UQMetric'].str.startswith(
            prefix_) & df['UQMetric'].str.endswith(suffix_)]
        if by_loss_function_:
            roc_auc_scores = filtered_df.groupby(
                ['UQMetric', 'LossFunction']
            ).agg({'RocAucScore': ['mean', 'std', 'count']})
        else:
            roc_auc_scores = filtered_df.groupby(['UQMetric',]).agg({
                'RocAucScore': ['mean', 'std', 'count']
            })
    else:
        filtered_df = df[dataset_mask & df['UQMetric'].str.startswith(prefix_)]
        filtered_df.loc[:, 'GeneralizedMetric'] = \
            filtered_df['UQMetric'].apply(
                lambda x: re.sub(r'\s+(Inner|Outer)$', '', x)
        )

        # Group by 'GeneralizedMetric' and calculate mean ROC AUC score
        if by_loss_function_:
            roc_auc_scores = \
                filtered_df.groupby(
                    ['GeneralizedMetric', 'LossFunction']
                ).agg({'RocAucScore': ['mean', 'std', 'count']})
        else:
            roc_auc_scores = filtered_df.groupby(['GeneralizedMetric',]).agg({
                'RocAucScore': ['mean', 'std', 'count']
            })
    # roc_auc_scores = \
    # filtered_df.groupby(['UQMetric', 'LossFunction'])['RocAucScore'].mean()

    roc_auc_df = roc_auc_scores.reset_index()
    roc_auc_df.rename(columns={'RocAucScore': 'MeanRocAucScore'}, inplace=True)
    return roc_auc_df


def get_raw_scores_dataframe(
    uq_results: dict,
):
    # Convert the nested dictionary into a DataFrame
    data = []
    for uq_name, loss_funcs in uq_results.items():
        for loss_name, datasets in loss_funcs.items():
            for dataset_name, scores in datasets.items():
                data.append({
                    'UQ Method': uq_name,
                    'Loss Function': loss_name,
                    'Dataset': dataset_name,
                    'Scores': scores
                })

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
            df['Loss Function'].str.startswith(loss_function) &
            df['Dataset'].str.startswith(dataset) &
            df['UQ Method'].str.startswith(f'Total {score}')
        )].Scores.values[0]

    exc_o = df[
        (df['Loss Function'].str.startswith(loss_function) &
         df['Dataset'].str.startswith(dataset) &
         df['UQ Method'].str.startswith(f'Excess {score} Outer')
         )].Scores.values[0]
    exc_i = df[
        (df['Loss Function'].str.startswith(loss_function) &
         df['Dataset'].str.startswith(dataset) &
         df['UQ Method'].str.startswith(f'Excess {score} Inner')
         )].Scores.values[0]

    bay_o = df[
        (df['Loss Function'].str.startswith(loss_function) &
         df['Dataset'].str.startswith(dataset) &
         df['UQ Method'].str.startswith(f'Bayes {score} Outer')
         )].Scores.values[0]
    bay_i = df[
        (df['Loss Function'].str.startswith(loss_function) &
         df['Dataset'].str.startswith(dataset) &
         df['UQ Method'].str.startswith(f'Bayes {score} Inner')
         )].Scores.values[0]

    bregman = df[
        (df['Loss Function'].str.startswith(loss_function) &
         df['Dataset'].str.startswith(dataset) &
         df['UQ Method'].str.startswith(f'Bregman Information {score}')
         )].Scores.values[0]

    scores_dict_ = {
        'Total': tot,

        "Excess outer": exc_o,
        "Bayes outer": bay_o,

        "Excess inner": exc_i,
        "Bayes inner": bay_i,

        "Bregman Information": bregman,

        'Total outer': exc_o + bay_o,
        'Total inner': exc_i + bay_i,
    }
    scores_df = pd.DataFrame.from_dict(scores_dict_)

    return scores_df

from collections import defaultdict, namedtuple

import matplotlib
import numpy as np
import pandas as pd
from psruq.source.postprocessing_utils import (
    get_missclassification_dataframe,
    get_ood_detection_dataframe,
    get_predicted_labels,
    get_raw_scores_dataframe,
    get_sampled_combinations_uncertainty_scores,
)


def pretty_matplotlib_config(fontsize=15, fonttype=42):
    matplotlib.rcParams["pdf.fonttype"] = fonttype
    matplotlib.rcParams["ps.fonttype"] = fonttype
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams.update({"font.size": fontsize})


ood_detection_pairs_ = [
    ("cifar10", "cifar100"),
    ("cifar10", "svhn"),
    ("cifar10", "blurred_cifar10"),
    ("cifar10", "blurred_cifar100"),
    ("cifar100", "cifar10"),
    ("cifar100", "svhn"),
    ("cifar100", "blurred_cifar100"),
    ("cifar100", "blurred_cifar10"),
]

# scores_dict_to_lists = {
#     "InD": [],
#     "ScoringRule": [],
#     "Bayes": [],
#     "Excess": [],
#     "Total": [],
#     "Bregman Information": [],
#     "Reverse Bregman Information": [],
#     "Expected Pairwise Bregman Information": [],
#     "Bias": [],
#     "MV": [],
#     "MVBI": [],
#     "BiasBI": [],
# }


def aggregate_over_measures(
    dataframe_: pd.DataFrame,
    agg_func_: str,
    by_: list,
):
    measures = [c for c in dataframe_.columns if c not in ["OOD", "InD", "ScoringRule"]]
    res_dict_ = {val: [agg_func_] for val in measures}
    aggregated_ = dataframe_.groupby(by=by_).agg(res_dict_)
    return aggregated_


def collect_scores_into_dict_miss(
    dataframes_list_,
):
    scores_dict_ = defaultdict(
        list, {val: [] for val in dataframes_list_[0].RiskType.unique()}
    )
    std_dict_ = defaultdict(
        list, {val: [] for val in dataframes_list_[0].RiskType.unique()}
    )

    for dataframe_ in dataframes_list_:
        for ind in dataframe_.training_dataset.unique():
            df_aux_ = dataframe_[(dataframe_["training_dataset"] == ind)]

            mean_rocauc_dict = dict(
                df_aux_.groupby(by=["RiskType"])
                .agg({"RocAucScore": ["mean"]})[("RocAucScore", "mean")]
                .reset_index()
                .values
            )
            std_rocauc_dict = dict(
                df_aux_.groupby(by=["RiskType"])
                .agg({"RocAucScore": ["std"]})[("RocAucScore", "std")]
                .reset_index()
                .values
            )
            next_iter = True
            for k in mean_rocauc_dict:
                if k in scores_dict_:
                    scores_dict_[k].append(mean_rocauc_dict[k])
                    std_dict_[k].append(std_rocauc_dict[k])
                    next_iter = False
            if next_iter:
                continue

            scores_dict_["InD"].append(ind)
            scores_dict_["ScoringRule"].append(df_aux_["LossFunction"].unique())

            std_dict_["InD"].append(ind)
            std_dict_["ScoringRule"].append(df_aux_["LossFunction"].unique())
    return scores_dict_, std_dict_


def collect_scores_into_dict(
    dataframes_list,
    ood_detection_pairs,
):
    scores_dict_ = defaultdict(
        list, {val: [] for val in dataframes_list[0].RiskType.unique()}
    )
    std_dict_ = defaultdict(
        list, {val: [] for val in dataframes_list[0].RiskType.unique()}
    )

    scores_dict_["OOD"] = []
    std_dict_["OOD"] = []

    for dataframe_ in dataframes_list:
        for ind, ood in ood_detection_pairs:
            df_aux_ = dataframe_[
                (dataframe_["training_dataset"] == ind) & (dataframe_["Dataset"] == ood)
            ]

            mean_rocauc_dict = dict(
                df_aux_.groupby(by=["RiskType"])
                .agg({"RocAucScore": ["mean"]})[("RocAucScore", "mean")]
                .reset_index()
                .values
            )
            std_rocauc_dict = dict(
                df_aux_.groupby(by=["RiskType"])
                .agg({"RocAucScore": ["std"]})[("RocAucScore", "std")]
                .reset_index()
                .values
            )

            # next_iter = True
            for k in mean_rocauc_dict:
                if k in scores_dict_:
                    scores_dict_[k].append(mean_rocauc_dict[k])
                    std_dict_[k].append(std_rocauc_dict[k])
                    # next_iter = False
            # if next_iter:
            #     continue

            scores_dict_["InD"].append(ind)
            scores_dict_["OOD"].append(ood)
            scores_dict_["ScoringRule"].append(df_aux_["LossFunction"].unique())

            std_dict_["InD"].append(ind)
            std_dict_["OOD"].append(ood)
            std_dict_["ScoringRule"].append(df_aux_["LossFunction"].unique())
    return scores_dict_, std_dict_


def extract_same_different_dataframes(
    dataframe_: pd.DataFrame,
):
    df = dataframe_.copy()
    df_logscore_logscore = df[
        (df["base_rule"] == "Logscore") & (df["LossFunction"] == "Logscore")
    ]
    df_brier_brier = df[(df["base_rule"] == "Brier") & (df["LossFunction"] == "Brier")]
    df_spherical_spherical = df[
        (df["base_rule"] == "Spherical") & (df["LossFunction"] == "Spherical")
    ]

    df_logscore_not_logscore = df[
        (df["base_rule"] != "Logscore") & (df["LossFunction"] == "Logscore")
    ]
    df_brier_not_brier = df[
        (df["base_rule"] != "Brier") & (df["LossFunction"] == "Brier")
    ]
    df_spherical_not_spherical = df[
        (df["base_rule"] != "Spherical") & (df["LossFunction"] == "Spherical")
    ]

    dataframes_ = namedtuple(
        "SameDiffDF",
        [
            "logscore_logscore",
            "brier_brier",
            "spherical_spherical",
            "logscore_not_logscore",
            "brier_not_brier",
            "spherical_not_spherical",
        ],
    )
    return dataframes_(
        logscore_logscore=df_logscore_logscore,
        brier_brier=df_brier_brier,
        spherical_spherical=df_spherical_spherical,
        logscore_not_logscore=df_logscore_not_logscore,
        brier_not_brier=df_brier_not_brier,
        spherical_not_spherical=df_spherical_not_spherical,
    )


def build_tables(
    training_dataset_names: list[str],
    loss_function_names: list[str],
    list_extraction_datasets: list[str],
    list_ood_datasets: list[str],
    model_ids: np.ndarray,
    temperature: float,
    use_cached: bool,
):
    full_dataframe = None
    full_ood_rocauc_dataframe = None
    full_mis_rocauc_dataframe = None

    for training_dataset_name in training_dataset_names:
        if training_dataset_name not in [
            "missed_class_cifar10",
            "noisy_cifar10",
            "noisy_cifar100",
        ]:
            architectures = ["resnet18", "vgg"]
            training_dataset_name_aux = training_dataset_name
        else:
            architectures = ["resnet18"]
            training_dataset_name_aux = training_dataset_name.split("_")[-1]
        for architecture in architectures:
            uq_results, embeddings_per_dataset, targets_per_dataset = (
                get_sampled_combinations_uncertainty_scores(
                    loss_function_names=loss_function_names,
                    training_dataset_name=training_dataset_name,
                    architecture=architecture,
                    model_ids=model_ids,
                    list_extraction_datasets=list_extraction_datasets,
                    temperature=temperature,
                    use_different_approximations=False,
                    use_cached=use_cached,
                )
            )

            df_ood = get_ood_detection_dataframe(
                ind_dataset=training_dataset_name_aux,
                uq_results=uq_results,
                list_ood_datasets=list_ood_datasets,
            )
            df_ood["architecture"] = architecture
            df_ood["training_dataset"] = training_dataset_name

            max_ind = int(
                targets_per_dataset[training_dataset_name_aux].shape[0] / len(model_ids)
            )
            true_labels = targets_per_dataset[training_dataset_name_aux][:max_ind]

            pred_labels = get_predicted_labels(
                embeddings_per_dataset=embeddings_per_dataset,
                training_dataset_name=training_dataset_name_aux,
            )

            df_misclassification = get_missclassification_dataframe(
                ind_dataset=training_dataset_name_aux,
                uq_results=uq_results,
                true_labels=true_labels,
                pred_labels=pred_labels,
            )
            df_misclassification["architecture"] = architecture
            df_misclassification["training_dataset"] = training_dataset_name

            scores_df_unravel = get_raw_scores_dataframe(uq_results=uq_results)
            scores_df_unravel["architecture"] = architecture
            scores_df_unravel["training_dataset"] = training_dataset_name

            if full_dataframe is None:
                full_dataframe = scores_df_unravel
                full_ood_rocauc_dataframe = df_ood
                full_mis_rocauc_dataframe = df_misclassification
            else:
                full_dataframe = pd.concat([full_dataframe, scores_df_unravel])
                full_ood_rocauc_dataframe = pd.concat(
                    [full_ood_rocauc_dataframe, df_ood]
                )
                full_mis_rocauc_dataframe = pd.concat(
                    [full_mis_rocauc_dataframe, df_misclassification]
                )

    return full_dataframe, full_ood_rocauc_dataframe, full_mis_rocauc_dataframe


def stratify_measure(
    table: pd.DataFrame,
    base_score_dict: dict,
):
    pattern_baserule = r"(Logscore|Brier|Neglog|Maxprob|Spherical)"
    pattern_risk = r"(Total|Bayes|Excess|Reverse Bregman Information|Bregman Information|Expected Pairwise Bregman Information|MVBI|MV|BiasBI|Bias)"

    table["base_rule"] = table["UQMetric"].str.extract(pattern_baserule)
    table["RiskType"] = table["UQMetric"].str.extract(pattern_risk)
    table["LossFunction"] = table["LossFunction"].replace(base_score_dict)

    return table


# if __name__ == '__main__':
#     base_score_dict_ = {
#         "cross_entropy": "Logscore",
#         "brier_score": "Brier",
#         "spherical_score": "Spherical",
#     }

#     training_dataset_names_ = [
#         'cifar10',
#         'cifar100',
#         'noisy_cifar100',
#         'missed_class_cifar10',
#         'noisy_cifar10',
#     ]

#     list_extraction_datasets_ = [
#         'cifar10',
#         'cifar100',
#         'svhn',
#         'blurred_cifar100',
#         'blurred_cifar10',
#     ]

#     list_ood_datasets_ = [el for el in list_extraction_datasets_]
#     loss_function_names_ = [
#         'brier_score',
#         'cross_entropy',
#         'spherical_score'
#     ]
#     model_ids_ = np.arange(20)
#     temperature_ = 1.0
#     use_cached_ = True

#     full_tab, ood_rocauc, mi_rocauc = build_tables(
#         list_extraction_datasets=list_extraction_datasets_,
#         training_dataset_names=training_dataset_names_,
#         loss_function_names=loss_function_names_,
#         list_ood_datasets=list_ood_datasets_,
#         model_ids=model_ids_,
#         temperature=temperature_,
#         use_cached=use_cached_,
#     )

#     for name, t_ in [
#         ('full_dataframe', full_tab),
#         ('full_ood_rocauc', ood_rocauc),
#         ('full_mis_rocauc', mi_rocauc),
#     ]:
#         t_s = stratify_measure(
#             table=t_,
#             base_score_dict=base_score_dict_,
#         )
#         t_s.to_csv(os.path.join('tables', f'{name}.csv'), index=False)

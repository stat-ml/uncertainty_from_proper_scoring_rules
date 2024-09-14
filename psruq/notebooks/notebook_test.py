from source.source.postprocessing_utils import (
    get_uncertainty_scores,
    get_predicted_labels,
    get_missclassification_dataframe,
    get_ood_detection_dataframe,
    get_raw_scores_dataframe,
)

from source.datasets.constants import DatasetName
from source.losses.constants import LossName
from source.models.constants import ModelName
from source.metrics import (
    ApproximationType,
    GName,
    RiskType,
)

import pandas as pd
import numpy as np

pd.set_option("display.max_rows", None)


base_score_dict = {
    "cross_entropy": "Logscore",
    "brier_score": "Brier",
    "spherical_score": "Spherical",
}

training_dataset_names = [
    "cifar10",
    "cifar100",
    "noisy_cifar100",
    "missed_class_cifar10",
    "noisy_cifar10",
]
temperature = 1.0
model_ids = np.arange(20)

list_extraction_datasets = [
    "cifar10",
    "cifar100",
    "svhn",
    "blurred_cifar100",
    "blurred_cifar10",
]
list_ood_datasets = [el for el in list_extraction_datasets]

loss_function_names = [el for el in LossName]

full_dataframe = None
full_ood_rocauc_dataframe = None
full_mis_rocauc_dataframe = None


for training_dataset_name in training_dataset_names:
    if training_dataset_name not in [
        "missed_class_cifar10",
        "noisy_cifar10",
        "noisy_cifar100",
    ]:
        architectures = [ModelName.RESNET18, ModelName.VGG19]
        training_dataset_name_aux = training_dataset_name
    else:
        architectures = ["resnet18"]
        training_dataset_name_aux = training_dataset_name.split("_")[-1]
    for architecture in architectures:
        # try:
        uq_results, embeddings_per_dataset, targets_per_dataset = (
            get_uncertainty_scores(
                loss_function_names=loss_function_names,
                training_dataset_name=training_dataset_name,
                architecture=architecture,
                model_ids=model_ids,
                list_extraction_datasets=list_extraction_datasets,
                temperature=temperature,
                use_cached=False,
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

        # except Exception as ex:
        #     print(training_dataset_name, ex)
        #     continue

        scores_df_unravel = get_raw_scores_dataframe(uq_results=uq_results)
        scores_df_unravel["architecture"] = architecture
        scores_df_unravel["training_dataset"] = training_dataset_name

        if full_dataframe is None:
            full_dataframe = scores_df_unravel
            full_ood_rocauc_dataframe = df_ood
            full_mis_rocauc_dataframe = df_misclassification
        else:
            full_dataframe = pd.concat([full_dataframe, scores_df_unravel])
            full_ood_rocauc_dataframe = pd.concat([full_ood_rocauc_dataframe, df_ood])
            full_mis_rocauc_dataframe = pd.concat(
                [full_mis_rocauc_dataframe, df_misclassification]
            )

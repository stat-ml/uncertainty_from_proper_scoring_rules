import pandas as pd
import numpy as np
from src.evaluation_utils import collect_stats


def get_metrics_results(
        loss_function_names: list[str],
        training_dataset_name: str,
        architecture: str,
        model_ids: np.ndarray
):
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

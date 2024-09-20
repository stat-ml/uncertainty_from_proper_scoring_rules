import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from pretrained_models.source.std_loading import cust_load_model
from pretrained_models.source.utils import (
    ROOT_PATH,
    make_logits_path,
    make_model_load_path,
)
from source.datasets.constants import DatasetName
from source.source.evaluation_utils import (
    get_additional_evaluation_metrics,
    load_dataloader_for_extraction,
    load_dict,
    save_dict,
)


def save_additional_stats(
    dataset_name_: str,
    model_id_: int,
):
    logits_path = make_logits_path(
        version=model_id_,
        training_dataset_name=dataset_name_,
        extraction_dataset_name=dataset_name_,
    )

    # Loading the dictionary from the file
    loaded_dict = load_dict(load_path=logits_path)

    actual_acc = get_additional_evaluation_metrics(embeddings_dict=loaded_dict)

    try:
        with open(
            os.path.join(Path(logits_path).parent, "results_dict.json"), "w"
        ) as file:
            json.dump(
                fp=file,
                obj=actual_acc,
                indent=4,
            )
    except OSError:
        import pdb

        pdb.set_trace()
        print("oh")


def extract_embeddings(
    training_dataset_name: str,
    extraction_dataset_name: str,
    num_classes: int,
    model_id: int,
    severity: int | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = make_model_load_path(
        version=model_id, training_dataset=training_dataset_name
    )
    model = cust_load_model(
        style="cifar" if training_dataset_name.startswith("cifar") else "imagenet",
        num_classes=num_classes,
        arch=18,
        path=model_path,
        conv_bias=False,
    )

    model = model.to(device)
    model.eval()

    loader = load_dataloader_for_extraction(
        training_dataset_name=training_dataset_name,
        extraction_dataset_name=extraction_dataset_name,
        severity=severity,
    )

    output_embeddings = {}
    output_embeddings["embeddings"] = []
    output_embeddings["labels"] = []

    with torch.no_grad():
        for _, (inputs, targets) in tqdm(enumerate(loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            output_embeddings["embeddings"].append(outputs.cpu().numpy())
            output_embeddings["labels"].append(targets.cpu().numpy())
    output_embeddings["embeddings"] = np.vstack(output_embeddings["embeddings"])
    output_embeddings["labels"] = np.hstack(output_embeddings["labels"])

    # Saving the dictionary to a file using pickle
    save_path = make_logits_path(
        extraction_dataset_name=extraction_dataset_name,
        training_dataset_name=training_dataset_name,
        version=model_id,
        severity=severity,
    )
    save_dict(save_path=save_path, dict_to_save=output_embeddings)


if __name__ == "__main__":
    training_datasets = [
        # DatasetName.CIFAR10.value,
        DatasetName.CIFAR100.value,
    ]
    model_ids = np.arange(1)

    # iterate over training datasets
    for training_dataset_name in training_datasets:
        if training_dataset_name in [
            DatasetName.CIFAR100.value,
            DatasetName.CIFAR100_NOISY_LABEL.value,
        ]:
            n_classes = 100
        elif training_dataset_name in [
            DatasetName.TINY_IMAGENET.value,
        ]:
            n_classes = 200
        else:
            n_classes = 10

        for extraction_dataset_name in [
            # DatasetName.CIFAR10.value,
            # DatasetName.CIFAR100.value,
            # DatasetName.SVHN.value,
            # DatasetName.CIFAR10_BLURRED.value,
            # DatasetName.CIFAR100_BLURRED.value,
            # DatasetName.TINY_IMAGENET.value,
            # DatasetName.CIFAR10C.value,
            DatasetName.CIFAR100C.value,
        ]:
            # different loss functions
            for model_id in model_ids:
                # and different ensemble members
                print(
                    (
                        f"Training dataset: {training_dataset_name} ..."
                        f"Extraction dataset: {extraction_dataset_name} "
                        f"Loading resnet18, "
                        f"model_id={model_id} "
                    )
                )
                print("Extracting embeddings....")
                if extraction_dataset_name in [
                    DatasetName.CIFAR100C.value,
                    DatasetName.CIFAR10C.value,
                ]:
                    for severity in range(1, 6):
                        extract_embeddings(
                            training_dataset_name=training_dataset_name,
                            extraction_dataset_name=extraction_dataset_name,
                            num_classes=n_classes,
                            model_id=model_id,
                            severity=severity,
                        )
                else:
                    extract_embeddings(
                        training_dataset_name=training_dataset_name,
                        extraction_dataset_name=extraction_dataset_name,
                        num_classes=n_classes,
                        model_id=model_id,
                        severity=None,
                    )
                print("Finished embeddings extraction!")

                if extraction_dataset_name == training_dataset_name:
                    print("Saving additional evaluation params...")
                    save_additional_stats(
                        dataset_name_=training_dataset_name,
                        model_id_=model_id,
                    )

        # stats_dict = collect_stats(
        #     architecture=architecture,
        #     dataset_name=dataset_name,
        #     loss_function_name=loss_function_name,
        #     model_ids=model_ids,
        # )
        # print(stats_dict)

    print("Finished!")

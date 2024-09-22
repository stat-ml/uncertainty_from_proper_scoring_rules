import numpy as np
import torch
from tqdm.auto import tqdm

from source.datasets.constants import DatasetName
from source.losses.constants import LossName
from source.models.constants import ModelName, ModelSource
from source.models.load_models import load_model_from_source
from source.source.data_utils import load_dataloader_for_extraction, save_dict
from source.source.evaluation_utils import save_additional_stats
from source.source.path_utils import make_logits_path


def extract_logits(
    training_dataset_name: str,
    extraction_dataset_name: str,
    n_classes: int,
    model_id: int,
    severity: int | None,
    architecture: str,
    loss_function_name: str,
    model_source: str,
):
    """The function extracts and save logits for a specific model

    Args:
        architecture (str): _description_
        loss_function_name (str): _description_
        training_dataset_name (str): _description_
        extraction_dataset_name (str): _description_
        model_id (int): _description_
        model_id int | None: _description_
        n_classes (int): _description_
        model_source (ModelSource): _description_
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model_from_source(
        model_source=model_source,
        architecture=architecture,
        training_dataset_name=training_dataset_name,
        extraction_dataset_name=extraction_dataset_name,
        loss_function_name=loss_function_name,
        model_id=model_id,
        n_classes=n_classes,
        device=device,
        severity=severity,
    )
    if model is None:
        return
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
        model_id=model_id,
        severity=severity,
        model_source=model_source,
        architecture=architecture,
        loss_function_name=loss_function_name,
    )
    save_dict(save_path=save_path, dict_to_save=output_embeddings)


if __name__ == "__main__":
    model_source = ModelSource.OUR_MODELS.value
    architecture = ModelName.RESNET18.value  #'vgg'  # 'resnet18' 'vgg'
    training_datasets = [
        DatasetName.CIFAR10.value,
        DatasetName.CIFAR100.value,
        # 'missed_class_cifar10',
        # "noisy_cifar10",
        # "noisy_cifar100",
    ]  # ['cifar10', 'cifar100']
    model_ids = np.arange(20)

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
            DatasetName.CIFAR10C.value,
            DatasetName.TINY_IMAGENET.value,
        ]:
            # iterate over datasets from which we want get embeddings
            for loss_function_name in [el.value for el in LossName]:
                if (model_source == ModelSource.TORCH_UNCERTAINTY.value) and (
                    loss_function_name != LossName.CROSS_ENTROPY.value
                ):
                    continue
                # different loss functions
                for model_id in model_ids:
                    # and different ensemble members
                    print(
                        (
                            f"Training dataset: {training_dataset_name} ..."
                            f"Extraction dataset: {extraction_dataset_name} "
                            f"Loading {architecture}, "
                            f"model_id={model_id} "
                            f"and loss {loss_function_name}"
                        )
                    )
                    print("Extracting embeddings....")

                    if extraction_dataset_name in [
                        # DatasetName.CIFAR100C.value,
                        DatasetName.CIFAR10C.value,
                    ]:
                        for severity in range(1, 6):
                            extract_logits(
                                training_dataset_name=training_dataset_name,
                                extraction_dataset_name=extraction_dataset_name,
                                n_classes=n_classes,
                                model_id=model_id,
                                severity=severity,
                                architecture=architecture,
                                loss_function_name=loss_function_name,
                                model_source=model_source,
                            )
                    else:
                        extract_logits(
                            training_dataset_name=training_dataset_name,
                            extraction_dataset_name=extraction_dataset_name,
                            n_classes=n_classes,
                            model_id=model_id,
                            severity=None,
                            architecture=architecture,
                            loss_function_name=loss_function_name,
                            model_source=model_source,
                        )
                    print("Finished embeddings extraction!")

                    if extraction_dataset_name == training_dataset_name:
                        print("Saving additional evaluation params...")
                        save_additional_stats(
                            dataset_name=training_dataset_name,
                            model_id=model_id,
                            architecture=architecture,
                            loss_function_name=loss_function_name,
                        )

        # stats_dict = collect_stats(
        #     architecture=architecture,
        #     dataset_name=dataset_name,
        #     loss_function_name=loss_function_name,
        #     model_ids=model_ids,
        # )
        # print(stats_dict)

    print("Finished!")

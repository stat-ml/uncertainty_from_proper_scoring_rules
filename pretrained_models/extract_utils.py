import torch
from tqdm.auto import tqdm
import os
from source.source.evaluation_utils import (
    save_dict,
    load_dataloader_for_extraction,
    get_additional_evaluation_metrics,
)
from scripts.std_loading import cust_load_model
from scripts.utils import ROOT_PATH, make_model_load_path, make_logits_path
import numpy as np
import json


# def save_additional_stats(
#     dataset_name: str,
#     model_id: int,
# ):
#     load_path = make_model_load_path(
#         version=model_id,
#         training_dataset=dataset_name,
#     )

#     logits_path = make_logits_path(
#         version=model_id,
#         training_dataset_name=dataset_name,
#         extraction_dataset_name=training_dataset_name,
#     )

#     actual_acc = get_additional_evaluation_metrics(embeddings_dict=logits_path)
#     actual_acc.update({"last_acc": last_acc / 100})

#     try:
#         with open(os.path.join(logits_path, "results_dict.json"), "w") as file:
#             json.dump(
#                 fp=file,
#                 obj=actual_acc,
#                 indent=4,
#             )
#     except OSError:
#         import pdb

#         pdb.set_trace()
#         print("oh")


def extract_embeddings(
    training_dataset_name: str,
    extraction_dataset_name: str,
    num_classes: int,
    model_id: int,
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
    )
    save_dict(save_path=save_path, dict_to_save=output_embeddings)


if __name__ == "__main__":
    training_datasets = ["cifar10", "cifar100", "tiny_imagenet"]
    model_ids = np.arange(2)

    # iterate over training datasets
    for training_dataset_name in training_datasets:
        if training_dataset_name in ["cifar100", "noisy_cifar100"]:
            n_classes = 100
        elif training_dataset_name in ["tiny_imagenet"]:
            n_classes = 200
        else:
            n_classes = 10

        for extraction_dataset_name in [
            "cifar10",
            "cifar100",
            "svhn",
            "blurred_cifar100",
            "blurred_cifar10",
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
                extract_embeddings(
                    training_dataset_name=training_dataset_name,
                    extraction_dataset_name=extraction_dataset_name,
                    num_classes=n_classes,
                    model_id=model_id,
                )
                print("Finished embeddings extraction!")

                # if extraction_dataset_name == training_dataset_name:
                #     print("Saving additional evaluation params...")
                #     save_additional_stats(
                #         dataset_name=training_dataset_name,
                #         model_id=model_id,
                #     )

        # stats_dict = collect_stats(
        #     architecture=architecture,
        #     dataset_name=dataset_name,
        #     loss_function_name=loss_function_name,
        #     model_ids=model_ids,
        # )
        # print(stats_dict)

    print("Finished!")

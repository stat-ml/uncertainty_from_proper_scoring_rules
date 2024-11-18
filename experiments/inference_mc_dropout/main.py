import argparse
import logging
import os

import torch
import tqdm

from source.datasets import DatasetName, get_dataloaders
from source.models import ModelName, get_model
from source.source.path_config import REPOSITORY_ROOT

LOGGER = logging.getLogger(__name__)
PWD = os.path.dirname(os.path.realpath(__file__))
ALLOWED_DROPOUT_MODELS = list(filter(
    lambda x: 'dropout' in x,
    [element.value for element in ModelName]
))
                

def get_accuracy(ground_truth: torch.Tensor, probabilities: torch.Tensor) -> float:
    return float(
        (ground_truth == probabilities.argmax(-1)).float().mean().cpu().detach()
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file_path",
        type=str,
        default=None,
        help="Path to the model weights. It will look, using location where the main file is located as root.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar10_one_batch",
        help=f"Which dataset to use. Available options are: {[element.value for element in DatasetName]}",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="resnet18_dropout",
        help=f"For now, only resnet18_dropout is supported.",
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        type=int,
        default=5,
        help=f"Number of samples to use for bayesian inference.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Wether to show additional information or not.",
    )
    parser.add_argument(
        "-t",
        "--include_test",
        action="store_true",
        help="Wether to include inference on test part of the dataset or not.",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        action="store_true",
        help="Which cuda device to use. If set to -1 cpu will be used. Default value is -1.",
    )

    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> argparse.Namespace:
    """
    Validate command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    if arguments.file_path is None:
        raise RuntimeError(
            "File path should be given to command line argument via -f or --file_path argument."
        )

    full_file_path = f"{PWD}/{arguments.file_path}"

    if not os.path.isfile(f"{PWD}/{arguments.file_path}"):
        raise RuntimeError(f"File does not exists on path {full_file_path}.")

    try:
        DatasetName(arguments.dataset)
    except ValueError:
        raise ValueError(
            f"{arguments.dataset} --  no such dataset available. "
            + f"Available options are: {[element.value for element in DatasetName]}"
        )
    
    try:
        ModelName(arguments.model_name)
    except ValueError:
        raise ValueError(
            f"{arguments.model_name} --  no such model type available. "
            + f"Available options are: {ALLOWED_DROPOUT_MODELS}"
        )
    
    if arguments.number_of_samples <= 0:
        raise ValueError("Number of samples should be positive.")

    return arguments


if __name__ == "__main__":
    arguments = validate_arguments(arguments=parse_arguments())

    logger_level = logging.DEBUG if arguments.verbose else logging.WARNING
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logger_level,
    )
    LOGGER.setLevel(logger_level)
    number_of_samples = arguments.number_of_samples
    file_path = arguments.file_path
    path_to_checkpoint = f"{PWD}/{file_path}"
    dataset_type = arguments.dataset
    checkpoint = torch.load(
        f=path_to_checkpoint, map_location=torch.device("cpu"), weights_only=False
    )
    
    try:
        n_classes = checkpoint['net'][list(checkpoint['net'].keys())[-1]].shape[0]
    except:
        raise RuntimeError(f"Hack to determine n_classes didnt work for checkpoint {checkpoint}")
    
    model = get_model(model_name=arguments.model_name, n_classes=n_classes)
    LOGGER.info(f"Model {model.__class__.__name__} loaded.")

    if "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)
    LOGGER.info(f"Weights from {path_to_checkpoint} loaded.")

    classifier = model.linear
    model.dropout = torch.nn.Identity()
    model.linear = torch.nn.Identity()

    trainloader, testloader = get_dataloaders(dataset=arguments.dataset)
    LOGGER.info(f"Dataset {arguments.dataset} loaded.")

    device = (
        torch.device("cuda")
        if arguments.cuda else torch.device("cpu")
    )

    model.to(device=device)
    classifier.to(device=device)

    logits_dict = {
        "dataset": arguments.dataset,
        "model_type": arguments.model_name,
        "logits": torch.Tensor()
    }
    masks = (torch.bernoulli(torch.ones((1, 5, 512)).fill_(0.5)) / 0.5).to(device)

    with torch.no_grad():
        LOGGER.info(f"Computing logits on train part of the dataset.")
        for X, y in tqdm.tqdm(
            iterable=trainloader, total=len(trainloader), disable=(not arguments.verbose)
        ):
            X = X.to(device)
            y = y.to(device)

            batch_features = torch.stack([
                model(X)
                for _ in range(number_of_samples)
            ], dim=1)
            batch_features_droped_out = batch_features * masks
            batch_logits = classifier(batch_features_droped_out)
            probabilities = torch.nn.functional.softmax(
                batch_logits, dim=-1
            )
            distribution_prediction = probabilities.mean(dim=1)
            logits_dict["logits"] = torch.cat([
                logits_dict["logits"],
                batch_logits.to("cpu")
            ], dim=0)
    
    if arguments.include_test:    
        LOGGER.info(f"Computing logits on test loader.")
        with torch.no_grad():
            for X, y in tqdm.tqdm(
                iterable=testloader, total=len(testloader), disable=(not arguments.verbose)
            ):
                X = X.to(device)
                y = y.to(device)

                batch_features = torch.stack([
                    model(X)
                    for _ in range(number_of_samples)
                ], dim=1)
                batch_features_droped_out = batch_features * masks
                batch_logits = classifier(batch_features_droped_out)
                probabilities = torch.nn.functional.softmax(
                    batch_logits, dim=-1
                )
                distribution_prediction = probabilities.mean(dim=1)
                logits_dict["logits"] = torch.cat([
                    logits_dict["logits"],
                    batch_logits.to("cpu")
                ], dim=0)
    else:
        LOGGER.info(f"Computing logits on test part is skipped, if you want to include test, pass --include_test flag.")

    architecture = model.__class__.__name__.lower()
    dataset_type = arguments.dataset
    experiment_folder_root = f"{REPOSITORY_ROOT}/experiments/inference_mc_dropout" 
    output_folder_path = f"{experiment_folder_root}/results/{dataset_type}/{architecture}"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_dict_path = f"{output_folder_path}/logits_dict.pth"

    torch.save(logits_dict, output_dict_path)

    LOGGER.info(f"Logits are saved to {output_dict_path}")

    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

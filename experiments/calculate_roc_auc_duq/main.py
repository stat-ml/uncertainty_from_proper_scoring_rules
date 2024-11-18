import argparse
import itertools
import logging
import os

import torch
import tqdm
from sklearn import metrics

from source.datasets import DatasetName, get_dataloaders
from source.models import ModelName, get_model
from source.models.resnet_duq import DUQNN
from source.source.path_config import REPOSITORY_ROOT

LOGGER = logging.getLogger(__name__)
PWD = os.path.dirname(os.path.realpath(__file__))
AVAILABLE_MODELS = list(filter(
    lambda x: "duq" in x,
    [element.value for element in ModelName]
))


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
        "--ood_dataset",
        type=str,
        default="cifar10_one_batch",
        help=f"Which dataset to use. Available options are: {[element.value for element in DatasetName]}",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="resnet18_duq",
        help=f"For now, only resnet18_duq is supported.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Wether to show additional information or not.",
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
        DatasetName(arguments.ood_dataset)
    except ValueError:
        raise ValueError(
            f"{arguments.ood_dataset} --  no such dataset available. "
            + f"Available options are: {[element.value for element in DatasetName]}"
        )
    
    try:
        ModelName(arguments.model_name)
    except ValueError:
        raise ValueError(
            f"{arguments.model_name} --  no such model type available. "
            + f"Available options are: {AVAILABLE_MODELS}"
        )
    
    return arguments


if __name__ == "__main__":
    arguments = validate_arguments(arguments=parse_arguments())

    logger_level = logging.DEBUG if arguments.verbose else logging.WARNING
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logger_level,
    )
    LOGGER.setLevel(logger_level)
    file_path = arguments.file_path
    path_to_checkpoint = f"{PWD}/{file_path}"
    dataset_type = arguments.dataset
    ood_dataset_type = arguments.ood_dataset

    checkpoint = torch.load(
        f=path_to_checkpoint, map_location=torch.device("cpu"), weights_only=False
    )
    try:
        n_classes = 10
    except:
        raise RuntimeError(f"Hack to determine n_classes didnt work for checkpoint {checkpoint}")
    
    model = get_model(model_name=arguments.model_name, n_classes=n_classes)
    LOGGER.info(f"Model {model.__class__.__name__} loaded.")
 
    if "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)
    LOGGER.info(f"Weights from {path_to_checkpoint} loaded.")

    trainloader, testloader = get_dataloaders(dataset=arguments.dataset)
    LOGGER.info(f"Dataset {arguments.dataset} loaded.")

    ood_trainloader, ood_testloader = get_dataloaders(dataset=arguments.ood_dataset)
    LOGGER.info(f"Dataset {arguments.ood_dataset} loaded.")

    device = (
        torch.device("cuda")
        if arguments.cuda else torch.device("cpu")
    )

    model.to(device=device)
    auc_dict = {
        "dataset": arguments.dataset,
        "ood_dataset": arguments.ood_dataset,
        "model_type": arguments.model_name,
    }
    model.eval()
    assert type(model) is DUQNN

    id_uncertainties = torch.Tensor()
    ood_uncertainties = torch.Tensor()

    with torch.no_grad():
        LOGGER.info(f"Computing uncertainty on test part of iD ({dataset_type}) dataset.")
        for X, _ in tqdm.tqdm(
            iterable=testloader, total=len(testloader), disable=(not arguments.verbose)
        ):
            X = X.to(device)
            uncertainties, _ = model.forward(X).max(dim=1)
            id_uncertainties = torch.cat([
                id_uncertainties,
                uncertainties.to("cpu")
            ], dim=0)

    with torch.no_grad():
        LOGGER.info(f"Computing uncertainty on test part of OOD ({ood_dataset_type})  dataset.")
        for X, _ in tqdm.tqdm(
            iterable=ood_testloader,
            total=len(ood_testloader),
            disable=(not arguments.verbose)
        ):
            X = X.to(device)
            uncertainties, _ = model.forward(X).max(dim=1)
            ood_uncertainties = torch.cat([
                ood_uncertainties,
                uncertainties.to("cpu")
            ], dim=0)


    binary_labels = torch.cat((
        torch.ones(id_uncertainties.shape[0]).to(device),
        torch.zeros(ood_uncertainties.shape[0]).to(device)
    ))
    
    scores = torch.cat((id_uncertainties, ood_uncertainties))
    auroc = metrics.roc_auc_score(binary_labels.cpu().numpy(), scores.cpu().numpy())
    auc_dict['roc_auc'] = float(auroc)
    LOGGER.info(f"ROC AUC: {auroc}")
        
    architecture = model.__class__.__name__.lower()
    dataset_type = arguments.dataset
    experiment_folder_root = f"{REPOSITORY_ROOT}/experiments/calculate_roc_auc_duq" 
    output_folder_path = (
        f"{experiment_folder_root}/results/"
        f"{dataset_type}_vs_ood_{ood_dataset_type}/{architecture}"
    )

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    output_dict_path = f"{output_folder_path}/roc_auc.pth"

    torch.save(auc_dict, output_dict_path)

    LOGGER.info(f"ROC AUC is saved to {output_dict_path}")

    torch.cuda.empty_cache()
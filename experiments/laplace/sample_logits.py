import argparse
import logging
import os

import torch
import torch.utils.data
from laplace import KronLLLaplace
from psruq.datasets import DatasetName, get_dataset_class_instance, get_transforms
from psruq.losses import LossName
from psruq.models import ModelName, get_model

LOGGER = logging.getLogger(__name__)
PWD = os.path.dirname(os.path.realpath(__file__))


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
        help="Path to the model weights. The script will look for the file, using location where the main file is located as root. e.g. chekpoint/model.pth will look for the file at PATH_TO_MAINPY/checkpoint/model.pth",
    )
    parser.add_argument(
        "-o",
        "--out_of_distribution_dataset",
        type=str,
        default=None,
        help=f"Which type of OOD data to use to evaluate logits. Available options are: {[element.value for element in DatasetName]}",
    )
    parser.add_argument(
        "-d",
        "--in_distribution_dataset",
        type=str,
        default="cifar10",
        help=f"Which type of dataset to use to evaluate laplace approximation. Available options are: {[element.value for element in DatasetName]}",
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="CrossEntropy",
        help=f"Loss function type. Available options are: {[element.value for element in LossName]}",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="resnet18",
        help=f"Which model to use. Available options are: {[element.value for element in ModelName]}",
    )
    parser.add_argument(
        "-u",
        "--number_of_classes",
        type=int,
        default=10,
        help="Number of classes to use for prediction.",
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
        type=int,
        default=-1,
        help="Which cuda device to use. If set to -1 cpu will be used. Default value is -1.",
    )
    parser.add_argument(
        "-n",
        "--number_of_weight_samples",
        type=int,
        default=20,
        help="This parameter sets the amount of times the weights are going to be sample from the model distribution.",
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

    full_file_path = os.path.isfile(f"{PWD}/{arguments.file_path}")

    if not os.path.isfile(f"{PWD}/{arguments.file_path}"):
        raise RuntimeError(f"File does not exists on path {full_file_path}.")

    if arguments.out_of_distribution_dataset is None:
        raise RuntimeError(
            "Out of distribution data set be given to command line argument via -o or --out_of_distribution_dataset argument."
        )

    try:
        DatasetName(arguments.out_of_distribution_dataset)
    except ValueError:
        raise ValueError(
            f"{arguments.out_of_distribution_dataset} --  no such dataset available. "
            + f"Available options are: {[element.value for element in DatasetName]}"
        )

    try:
        DatasetName(arguments.in_distribution_dataset)
    except ValueError:
        raise ValueError(
            f"{arguments.in_distribution_dataset} --  no such dataset available. "
            + f"Available options are: {[element.value for element in DatasetName]}"
        )

    try:
        LossName(arguments.loss)
    except ValueError:
        raise ValueError(
            f"{arguments.loss} --  no such loss type available. "
            + f"Available options are: {[element.value for element in LossName]}"
        )

    try:
        ModelName(arguments.model_name)
    except ValueError:
        raise ValueError(
            f"{arguments.model_name} --  no such model type available. "
            + f"Available options are: {[element.value for element in ModelName]}"
        )

    if arguments.cuda != -1:
        torch.device("cuda", index=arguments.cuda)

    return arguments


if __name__ == "__main__":
    arguments = validate_arguments(arguments=parse_arguments())

    if arguments.loss != "CrossEntropy":
        LOGGER.warning(
            (
                "!-----------------------------------------------!"
                "Only CrossEntropy loss is supported for Laplace approximation at the moment.\n"
                "This parameter has no effect on the moddel, CrossEntropy loss will be choosen in all cases.\n"
                "!-----------------------------------------------!"
            )
        )

    logger_level = logging.DEBUG if arguments.verbose else logging.WARNING
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logger_level,
    )
    LOGGER.setLevel(logger_level)

    file_path = arguments.file_path
    path_to_checkpoint = f"{PWD}/{file_path}"

    checkpoint = torch.load(
        f=path_to_checkpoint, map_location=torch.device("cpu"), weights_only=False
    )

    model = get_model(
        model_name=arguments.model_name,
        n_classes=arguments.number_of_classes,
    )
    LOGGER.info(
        f"Model {model.__class__.__name__} with {arguments.number_of_classes} classes loaded."
    )

    if "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)

    LOGGER.info(f"Weights from {path_to_checkpoint} loaded.")

    in_distribution_dataset_class_instance = get_dataset_class_instance(
        dataset=arguments.in_distribution_dataset
    )

    (
        train_in_distribution_dataset_transformations,
        test_in_distribution_dataset_transformations,
    ) = get_transforms(dataset=arguments.in_distribution_dataset)

    trainloader = torch.utils.data.DataLoader(
        dataset=in_distribution_dataset_class_instance(
            root="./data",
            train=True,
            download=True,
            transform=train_in_distribution_dataset_transformations,
        ),
        batch_size=128,
        shuffle=True,
    )

    LOGGER.info(f"In distribution dataset {arguments.in_distribution_dataset} loaded.")

    out_of_distribution_dataset_class_instance = get_dataset_class_instance(
        dataset=arguments.out_of_distribution_dataset
    )

    testloader = torch.utils.data.DataLoader(
        dataset=out_of_distribution_dataset_class_instance(
            root="./data",
            train=False,
            download=True,
            transform=test_in_distribution_dataset_transformations,
        ),
        batch_size=128,
        shuffle=True,
    )

    LOGGER.info(
        f"Out of distribution dataset {arguments.out_of_distribution_dataset} loaded."
    )

    device = (
        torch.device("cuda", index=arguments.cuda)
        if arguments.cuda != -1
        else torch.device("cpu")
    )

    model.to(device=device)

    laplace_model = KronLLLaplace(model=model, likelihood="classification")

    LOGGER.info("Fitting Hessian matrix on train data.")
    laplace_model.fit(train_loader=trainloader)

    logits_tensor = torch.empty(
        size=(0, arguments.number_of_classes, arguments.number_of_weight_samples),
        dtype=torch.float32,
        device="cpu",
    )

    labels = torch.empty(size=(0,), dtype=torch.float32, device="cpu")

    with torch.no_grad():
        for X, y in testloader:
            logits_tensor = torch.cat(
                [
                    logits_tensor,
                    laplace_model._nn_predictive_samples(
                        X=X, n_samples=arguments.number_of_weight_samples
                    )
                    .detach()
                    .cpu()
                    .transpose(0, 1)
                    .transpose(1, 2),
                ],
                dim=0,
            )

            labels = torch.cat([labels, y.detach().cpu()], dim=0)

    LOGGER.info("Logits and labels created.")

    path_to_save_logits = (
        f"{PWD}/"
        f"logits_results/"
        f"checkpoints_{arguments.in_distribution_dataset}/"
        f"{arguments.model_name}/"
        f"{arguments.loss}/"
    )
    pickled_dict_name = f"{arguments.out_of_distribution_dataset}.pth"

    if not os.path.isdir(path_to_save_logits):
        LOGGER.info(
            f"Path {path_to_save_logits} does not exsists. Creating all the folders on path"
        )
        os.makedirs(path_to_save_logits)

    LOGGER.info(f"Saving model weights to {path_to_save_logits}/ckpt.pth")
    torch.save(model.state_dict(), f"{path_to_save_logits}/ckpt.pth")

    LOGGER.info(
        f"Saving logits and labels to {path_to_save_logits}/{pickled_dict_name}.pth"
    )
    torch.save(
        {
            "logits": logits_tensor,
            "logits_shape": "(batch_size, n_classes, n_weight_samples)",
            "labels": labels,
            "labels_shape": "(batch_size, )",
        },
        f"{path_to_save_logits}/{pickled_dict_name}.pth",
    )

    torch.cuda.empty_cache()

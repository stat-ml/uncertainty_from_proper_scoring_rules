import argparse
import logging
import os

import torch
import tqdm
from laplace import KronLLLaplace, LinkApprox, PredType
from psruq.datasets import DatasetName, get_dataloaders
from psruq.losses import LossName, get_loss_function
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
        help="Path to the model weights. It will look, using location where the main file is located as root.",
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="CrossEntropy",
        help=f"Loss function type. Available options are: {[element.value for element in LossName]}",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar10",
        help=f"Which dataset to use. Available options are: {[element.value for element in DatasetName]}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Wether to show additional information or not.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="resnet18",
        help=f"Which model to use. Available options are: {[element.value for element in ModelName]}",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results/experiment.pth",
        help="Path to save results of the experiment.",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        type=int,
        default=-1,
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

    full_file_path = os.path.isfile(f"{PWD}/{arguments.file_path}")

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

    model = get_model(model_name=arguments.model_name)
    LOGGER.info(f"Model {model.__class__.__name__} loaded.")

    if "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    else:
        model.load_state_dict(checkpoint)
    LOGGER.info(f"Weights from {path_to_checkpoint} loaded.")

    trainloader, testloader = get_dataloaders(dataset=arguments.dataset)
    LOGGER.info(f"Dataset {arguments.dataset} loaded.")

    loss_function = get_loss_function(loss_type=arguments.loss)
    LOGGER.info(f"Using {arguments.loss} as loss function.")

    device = (
        torch.device("cuda", index=arguments.cuda)
        if arguments.cuda != -1
        else torch.device("cpu")
    )

    model.to(device=device)
    loss_function.to(device=device)

    laplace_model = KronLLLaplace(model=model, likelihood="classification")

    LOGGER.info("Fitting Hessian matrix on train data.")
    laplace_model.fit(train_loader=trainloader)

    metric_dict = {
        "dataset": arguments.dataset,
        "model_type:": arguments.model_name,
        "loss_type": arguments.loss,
        "test_loss": 0.0,
        "test_accuracy": 0.0,
        "test_instances_count": 0,
    }

    with torch.no_grad():
        for X, y in tqdm.tqdm(
            iterable=testloader, total=len(testloader), disable=(not arguments.verbose)
        ):
            X = X.to(device)
            y = y.to(device)
            distribution_prediction = laplace_model(
                x=X,
                pred_type=PredType.NN,
                link_approx=LinkApprox.MC,
                n_samples=15,
            )

            if type(distribution_prediction) is not torch.Tensor:
                raise RuntimeError(
                    "Laplace model returns a tuple, but tensor is expected."
                )

            loss_value = loss_function(distribution_prediction, y)
            accuracy_value = get_accuracy(y, distribution_prediction)

            batch_size = X.shape[0]
            metric_dict["test_loss"] += float(loss_value.cpu().detach()) * batch_size
            metric_dict["test_accuracy"] += accuracy_value * batch_size
            metric_dict["test_instances_count"] += batch_size

    metric_dict["test_loss"] /= metric_dict["test_instances_count"]
    metric_dict["test_accuracy"] /= metric_dict["test_instances_count"]

    torch.save(metric_dict, arguments.output_path)

    LOGGER.info(f"Validation results:")
    for key, value in metric_dict.items():
        LOGGER.info(f"\t{key}:\t{value}")

    LOGGER.info(f"Metric is saved to {arguments.output_path}")

    torch.cuda.empty_cache()

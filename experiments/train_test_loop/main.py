import argparse
import datetime
import logging
import os

import torch
import torch.optim
import torch.utils
import torch.utils.data

from source.datasets import DatasetName, get_dataloaders
from source.losses import LossName, get_loss_function
from source.models import ModelName, get_model
from source.source.path_config import REPOSITORY_ROOT

LOGGER = logging.getLogger(__name__)
PWD = os.path.dirname(os.path.realpath(__file__))
BEST_ACCURACY = 0

# Training
def train(
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.SGD,
        criterion: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    LOGGER.info(
        (
            f"Training | Epoch: {epoch} "
            f"| Loss: {train_loss / len(dataloader):.3f} "
            f"| Acc: {100.0 * correct / total:.3f}"
        )
    )


def test(
        epoch: int,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ):
    global BEST_ACCURACY
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    LOGGER.info(
        (
            f"Testing | Epoch: {epoch} "
            f"| Loss: {test_loss / len(dataloader):.3f} "
            f"| Acc: {100.0 * correct / total:.3f}"
        )
    )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > BEST_ACCURACY:
        LOGGER.info("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        
        architecture = model.__class__.__name__.lower()
        loss_name = criterion.__class__.__name__.lower()
        dataset_type = dataloader.dataset.__class__.__name__.lower()

        if not os.path.exists(f"{REPOSITORY_ROOT}/checkpoint/{dataset_type}/{architecture}/{loss_name}"):
            os.makedirs(f"{REPOSITORY_ROOT}/checkpoint/{dataset_type}/{architecture}/{loss_name}")

        torch.save(
            state, f"{REPOSITORY_ROOT}/checkpoint/{dataset_type}/{architecture}/{loss_name}/checkpoint.pth"
        )

        BEST_ACCURACY = acc

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
        "--lr", 
        default=0.1,
        type=float,
        help="learning rate. Optimizer by default is SGD(..., momentum=0.9, weight_decay=5e-4)"
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
        "-c",
        "--cuda",
        type=int,
        default=-1,
        help="Which cuda device to use. If set to -1 cpu will be used. Default value is -1.",
    )
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Wether to log information to file or not.",
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
        filename=(
            f"{REPOSITORY_ROOT}/"
            f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_train_test_loop.log"
         ) if arguments.log_to_file else None
    )
    LOGGER.setLevel(logger_level)

    model = get_model(model_name=arguments.model_name, n_classes=100)
    LOGGER.info(f"Model {model.__class__.__name__} loaded.")

    trainloader, testloader = get_dataloaders(dataset=arguments.dataset)
    LOGGER.info(f"Dataset {arguments.dataset} loaded.")

    loss_function = get_loss_function(loss_type=arguments.loss)
    LOGGER.info(f"Using {arguments.loss} as loss function.")

    device = (
        torch.device("cuda", index=arguments.cuda)
        if arguments.cuda != -1
        else torch.device("cpu")
    )

    LOGGER.info(f"Using {device} as device.")

    model.to(device=device)
    loss_function.to(device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=arguments.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(200):

        train(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            criterion=loss_function,
            dataloader=trainloader,
            device=device
        )

        test(
            epoch=epoch,
            model=model,
            criterion=loss_function,
            dataloader=testloader,
            device=device,
        )

        scheduler.step()


    torch.cuda.empty_cache()

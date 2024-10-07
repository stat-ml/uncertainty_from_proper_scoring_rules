"""Train CIFAR10 with PyTorch."""

import argparse
import os

import torch
import torch.optim as optim
from psruq.datasets import get_dataloaders
from psruq.losses import get_loss_function
from psruq.models import get_model
from psruq.utils import progress_bar

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--model_id", type=int, help="model id (for ensembles)", default=0)
parser.add_argument(
    "--architecture",
    choices=["resnet18", "vgg"],
    type=str,
    help="Model architecture.",
    default="resnet18",
)
parser.add_argument(
    "--dataset",
    choices=[
        "cifar10",
        "noisy_cifar10",
        "svhn",
        "missed_class_cifar10",
    ],
    type=str,
    help="Training dataset.",
    default="cifar10",
)
parser.add_argument(
    "--loss",
    choices=["cross_entropy", "brier_score", "spherical_score", "neglog_score"],
    type=str,
    help="Name of the loss function.",
    default="neglog_score",
)
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
args = parser.parse_args()

####
architecture = args.architecture
model_id = args.model_id
loss_name = args.loss
####


device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
trainloader, testloader = get_dataloaders(
    dataset=args.dataset,
    missed_label=model_id // 2,
)


print(f"Using {architecture} for training...")
print(f"Current model id is {model_id}...")
print(f"Using {loss_name} for training...")
# Model
print("==> Building model..")

net = get_model(model_name=architecture, n_classes=10)

print("Used device is ", device)
net = net.to(device)

if args.resume:
    raise ValueError("Resume is not supported!")

criterion = get_loss_function(loss_type=loss_name)
optimizer = optim.sgd.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    # Save checkpoint.
    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if args.dataset == "cifar10":
            save_folder = "checkpoints"
        elif args.dataset == "noisy_cifar10":
            save_folder = "checkpoints_noisy_cifar10"
        elif args.dataset == "missed_class_cifar10":
            save_folder = "checkpoints_missed_class_cifar10"
        elif args.dataset == "svhn":
            save_folder = "checkpoints_svhn"
        else:
            raise ValueError(f"{args.dataset} -- no such dataset")

        if not os.path.exists(f"{save_folder}/{architecture}/{loss_name}/{model_id}"):
            os.makedirs(f"{save_folder}/{architecture}/{loss_name}/{model_id}")
        torch.save(
            state, f"./{save_folder}/{architecture}/{loss_name}/{model_id}/ckpt.pth"
        )
        if args.dataset == "missed_class_cifar10":
            torch.save(
                trainloader.dataset,
                f"./{save_folder}/{architecture}/{loss_name}/{model_id}/dataset.pth",
            )
        best_acc = acc


for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()

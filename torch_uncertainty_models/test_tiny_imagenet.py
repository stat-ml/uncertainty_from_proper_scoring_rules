from pathlib import Path
import os
from torch_uncertainty.models.resnet import resnet
from safetensors.torch import load_file
from torch_uncertainty.datamodules.classification.tiny_imagenet import (
    TinyImageNetDataModule,
)
from torchmetrics import Accuracy


def load_model(version: int):
    """Load the model corresponding to the given version."""
    model = resnet(
        arch=18,
        num_classes=200,
        in_channels=3,
        style="cifar",
        conv_bias=False,
    )
    path = f"/home/nkotelevskii/github/uncertainty_from_proper_scoring_rules/torch_uncertainty_models/models/tiny-imagenet-resnet18-0-1023/version_{version}.safetensors"
    print(path)
    if not os.path.exists(path):
        raise ValueError("File does not exist")

    state_dict = load_file(path)
    model.load_state_dict(state_dict=state_dict)
    return model


acc = Accuracy("multiclass", num_classes=200)
data_module = TinyImageNetDataModule(
    root="/home/nkotelevskii/github/uncertainty_from_proper_scoring_rules/datasets/",
    batch_size=32,
)
model = load_model(100)

model.eval()
data_module.setup("test")

for batch in data_module.test_dataloader()[0]:
    x, y = batch
    y_hat = model(x)
    acc.update(y_hat, y)
print(f"Accuracy on the test set: {acc.compute():.3%}")

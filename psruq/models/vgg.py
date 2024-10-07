"""VGG11/13/16/19 in Pytorch."""

from typing import List

import torch
import torch.nn as nn

VGG_NAME_TO_CONFIGURATION_DICT = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(torch.nn.Module):
    def __init__(self, vgg_name, n_classes):
        super(VGG, self).__init__()
        if n_classes == 10:
            self.classifier = torch.nn.Linear(512, n_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )
        self.features = self._make_layers(
            configuration=VGG_NAME_TO_CONFIGURATION_DICT[vgg_name]
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration: List):
        layers = []
        in_channels = 3
        for x in configuration:
            if x == "M":
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(x),
                    torch.nn.ReLU(inplace=True),
                ]
                in_channels = x
        if hasattr(self.classifier, "out_features"):
            layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
        return torch.nn.Sequential(*layers)

"""DUQ ResNet in Pytorch.

Script is adapted from https://github.com/y0ast/deterministic-uncertainty-quantification .
And based on 
Uncertainty Estimation Using a Single Deep Deterministic Neural Network https://arxiv.org/pdf/2003.02037
"""
import torch
import torch.nn as nn
import torchvision


class DUQNN(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor
 
        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N
        
        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x) -> torch.Tensor:
        z = self.feature_extractor(x)
        y_pred = self.rbf(z)

        return y_pred

def ResNet18DUQ(n_classes):
    resnet18 = torchvision.models.resnet18()

    # Adapted resnet from:
    # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    resnet18.conv1 = torch.nn.Conv2d(
        in_channels=3, 
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    resnet18.maxpool = torch.nn.Identity()
    resnet18.fc = torch.nn.Identity()

    return DUQNN(
        feature_extractor=resnet18,
        num_classes=n_classes,
        ### Parameters are taken from https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/train_duq_cifar.py
        centroid_size=512,
        model_output_size=512,
        length_scale=0.1,
        gamma=0.999,
    )

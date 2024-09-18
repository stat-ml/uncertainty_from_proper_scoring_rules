import torch
import torch.nn as nn
import torch.nn.functional as F


def targets2vector(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    # Ensure targets are one-hot encoded
    if len(targets.shape) == 0 or targets.shape[-1] != n_classes:
        targets_vector = F.one_hot(targets, num_classes=n_classes).float()
    else:
        targets_vector = targets
    return targets_vector


class SphericalScoreLoss(nn.Module):
    def __init__(self):
        super(SphericalScoreLoss, self).__init__()

    def forward(
        self, inputs_: torch.Tensor, targets: torch.Tensor, is_logit: bool = True
    ) -> torch.Tensor:
        """
        Calculate the Spherical Score Loss for multi-class classification

        Parameters:
        - inputs_: Tensor of predicted inputs
        - targets: Tensor of actual labels, not one-hot encoded
        - is_logit: Flag, true if logits provided

        Returns:
        - Spherical Score loss: Tensor
        """
        n_classes = inputs_.size(1)
        targets_vector = targets2vector(targets=targets, n_classes=n_classes)
        if is_logit:
            predictions = F.softmax(inputs_, dim=-1)
        else:
            predictions = inputs_

        normed_predictions = predictions / torch.linalg.norm(
            predictions, dim=-1, keepdim=True
        )

        loss = torch.mean(-torch.sum(normed_predictions * targets_vector, dim=-1))
        return loss


class NegLogScore(nn.Module):
    def __init__(self):
        super(NegLogScore, self).__init__()

    def forward(
        self, inputs_: torch.Tensor, targets: torch.Tensor, is_logit: bool = True
    ) -> torch.Tensor:
        """
        Calculate the NegLogScore Loss for multi-class classification

        Parameters:
        - inputs_: Tensor of predicted inputs
        - targets: Tensor of actual labels, not one-hot encoded
        - is_logit: Flag, true if logits provided

        Returns:
        - NegLogScore loss: Tensor
        """
        n_classes = inputs_.size(1)
        targets_vector = targets2vector(targets=targets, n_classes=n_classes)

        if is_logit:
            predictions = F.softmax(inputs_, dim=-1)
        else:
            predictions = inputs_

        coeff = ((predictions - targets_vector) / (predictions**2)).detach()
        coeff = coeff * targets_vector
        print(coeff.max())
        print(coeff.min())
        clipped_coeff = torch.clip(coeff, min=-1.0, max=1.0)
        print(clipped_coeff.max())
        print(clipped_coeff.min())

        loss = torch.mean(torch.sum(clipped_coeff * predictions, dim=-1))
        return loss


class BrierScoreLoss(nn.Module):
    def __init__(self):
        super(BrierScoreLoss, self).__init__()

    def forward(
        self, inputs_: torch.Tensor, targets: torch.Tensor, is_logit: bool = True
    ) -> torch.Tensor:
        """
        Calculate the BrierScoreLoss for multi-class classification

        Parameters:
        - inputs_: Tensor of predicted inputs
        - targets: Tensor of actual labels, not one-hot encoded
        - is_logit: Flag, true if logits provided

        Returns:
        - BrierScore loss: Tensor
        """
        n_classes = inputs_.size(1)
        targets_vector = targets2vector(targets=targets, n_classes=n_classes)

        if is_logit:
            predictions = F.softmax(inputs_, dim=-1)
        else:
            predictions = inputs_

        loss = torch.mean(torch.sum((predictions - targets_vector) ** 2, dim=-1))
        return loss


def get_loss_function(loss_name: str) -> torch.nn.Module:
    match loss_name:
        case "cross_entropy":
            loss = nn.CrossEntropyLoss()
        case "brier_score":
            loss = BrierScoreLoss()
        case "spherical_score":
            loss = SphericalScoreLoss()
        case "neglog_score":
            loss = NegLogScore()
        case _:
            print("No such loss")
            raise ValueError
    return loss

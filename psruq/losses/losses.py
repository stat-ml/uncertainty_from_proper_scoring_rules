import psruq.losses.constants
import psruq.utils
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        targets_vector = psruq.utils.targets2vector(
            targets=targets, n_classes=n_classes
        )
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
        targets_vector = psruq.utils.targets2vector(
            targets=targets, n_classes=n_classes
        )

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
        targets_vector = psruq.utils.targets2vector(
            targets=targets, n_classes=n_classes
        )

        if is_logit:
            predictions = F.softmax(inputs_, dim=-1)
        else:
            predictions = inputs_

        loss = torch.mean(torch.sum((predictions - targets_vector) ** 2, dim=-1))
        return loss


def get_loss_function(loss_type: str) -> torch.nn.Module:
    match psruq.losses.constants.LossName(loss_type):
        case psruq.losses.constants.LossName.CROSS_ENTROPY:
            loss = nn.CrossEntropyLoss()
        case psruq.losses.constants.LossName.BRIER_SCORE:
            loss = BrierScoreLoss()
        case psruq.losses.constants.LossName.SPHERICAL_SCORE:
            loss = SphericalScoreLoss()
        case psruq.losses.constants.LossName.NEG_LOG_SCORE:
            loss = NegLogScore()
        case _:
            raise ValueError(
                f"{loss_type} --  no such loss type available. ",
                f"Available options are: {[element.value for element in psruq.losses.constants.LossName]}",
            )
    return loss

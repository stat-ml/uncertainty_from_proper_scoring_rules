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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Spherical Score Loss for multi-class classification
        
        Parameters:
        - logits: Tensor of predicted logits for each class (as returned before a softmax function)
        - targets: Tensor of actual labels, not one-hot encoded
        
        Returns:
        - Spherical Score loss: Tensor
        """
        n_classes = logits.size(1)
        targets_vector = targets2vector(targets=targets, n_classes=n_classes)
        predictions = F.softmax(logits, dim=-1)

        normed_targets = targets_vector / torch.linalg.norm(targets)
        normed_predictions = predictions / torch.linalg.norm(predictions)

        loss = -torch.linalg.norm(targets) * torch.dot(normed_predictions,
                                                       normed_targets)
        return loss


class NegLogScore(nn.Module):
    def __init__(self):
        super(NegLogScore, self).__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Negative Logarithm Loss for multi-class classification
        
        Parameters:
        - logits: Tensor of predicted logits for each class (as returned before a softmax function)
        - targets: Tensor of actual labels, not one-hot encoded
        
        Returns:
        - Negative logarithm score loss: Tensor
        """
        n_classes = logits.size(1)
        targets_vector = targets2vector(targets=targets, n_classes=n_classes)
        predictions = F.softmax(logits, dim=-1)

        loss = torch.sum(
                torch.log(predictions) - 1 + targets_vector / predictions,
                dim=-1) 
        return loss




class BrierScoreLoss(nn.Module):
    def __init__(self):
        super(BrierScoreLoss, self).__init__()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Brier Score Loss for multi-class classification
        
        Parameters:
        - logits: Tensor of predicted logits for each class (as returned before a softmax function)
        - targets: Tensor of actual labels, not one-hot encoded
        
        Returns:
        - Brier score loss: Tensor
        """
        n_classes = logits.size(1)
        targets_vector = targets2vector(targets=targets, n_classes=n_classes)
        predictions = F.softmax(logits, dim=-1)
        loss = torch.mean(
                torch.sum((predictions - targets_vector) ** 2, dim=-1)
                )
        return loss


def get_loss_function(loss_name: str) -> torch.nn.Module:
    match loss_name:
        case 'cross_entropy':
            loss = nn.CrossEntropyLoss()
        case 'brier_score':
            loss = BrierScoreLoss() 
        case 'spherical_score':
            loss = SphericalScoreLoss()
        case _:
            print("No such loss")
            raise ValueError
    return loss



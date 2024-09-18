import torch
import torch.nn.functional as F


def targets2vector(targets: torch.Tensor, n_classes: int) -> torch.Tensor:
    # Ensure targets are one-hot encoded
    if len(targets.shape) == 0 or targets.shape[-1] != n_classes:
        targets_vector = F.one_hot(targets, num_classes=n_classes).float()
    else:
        targets_vector = targets
    return targets_vector

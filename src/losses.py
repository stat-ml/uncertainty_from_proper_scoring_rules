import torch
import torch.nn as nn

import torch.nn.functional as F

class SphericalScoreLoss(nn.Module):
    def __init__(self):
        super(SphericalScoreLoss, self).__init__()
 

class BrierScoreLoss(nn.Module):
    def __init__(self):
        super(BrierScoreLoss, self).__init__()
    
    def forward(self, logits, targets):
        """
        Calculate the Brier Score Loss for multi-class classification
        
        Parameters:
        - logits: Tensor of predicted logits for each class (as returned before a softmax function)
        - targets: Tensor of actual labels, not one-hot encoded
        
        Returns:
        - Brier score loss: Tensor
        """
        # Ensure targets are one-hot encoded
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float() 
        predictions = F.softmax(logits, dim=-1)
        loss = torch.mean((predictions - targets_one_hot) ** 2)
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



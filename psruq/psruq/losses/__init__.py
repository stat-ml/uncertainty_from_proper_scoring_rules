from .constants import LossName
from .losses import get_loss_function

__all__ = [
    "LossName",
    "get_loss_function",
]
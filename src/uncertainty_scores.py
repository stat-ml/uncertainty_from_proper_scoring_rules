import numpy as np


def log_score_divergence(
        true_vec: np.ndarray,
        pred_vec: np.ndarray,
        ) -> np.ndarray:
    """
    Effectively it is KL divergence
    """
    

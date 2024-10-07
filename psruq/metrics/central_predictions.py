import numpy as np
from psruq.metrics.utils import posterior_predictive, safe_softmax


def log_score_central_prediction(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    mean_logit = np.mean(logits, axis=0, keepdims=True) / T
    central_pred = safe_softmax(mean_logit)
    return central_pred


def brier_score_central_prediction(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    return posterior_predictive(logits, T)


def zero_one_central_prediction(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    tilde_p = np.mean(
        logits == logits.max(axis=-1, keepdims=True), axis=0, keepdims=True
    )
    central_pred = (tilde_p != 0.0) * np.ones_like(tilde_p)
    central_pred = central_pred / np.sum(central_pred, axis=-1, keepdims=True)
    n_classes = logits.shape[-1]
    central_pred = np.ones_like(logits[0])[None] / n_classes
    # return central_pred
    return posterior_predictive(logits, T)


def spherical_score_central_prediction(logits: np.ndarray, T: float = 1.0):
    probs = safe_softmax(logits / T)
    K = logits.shape[-1]

    norms = np.linalg.norm(probs, axis=-1, keepdims=True, ord=2)
    x = np.mean(probs / norms, axis=0, keepdims=True)

    x0 = np.ones(K).reshape(1, 1, K) / K
    x0_norm = np.linalg.norm(x0, ord=2, keepdims=True, axis=-1)

    y_orthogonal = x - np.sum(x * x0, axis=-1, keepdims=True) * (x0 / x0_norm**2)
    y_orthogonal_norm = np.linalg.norm(y_orthogonal, ord=2, keepdims=True, axis=-1)

    central_pred = x0 + (y_orthogonal / np.sqrt(1 - y_orthogonal_norm**2)) * x0_norm
    return central_pred

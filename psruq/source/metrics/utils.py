import numpy as np


def safe_softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def posterior_predictive(logits_: np.ndarray, T: float = 1.0) -> np.ndarray:
    prob_p = safe_softmax(logits_ / T)
    ppd = np.mean(prob_p, axis=0, keepdims=True)
    return ppd

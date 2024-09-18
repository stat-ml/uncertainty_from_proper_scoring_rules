import numpy as np


# Log score
def logscore_g(prob: np.ndarray) -> np.ndarray:
    return np.sum(prob * np.log(prob), axis=-1, keepdims=True)


def logscore_grad_g(prob: np.ndarray) -> np.ndarray:
    return 1 + np.log(prob)


# Brier score
def brier_g(prob: np.ndarray) -> np.ndarray:
    return -np.sum(prob * (1 - prob), axis=-1, keepdims=True)


def brier_grad_g(prob: np.ndarray) -> np.ndarray:
    return 2 * prob - 1


# Zero-one score
def zero_one_g(prob: np.ndarray) -> np.ndarray:
    return np.max(prob, axis=-1, keepdims=True) - 1


def zero_one_grad_g(prob: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(prob)
    N, M, K, C = prob.shape

    if np.all(prob == 1 / C):
        max_indices = np.random.randint(low=0, high=C, size=(N, M, K))
    else:
        max_indices = np.argmax(prob, axis=-1)

    mask[np.arange(N)[:, None], np.arange(M)[:, None], np.arange(K), max_indices] = 1
    return mask


# Spherical
def spherical_g(prob: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prob, axis=-1, keepdims=True) - 1


def spherical_grad_g(prob: np.ndarray) -> np.ndarray:
    return prob / np.linalg.norm(prob, axis=-1, keepdims=True)

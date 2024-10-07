from typing import Optional

import numpy as np
from psruq.metrics.constants import ApproximationType, GName, RiskType
from psruq.metrics.getters import (
    get_central_prediction,
    get_g_functions,
    get_probability_approximation,
    get_specific_risk,
)
from psruq.metrics.utils import posterior_predictive, safe_softmax
from scipy.special import logsumexp


def energy(logits: np.ndarray, T: float) -> np.ndarray:
    return -T * logsumexp(logits / T, axis=-1)


def get_energy_inner(logits: np.ndarray, T: float) -> np.ndarray:
    return np.squeeze(energy(np.mean(logits, keepdims=True, axis=0), T=T))


def get_energy_outer(logits: np.ndarray, T: float) -> np.ndarray:
    return np.squeeze(np.mean(energy(logits, T=T), axis=0, keepdims=True))


def get_risk_approximation(
    g_name: GName,
    risk_type: RiskType,
    logits: np.ndarray,
    gt_approx: ApproximationType,
    T: float = 1.0,
    probabilities: Optional[np.ndarray] = None,
    pred_approx: Optional[ApproximationType] = None,
) -> np.ndarray:
    if probabilities is None:
        probabilities = safe_softmax(logits)

    risk = get_specific_risk(g_name=g_name, risk_type=risk_type)
    prob_gt = get_probability_approximation(
        g_name=g_name, approximation=gt_approx, logits=logits, T=T
    )
    prob_pred = get_probability_approximation(
        g_name=g_name, approximation=pred_approx, logits=logits, T=T
    )
    if risk_type.value == RiskType.BAYES_RISK.value:
        result = np.mean(risk(prob_gt=prob_gt), axis=0)
    else:
        result = np.mean(risk(prob_gt=prob_gt, prob_pred=prob_pred), axis=(0, 1))

    return np.squeeze(result)


def check_scalar_product(
    g_name: GName,
    logits: np.ndarray,
    T: float = 1.0,
    probabilities: Optional[np.ndarray] = None,
) -> np.ndarray:
    if probabilities is None:
        probabilities = safe_softmax(logits)
    _, g_grad_func = get_g_functions(g_name=g_name)
    bma_probs = posterior_predictive(logits, T=T)
    central_probs = get_central_prediction(g_name=g_name)(logits=logits, T=T)

    if g_name.value == GName.ZERO_ONE_SCORE.value:
        probabilities = probabilities[None]
        central_probs = central_probs[None]
    grad_pred = g_grad_func(probabilities)
    grad_central = g_grad_func(central_probs)
    if g_name.value == GName.ZERO_ONE_SCORE.value:
        probabilities = probabilities[0]
        central_probs = central_probs[0]

    vec_1 = np.mean(grad_pred, axis=0, keepdims=True) - grad_central
    vec_2 = central_probs - bma_probs
    res = np.sum(vec_1 * vec_2, axis=-1)
    return res

import numpy as np
from source.metrics.risk_primitives import total_risk, excess_risk, bayes_risk
from functools import partial
from source.metrics.g_primitives import (
    logscore_g,
    logscore_grad_g,
    brier_g,
    brier_grad_g,
    zero_one_g,
    zero_one_grad_g,
    spherical_g,
    spherical_grad_g,
)

from source.metrics.central_predictions import (
    log_score_central_prediction,
    brier_score_central_prediction,
    zero_one_central_prediction,
    spherical_score_central_prediction,
)
from source.metrics.constants import GName, RiskType, ApproximationType
from source.metrics.utils import safe_softmax, posterior_predictive


def get_risk_function(risk_type: RiskType) -> callable:
    match risk_type:
        case RiskType.TOTAL_RISK:
            risk = total_risk
        case RiskType.BAYES_RISK:
            risk = bayes_risk
        case RiskType.EXCESS_RISK:
            risk = excess_risk
        case _:
            raise ValueError(
                f"{risk_type} --  no such risk type available. ",
                f"Available options are: {[element.value for element in RiskType]}",
            )
    return risk


def get_g_functions(g_name: str) -> tuple[callable, callable]:
    match g_name:
        case GName.LOG_SCORE:
            g_func = logscore_g
            g_grad_func = logscore_grad_g
        case GName.BRIER_SCORE:
            g_func = brier_g
            g_grad_func = brier_grad_g
        case GName.SPHERICAL_SCORE:
            g_func = spherical_g
            g_grad_func = spherical_grad_g
        case GName.ZERO_ONE_SCORE:
            g_func = zero_one_g
            g_grad_func = zero_one_grad_g
        case _:
            raise ValueError(
                f"{g_name} --  no such G-function available. ",
                f"Available options are: {[element.value for element in GName]}",
            )
    return g_func, g_grad_func


def get_specific_risk(
    g_name: GName,
    risk_type: RiskType,
) -> callable:
    g_func, g_grad_func = get_g_functions(g_name=g_name)
    risk = get_risk_function(risk_type=risk_type)

    match risk_type:
        case RiskType.BAYES_RISK:
            specific_risk = partial(risk, g=g_func)
        case RiskType.TOTAL_RISK | RiskType.EXCESS_RISK:
            specific_risk = partial(risk, g=g_func, grad_g=g_grad_func)
        case _:
            raise ValueError(
                f"{risk_type} --  no such risk type available. ",
                f"Available options are: {[element.value for element in RiskType]}",
            )
    return specific_risk


def get_central_prediction(
    g_name: GName,
) -> callable:
    match g_name:
        case GName.LOG_SCORE:
            central_pred = log_score_central_prediction
        case GName.BRIER_SCORE:
            central_pred = brier_score_central_prediction
        case GName.SPHERICAL_SCORE:
            central_pred = spherical_score_central_prediction
        case GName.ZERO_ONE_SCORE:
            central_pred = zero_one_central_prediction
        case _:
            raise ValueError(
                f"{g_name} --  no such G-function available. ",
                f"Available options are: {[element.value for element in GName]}",
            )
    return central_pred


def get_probability_approximation(
    g_name: GName,
    approximation: ApproximationType,
    logits: np.ndarray,
    T: float = 1.0,
) -> np.ndarray:

    match approximation:
        case ApproximationType.OUTER:
            resulting_probs = safe_softmax(x=logits)
        case ApproximationType.INNER:
            resulting_probs = posterior_predictive(logits, T=T)
        case ApproximationType.CENTRAL:
            resulting_probs = get_central_prediction(g_name=g_name)(logits=logits, T=T)
        case _:
            raise ValueError(
                f"{approximation} --  no such approximation available. ",
                f"Available options are: {[element.value for element in ApproximationType]}",
            )

    return resulting_probs

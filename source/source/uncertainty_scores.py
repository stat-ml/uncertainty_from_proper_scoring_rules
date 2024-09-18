import numpy as np
from scipy.special import logsumexp


def compute_bias(
    logits_gt,
    logits_pred,
    central_prediction,
    func_,
):
    ppd = posterior_predictive(logits_=logits_gt)
    logits_ppd = np.log(ppd)
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)

    central_pred = central_prediction(logits_pred)
    logits_central_pred = np.log(central_pred)
    logits_central_pred = np.repeat(logits_central_pred, logits_pred.shape[0], axis=0)

    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_central_pred, func_=func_
    )
    return res


def compute_model_variance(
    logits_gt,
    logits_pred,
    central_prediction,
    func_,
):
    central_pred = central_prediction(logits_gt)
    logits_central_pred = np.log(central_pred)
    logits_central_pred = np.repeat(logits_central_pred, logits_pred.shape[0], axis=0)

    res = make_pairwise_calculation(
        logits_gt=logits_central_pred, logits_pred=logits_pred, func_=func_
    )
    return res


def make_pairwise_calculation(
    logits_gt: np.ndarray,
    logits_pred: np.ndarray,
    func_: callable,
):
    all_pairs = []
    for l_gt in logits_gt:
        for l_pred in logits_pred:
            specific_pair = func_(l_gt, l_pred)
            all_pairs.append(specific_pair)
    all_pairs = np.vstack(all_pairs)
    return np.mean(all_pairs, axis=0)


def safe_softmax(x):
    """Softmax

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def logscore_distance(left_logits, right_logits):
    """KL divergence

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    p_safe = safe_softmax(left_logits)
    res = np.sum(
        p_safe
        * (
            left_logits
            - right_logits
            + logsumexp(right_logits, axis=-1, keepdims=True)
            - logsumexp(left_logits, axis=-1, keepdims=True)
        ),
        axis=-1,
    )
    return res


def maxprob_diff(logits_gt, logits_pred):
    prob_gt = safe_softmax(logits_gt)
    prob_pred = safe_softmax(logits_pred)

    if len(prob_gt.shape) > 2:
        prob_gt = prob_gt[0]
    if len(prob_pred.shape) > 2:
        prob_pred = prob_pred[0]

    diffs = []
    for gt_prob, est_prob in zip(prob_gt, prob_pred):
        diffs.append(np.max(gt_prob) - gt_prob[np.argmax(est_prob)])
    diffs = np.hstack(diffs)
    return diffs


def brier_distance(logits_gt, logits_pred):
    prob_gt = safe_softmax(logits_gt)
    prob_pred = safe_softmax(logits_pred)

    res = np.sum((prob_gt - prob_pred) ** 2, axis=-1)

    return res


def neglog_distance(logits_gt, logits_pred):
    p_exp = safe_softmax(logits_gt)
    q_exp = safe_softmax(logits_pred)

    is_dist = np.sum(
        p_exp / q_exp
        - logits_gt
        + logits_pred
        + logsumexp(logits_gt, axis=-1, keepdims=True)
        - logsumexp(logits_pred, axis=-1, keepdims=True)
        - 1,
        axis=-1,
    )

    return is_dist


def spherical_distance(logits_gt, logits_pred):
    p_prob = safe_softmax(logits_gt)
    q_prob = safe_softmax(logits_pred)

    p_norm = np.linalg.norm(p_prob, ord=2, axis=-1, keepdims=True)
    q_norm = np.linalg.norm(q_prob, ord=2, axis=-1, keepdims=True)

    pairwise_dot = np.sum((p_prob / p_norm) * (q_prob / q_norm), axis=-1)

    spherical_divs = p_norm[..., 0] * (1 - pairwise_dot)

    return spherical_divs


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=-1)


def posterior_predictive(logits_):
    prob_p = safe_softmax(logits_)
    ppd = np.mean(prob_p, axis=0, keepdims=True)
    return ppd


############################################################


def central_prediction_neglog(
    logits_: np.ndarray,
):
    eps = 1e-7
    probs = safe_softmax(logits_)
    mean_inverse_prob = np.mean(1 / (probs + eps), axis=0)[None]
    return 1 / mean_inverse_prob


def bias_neglog(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_bias(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        func_=neglog_distance,
        central_prediction=central_prediction_neglog,
    )
    return res


def mv_neglog(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_model_variance(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        func_=neglog_distance,
        central_prediction=central_prediction_neglog,
    )
    return res


def mv_bi_neglog(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return mv_neglog(logits_pred=logits_pred, logits_gt=logits_gt) + bi_neglog(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def bias_bi_neglog(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return bias_neglog(logits_pred=logits_pred, logits_gt=logits_gt) + bi_neglog(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def total_neglog_outer(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_neglog_outer_outer(
        logits_gt=logits_gt, logits_pred=logits_pred
    ) + bayes_neglog_outer(logits_gt=logits_gt, logits_pred=logits_pred)


def total_neglog_inner(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return bayes_neglog_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def bayes_neglog_outer(logits_gt, logits_pred=None):
    return np.mean(
        np.sum(logits_gt - logsumexp(logits_gt, axis=-1, keepdims=True), axis=-1),
        axis=0,
    )


def bayes_neglog_inner(logits_gt, logits_pred=None):
    ppd = posterior_predictive(logits_=logits_gt)[0]
    return np.sum(np.log(ppd), axis=-1)


def excess_neglog_outer_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_pred, func_=neglog_distance
    )
    return res


def excess_neglog_outer_inner(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_pred))
    logits_ppd = np.repeat(logits_ppd, logits_gt.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_ppd, func_=neglog_distance
    )
    return res


def excess_neglog_inner_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_gt))
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_pred, func_=neglog_distance
    )
    return res


def bi_neglog(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_neglog_outer_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def rbi_neglog(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_neglog_inner_outer(logits_gt=logits_gt, logits_pred=logits_pred)


############################################################


def central_prediction_spherical(
    logits_: np.ndarray,
):
    probs = safe_softmax(logits_)
    norms = np.linalg.norm(probs, axis=-1, keepdims=True, ord=2)
    avg_normed = np.mean(probs / norms, axis=0)
    central_pred = avg_normed / np.sum(avg_normed, axis=-1, keepdims=True)
    return central_pred[None]


def bias_spherical(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_bias(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        func_=spherical_distance,
        central_prediction=central_prediction_spherical,
    )
    return res


def mv_spherical(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_model_variance(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        func_=spherical_distance,
        central_prediction=central_prediction_spherical,
    )
    return res


def mv_bi_spherical(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return mv_spherical(logits_pred=logits_pred, logits_gt=logits_gt) + bi_spherical(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def bias_bi_spherical(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return bias_spherical(logits_pred=logits_pred, logits_gt=logits_gt) + bi_spherical(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def total_spherical_outer(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_spherical_outer_outer(
        logits_gt=logits_gt, logits_pred=logits_pred
    ) + bayes_spherical_outer(logits_gt=logits_gt, logits_pred=logits_pred)


def total_spherical_inner(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return bayes_spherical_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def bayes_spherical_outer(logits_gt, logits_pred=None):
    prob_p = safe_softmax(logits_gt)
    return 1 - np.mean(np.linalg.norm(prob_p, axis=-1, ord=2), axis=0)


def bayes_spherical_inner(logits_gt, logits_pred=None):
    ppd = posterior_predictive(logits_=logits_gt)[0]
    return 1 - np.linalg.norm(ppd, axis=-1, ord=2)


def excess_spherical_outer_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_pred, func_=spherical_distance
    )
    return res


def excess_spherical_outer_inner(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_pred))
    logits_ppd = np.repeat(logits_ppd, logits_gt.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_ppd, func_=spherical_distance
    )
    return res


def excess_spherical_inner_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_gt))
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_pred, func_=spherical_distance
    )
    return res


def bi_spherical(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_spherical_outer_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def rbi_spherical(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_spherical_inner_outer(logits_gt=logits_gt, logits_pred=logits_pred)


############################################################


def central_prediction_maxprob(
    logits_: np.ndarray,
):
    n_classes = logits_.shape[-1]
    res = np.ones_like(logits_[0])[None] / n_classes
    return res


def bias_maxprob(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_bias(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        func_=maxprob_diff,
        central_prediction=central_prediction_maxprob,
    )
    return res


def mv_maxprob(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_model_variance(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        func_=maxprob_diff,
        central_prediction=central_prediction_maxprob,
    )
    return res


def mv_bi_maxprob(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return mv_maxprob(logits_pred=logits_pred, logits_gt=logits_gt) + bi_maxprob(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def bias_bi_maxprob(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return bias_maxprob(logits_pred=logits_pred, logits_gt=logits_gt) + bi_maxprob(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def total_maxprob_outer(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_maxprob_outer_outer(
        logits_gt=logits_gt, logits_pred=logits_pred
    ) + bayes_maxprob_outer(logits_gt=logits_gt, logits_pred=logits_pred)


def total_maxprob_inner(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return bayes_maxprob_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def bayes_maxprob_outer(logits_gt, logits_pred=None):
    prob_p = safe_softmax(logits_gt)
    return 1 - np.mean(np.max(prob_p, axis=-1), axis=0)


def bayes_maxprob_inner(logits_gt, logits_pred=None):
    ppd = posterior_predictive(logits_=logits_gt)[0]
    return 1 - np.max(ppd, axis=-1)


def excess_maxprob_outer_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_pred, func_=maxprob_diff
    )
    return res


def excess_maxprob_inner_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_gt))
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_pred, func_=maxprob_diff
    )
    return res


def excess_maxprob_outer_inner(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_pred))
    logits_ppd = np.repeat(logits_ppd, logits_gt.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_ppd, func_=maxprob_diff
    )
    return res


def bi_maxprob(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_maxprob_outer_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def rbi_maxprob(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_maxprob_inner_outer(logits_gt=logits_gt, logits_pred=logits_pred)


############################################################


def central_prediction_brier(
    logits_: np.ndarray,
):
    return posterior_predictive(logits_=logits_)


def bias_brier(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_bias(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        central_prediction=central_prediction_brier,
        func_=brier_distance,
    )
    return res


def mv_brier(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_model_variance(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        central_prediction=central_prediction_brier,
        func_=brier_distance,
    )
    return res


def mv_bi_brier(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return mv_brier(logits_pred=logits_pred, logits_gt=logits_gt) + bi_brier(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def bias_bi_brier(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return bias_brier(logits_pred=logits_pred, logits_gt=logits_gt) + bi_brier(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def total_brier_outer(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_brier_outer_outer(
        logits_gt=logits_gt, logits_pred=logits_pred
    ) + bayes_brier_outer(logits_gt=logits_gt, logits_pred=logits_pred)


def total_brier_inner(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return bayes_brier_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def bayes_brier_outer(logits_gt, logits_pred=None):
    probs = safe_softmax(logits_gt)
    return 1 - np.mean(np.sum(probs**2, axis=-1), axis=0)


def bayes_brier_inner(logits_gt, logits_pred=None):
    ppd = posterior_predictive(logits_=logits_gt)[0]
    return 1 - np.sum(ppd**2, axis=-1)


def excess_brier_outer_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_pred, func_=brier_distance
    )
    return res


def excess_brier_inner_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_gt))
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_pred, func_=brier_distance
    )
    return res


def excess_brier_outer_inner(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_pred))
    logits_ppd = np.repeat(logits_ppd, logits_gt.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_ppd, func_=brier_distance
    )
    return res


def bi_brier(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_brier_outer_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def rbi_brier(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_brier_inner_outer(logits_gt=logits_gt, logits_pred=logits_pred)


############################################################


def central_prediction_logscore(
    logits_: np.ndarray,
):
    probs = safe_softmax(logits_)
    res = safe_softmax(np.mean(np.log(probs), axis=0, keepdims=True))
    return res


def bias_logscore(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_bias(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        central_prediction=central_prediction_logscore,
        func_=logscore_distance,
    )
    return res


def mv_logscore(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    res = compute_model_variance(
        logits_gt=logits_gt,
        logits_pred=logits_pred,
        central_prediction=central_prediction_logscore,
        func_=logscore_distance,
    )
    return res


def mv_bi_logscore(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return mv_logscore(logits_pred=logits_pred, logits_gt=logits_gt) + bi_logscore(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def bias_bi_logscore(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return bias_logscore(logits_pred=logits_pred, logits_gt=logits_gt) + bi_logscore(
        logits_pred=logits_pred, logits_gt=logits_gt
    )


def total_logscore_outer(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_logscore_outer_outer(
        logits_gt=logits_gt, logits_pred=logits_pred
    ) + bayes_logscore_outer(logits_gt=logits_gt, logits_pred=logits_pred)


def total_logscore_inner(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    """
    Expected Pairwise Cross Entropy
    """
    return bayes_logscore_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def excess_logscore_outer_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_pred, func_=logscore_distance
    )
    return res


def excess_logscore_inner_outer(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_gt))
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_pred, func_=logscore_distance
    )
    return res


def excess_logscore_inner_inner(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_gt))
    logits_ppd = np.repeat(logits_ppd, logits_pred.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_ppd, logits_pred=logits_ppd, func_=logscore_distance
    )
    return res


def excess_logscore_outer_inner(
    logits_gt: np.ndarray, logits_pred: np.ndarray
) -> np.ndarray:
    logits_ppd = np.log(posterior_predictive(logits_pred))
    logits_ppd = np.repeat(logits_ppd, logits_gt.shape[0], axis=0)
    res = make_pairwise_calculation(
        logits_gt=logits_gt, logits_pred=logits_ppd, func_=logscore_distance
    )
    return res


def bayes_logscore_outer(logits_gt, logits_pred=None):
    prob_p = safe_softmax(logits_gt)
    return np.mean(entropy(prob_p), axis=0)


def bayes_logscore_inner(logits_gt, logits_pred=None):
    ppd = posterior_predictive(logits_=logits_gt)[0]
    return entropy(ppd)


def bi_logscore(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_logscore_outer_inner(logits_gt=logits_gt, logits_pred=logits_pred)


def rbi_logscore(logits_gt: np.ndarray, logits_pred: np.ndarray) -> np.ndarray:
    return excess_logscore_inner_outer(logits_gt=logits_gt, logits_pred=logits_pred)


if __name__ == "__main__":
    # Example usage
    N_members, N_objects, N_classes = 5, 100, 1000  # Example dimensions
    A = np.random.randn(N_members, N_objects, N_classes)
    B = A
    # B = np.random.randn(N_members, N_objects, N_classes)

    print("Testing that Bayes outer is less than Bayes inner")
    for score, b_i, b_o in [
        ("Logscore", bayes_logscore_inner, bayes_logscore_outer),
        ("Brier", bayes_brier_inner, bayes_brier_outer),
        ("Spherical", bayes_spherical_inner, bayes_spherical_outer),
        ("Neglog", bayes_neglog_inner, bayes_neglog_outer),
        ("MaxProb", bayes_maxprob_inner, bayes_maxprob_outer),
    ]:
        try:
            assert np.all(b_o(A, B) <= b_i(A, B))
            print(f"{score} OK")
        except Exception as ex:
            print(f"{score} FAILED {ex}")
    print("Success!")
    print("*" * 30)
    print("*" * 30)

    print(
        ("Testing that inner and outer approximations" " lead to the same Total outer")
    )
    for score, b_i, b_o, e_io, e_oo in [
        (
            "Logscore",
            bayes_logscore_inner,
            bayes_logscore_outer,
            excess_logscore_inner_outer,
            excess_logscore_outer_outer,
        ),
        (
            "Brier",
            bayes_brier_inner,
            bayes_brier_outer,
            excess_brier_inner_outer,
            excess_brier_outer_outer,
        ),
        (
            "Spherical",
            bayes_spherical_inner,
            bayes_spherical_outer,
            excess_spherical_inner_outer,
            excess_spherical_outer_outer,
        ),
        (
            "Neglog",
            bayes_neglog_inner,
            bayes_neglog_outer,
            excess_neglog_inner_outer,
            excess_neglog_outer_outer,
        ),
        (
            "Maxprob",
            bayes_maxprob_inner,
            bayes_maxprob_outer,
            excess_maxprob_inner_outer,
            excess_maxprob_outer_outer,
        ),
    ]:
        try:
            tot_i = b_i(A, B) + e_io(A, B)
            tot_o = b_o(A, B) + e_oo(A, B)
            assert np.all(np.isclose(tot_i, tot_o))
            print(f"{score} OK")
        except Exception as ex:
            print(f"{score} FAILED {ex}")

    print("*" * 30)
    print("*" * 30)

    ################################################
    print(
        ("Testing that inner and outer approximations " "lead to the same Total inner")
    )
    for score, b_i, b_o, e_ii, e_oi in [
        (
            "Logscore",
            bayes_logscore_inner,
            bayes_logscore_outer,
            0,
            excess_logscore_outer_inner,
        ),
        ("Brier", bayes_brier_inner, bayes_brier_outer, 0, excess_brier_outer_inner),
        (
            "Spherical",
            bayes_spherical_inner,
            bayes_spherical_outer,
            0,
            excess_spherical_outer_inner,
        ),
        (
            "Neglog",
            bayes_neglog_inner,
            bayes_neglog_outer,
            0,
            excess_neglog_outer_inner,
        ),
        (
            "Maxprob",
            bayes_maxprob_inner,
            bayes_maxprob_outer,
            0,
            excess_maxprob_outer_inner,
        ),
    ]:
        try:
            tot_i = b_i(A, B) + e_ii
            tot_o = b_o(A, B) + e_oi(A, B)
            assert np.all(np.isclose(tot_i, tot_o))
            print(f"{score} OK")
        except Exception as ex:
            print(f"{score} FAILED {ex}")
    ################################################

    print("*" * 30)
    print("*" * 30)

    print(
        (
            "Testing that Difference of Excess risks is equal "
            "to the differece of Bayes risks (Bregman Information)"
        )
    )

    for score, b_i, b_o, e_io, e_oo, e_oi, bi in [
        (
            "Logscore",
            bayes_logscore_inner,
            bayes_logscore_outer,
            excess_logscore_inner_outer,
            excess_logscore_outer_outer,
            excess_logscore_outer_inner,
            bi_logscore,
        ),
        (
            "Brier",
            bayes_brier_inner,
            bayes_brier_outer,
            excess_brier_inner_outer,
            excess_brier_outer_outer,
            excess_brier_outer_inner,
            bi_brier,
        ),
        (
            "Spherical",
            bayes_spherical_inner,
            bayes_spherical_outer,
            excess_spherical_inner_outer,
            excess_spherical_outer_outer,
            excess_spherical_outer_inner,
            bi_spherical,
        ),
        (
            "Neglog",
            bayes_neglog_inner,
            bayes_neglog_outer,
            excess_neglog_inner_outer,
            excess_neglog_outer_outer,
            excess_neglog_outer_inner,
            bi_neglog,
        ),
        (
            "Maxprob",
            bayes_maxprob_inner,
            bayes_maxprob_outer,
            excess_maxprob_inner_outer,
            excess_maxprob_outer_outer,
            excess_maxprob_outer_inner,
            bi_maxprob,
        ),
    ]:
        try:
            excess_diff = e_oo(A, B) - e_io(A, B)
            bayes_diff = b_i(A, B) - b_o(A, B)
            excess_oi = e_oi(A, B)
            bregman = bi(A, B)
            assert np.all(np.isclose(excess_diff, bayes_diff))
            assert np.all(np.isclose(excess_diff, excess_oi))
            assert np.all(np.isclose(excess_diff, bregman))
            print(f"{score} OK")
        except Exception as ex:
            print(f"{score} FAILED {ex}")
            print(f"For Bayes diff ok?: {np.all(np.isclose(excess_diff, bayes_diff))}")
            print(f"For excess oi ok?: {np.all(np.isclose(excess_diff, excess_oi))}")
            print(f"For Bregman ok?: {np.all(np.isclose(excess_diff, bregman))}")

    ################################################

    print("*" * 30)
    print("*" * 30)

    print(
        (
            "Testing that Difference of Excess risks is equal "
            "to the differece of Bayes risks (Reverse Bregman Information)"
        )
    )

    for score, t_i, t_o, e_io, e_oo, e_oi, rbi, bias, mv in [
        (
            "Logscore",
            total_logscore_inner,
            total_logscore_outer,
            excess_logscore_inner_outer,
            excess_logscore_outer_outer,
            excess_logscore_outer_inner,
            rbi_logscore,
            bias_logscore,
            mv_logscore,
        ),
        (
            "Brier",
            total_brier_inner,
            total_brier_outer,
            excess_brier_inner_outer,
            excess_brier_outer_outer,
            excess_brier_outer_inner,
            rbi_brier,
            bias_brier,
            mv_brier,
        ),
        (
            "Spherical",
            total_spherical_inner,
            total_spherical_outer,
            excess_spherical_inner_outer,
            excess_spherical_outer_outer,
            excess_spherical_outer_inner,
            rbi_spherical,
            bias_spherical,
            mv_spherical,
        ),
        (
            "Neglog",
            total_neglog_inner,
            total_neglog_outer,
            excess_neglog_inner_outer,
            excess_neglog_outer_outer,
            excess_neglog_outer_inner,
            rbi_neglog,
            bias_neglog,
            mv_neglog,
        ),
        (
            "Maxprob",
            total_maxprob_inner,
            total_maxprob_outer,
            excess_maxprob_inner_outer,
            excess_maxprob_outer_outer,
            excess_maxprob_outer_inner,
            rbi_maxprob,
            bias_maxprob,
            mv_maxprob,
        ),
    ]:
        try:
            excess_diff = e_oo(A, B) - e_oi(A, B)
            total_diff = t_o(A, B) - t_i(A, B)
            excess_io = e_io(A, B)
            reverse_bregman = rbi(A, B)
            excess_oo = e_io(A, B) + e_oi(A, B)
            decomposition_excess_io = bias(A, B) + mv(A, B)
            assert np.all(np.isclose(excess_diff, total_diff))
            assert np.all(np.isclose(excess_diff, excess_io))
            assert np.all(np.isclose(excess_diff, reverse_bregman))
            assert np.all(np.isclose(excess_oo, e_oo(A, B)))
            assert np.all(np.isclose(decomposition_excess_io, excess_io))
            print(f"{score} OK")
        except Exception as ex:
            print(f"{score} FAILED {ex}")
            print(f"For Total diff ok?: {np.all(np.isclose(excess_diff, total_diff))}")
            print(f"For excess oi ok?: {np.all(np.isclose(excess_diff, excess_io))}")
            print(
                f"For Reverse Bregman ok?: {np.all(np.isclose(excess_diff, reverse_bregman))}"
            )
            print(f"For excess oo ok?: {np.all(np.isclose(excess_oo, e_oo(A, B)))}")
            print(
                f"For excess io decomp ok?: {np.all(np.isclose(decomposition_excess_io, excess_io))}"
            )
            print(
                f"Maxdiff in decomposition is: {np.max(np.abs(decomposition_excess_io - excess_io))}"
            )

    for func_ in [
        total_brier_outer,
        total_logscore_outer,
        total_neglog_outer,
        total_maxprob_outer,
        total_spherical_outer,
        bayes_brier_inner,
        bayes_brier_outer,
        bayes_logscore_inner,
        bayes_logscore_outer,
        bayes_maxprob_inner,
        bayes_maxprob_outer,
        bayes_neglog_inner,
        bayes_neglog_outer,
        bayes_spherical_inner,
        bayes_spherical_outer,
        excess_brier_inner_outer,
        excess_brier_outer_outer,
        excess_brier_outer_inner,
        excess_logscore_inner_outer,
        excess_logscore_outer_outer,
        excess_logscore_outer_inner,
        excess_maxprob_inner_outer,
        excess_maxprob_outer_outer,
        excess_maxprob_outer_inner,
        excess_neglog_inner_outer,
        excess_neglog_outer_outer,
        excess_neglog_outer_inner,
        excess_spherical_inner_outer,
        excess_spherical_outer_outer,
        excess_spherical_outer_inner,
        bi_brier,
        bi_logscore,
        bi_maxprob,
        bi_spherical,
        bi_neglog,
        rbi_brier,
        rbi_logscore,
        rbi_maxprob,
        rbi_spherical,
        rbi_neglog,
        bias_brier,
        bias_logscore,
        bias_maxprob,
        bias_spherical,
        bias_neglog,
        bias_bi_brier,
        bias_bi_logscore,
        bias_bi_maxprob,
        bias_bi_spherical,
        bias_bi_neglog,
        mv_brier,
        mv_logscore,
        mv_maxprob,
        mv_spherical,
        mv_neglog,
        mv_bi_brier,
        mv_bi_logscore,
        mv_bi_maxprob,
        mv_bi_spherical,
        mv_bi_neglog,
    ]:
        try:
            squared_dist_results = func_(A, B)
            assert squared_dist_results.shape == (N_objects,)
        except Exception as ex:
            print("NOOO")
            print(func_)
            print(ex)

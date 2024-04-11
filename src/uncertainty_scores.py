import numpy as np
from scipy.special import logsumexp


def safe_softmax(x):
    """Softmax

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def safe_kl_divergence(left_logits, right_logits):
    """KL divergence

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    p_safe = safe_softmax(left_logits)
    return np.sum(
        p_safe * (left_logits - right_logits +
                  logsumexp(right_logits, axis=-1, keepdims=True)
                  - logsumexp(left_logits, axis=-1, keepdims=True)), axis=-1)


def pairwise_kl(logits_gt, logits_pred):
    """Pairwise KL for two np.ndarray objects

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Expand P and Q to have shape
    # (N_members_A, N_members_B, N_objects, N_classes)
    # for broadcasting. This creates every combination
    #  of pairs between A and B.
    logits_gt_exp = logits_gt[:, np.newaxis, :, :]
    logits_pred_exp = logits_pred[np.newaxis, :, :, :]

    # Compute KL divergence for each combination of pairs
    kl_divs = safe_kl_divergence(logits_gt_exp, logits_pred_exp)

    # kl_divs now has shape (N_members_A, N_members_B, N_objects)
    # You might want to aggregate this further depending on your needs,
    # e.g., mean over objects.

    return kl_divs


def pairwise_ce(logits_gt, logits_pred):
    """Pairwise CE for two np.ndarray objects

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    logits_gt_exp = logits_gt[:, np.newaxis, :, :]
    logits_pred_exp = logits_pred[np.newaxis, :, :, :]

    gt_exp = safe_softmax(logits_gt_exp)

    ce_divs = np.sum(
        -gt_exp * (
            logits_pred_exp - logsumexp(
                logits_pred_exp, axis=-1, keepdims=True)
        ), axis=-1
    )

    return ce_divs


def pairwise_brier(logits_gt, logits_pred):
    """Pairwise Brier

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    p_exp = safe_softmax(logits_gt)[:, np.newaxis, :, :]
    q_exp = safe_softmax(logits_pred)[np.newaxis, :, :, :]

    brier_divs = np.sum((p_exp - q_exp) ** 2, axis=-1)

    return brier_divs


def pairwise_prob_diff(logits_gt, logits_pred):
    """Pairwise probability difference

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    prob_gt = safe_softmax(logits_gt)
    prob_pred = safe_softmax(logits_pred)

    argmax_indices = np.argmax(prob_pred, axis=-1)
    members_index = np.arange(prob_pred.shape[0])[:, None, None]
    objects_index = np.arange(prob_pred.shape[1])
    max_gt = prob_gt[members_index, objects_index, argmax_indices[:, None, :]]

    prob_divs = np.max(prob_gt[None, ...], axis=-1) - max_gt

    return prob_divs


def pairwise_IS_distance(logits_gt, logits_pred):
    """Pairwise Itakura–Saito distance

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    logits_gt = logits_gt[:, np.newaxis, :, :]
    logits_pred = logits_pred[:, np.newaxis, :, :]

    p_exp = safe_softmax(logits_gt)
    q_exp = safe_softmax(logits_pred)

    is_dist = np.sum(p_exp / q_exp -
                     logits_gt + logits_pred +
                     logsumexp(logits_gt, axis=-1, keepdims=True) -
                     logsumexp(logits_pred, axis=-1, keepdims=True) - 1,
                     axis=-1
                     )

    return is_dist


def pairwise_spherical(logits_gt, logits_pred):
    """Pairwise spherical

    Args:
        logits_gt (_type_): _description_
        logits_pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    p_exp = safe_softmax(logits_gt)[:, np.newaxis, :, :]
    q_exp = safe_softmax(logits_pred)[np.newaxis, :, :, :]

    p_exp_normed = p_exp / np.linalg.norm(p_exp, ord=2, axis=-1, keepdims=True)
    q_exp_normed = q_exp / np.linalg.norm(q_exp, ord=2, axis=-1, keepdims=True)

    p_normed = np.linalg.norm(p_exp, ord=2, axis=-1)
    pairwise_dot = np.sum(p_exp_normed * q_exp_normed, axis=-1)

    spherical_divs = p_normed * (1 - pairwise_dot)

    return spherical_divs


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=-1)


def posterior_predictive(logits_):
    prob_p = safe_softmax(logits_)
    ppd = np.mean(prob_p, axis=0, keepdims=True)
    return ppd


def mutual_information(logits_pred, logits_gt=None):
    """
    Mutual information, computed as HE - EH
    """
    HE = bayes_logscore_inner(logits_pred=logits_pred, logits_gt=None)
    EH = bayes_logscore_outer(logits_pred=logits_pred, logits_gt=None)

    bald_scores = HE - EH

    return bald_scores


def mutual_information_avg_kl(logits_pred, logits_gt=None):
    """
    Mutual information, computed as Expected KL[pred | ppd]
    """
    prob_p = safe_softmax(logits_pred)
    avg_predictions = posterior_predictive(logits_=logits_pred)
    return np.mean(
        np.sum(
            prob_p * np.log(prob_p / avg_predictions), axis=-1
        ), axis=0)


def logscore_central_prediction(
        logits_: np.ndarray,
):
    probs = safe_softmax(logits_)
    return safe_softmax(np.mean(np.log(probs), axis=0, keepdims=True))


def kl_bias(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    ppd = posterior_predictive(logits_=logits_gt)
    central_pred = logscore_central_prediction(logits_pred)
    return safe_kl_divergence(
        left_logits=np.log(ppd),
        right_logits=np.log(central_pred)
    ).squeeze()


def kl_model_variance(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    central_pred = logscore_central_prediction(logits_gt)
    prob_b = safe_softmax(logits_pred)
    return np.mean(
        np.sum(
            central_pred * np.log(central_pred / prob_b), axis=-1
        ), axis=0)


def kl_model_variance_plus_mi(
    logits_pred: np.ndarray,
    logits_gt: np.ndarray,
):
    return kl_model_variance(logits_pred=logits_pred, logits_gt=logits_gt) \
        + mutual_information_avg_kl(logits_pred=logits_pred,
                                    logits_gt=logits_gt)

############################################################


def total_neglog(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return (
        excess_neglog_outer(
            logits_gt=logits_gt, logits_pred=logits_pred)
        + bayes_neglog_outer(
            logits_gt=logits_gt, logits_pred=logits_pred)
    )


def bayes_neglog_outer(logits_gt, logits_pred=None):
    return np.mean(
        np.sum(
            logits_gt - logsumexp(logits_gt, axis=-1, keepdims=True),
            axis=-1), axis=0)


def bayes_neglog_inner(logits_gt, logits_pred=None):
    avg_predictions = posterior_predictive(logits_=logits_gt)[0]
    return np.sum(np.log(avg_predictions), axis=-1)


def excess_neglog_outer(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_IS_distance(logits_gt, logits_pred), axis=(0, 1)
    )


def excess_neglog_inner(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    logits_gt = posterior_predictive(logits_gt)

    p_exp = safe_softmax(logits_gt)
    q_exp = safe_softmax(logits_pred)

    is_dist = np.sum(p_exp / q_exp -
                     logits_gt + logits_pred +
                     logsumexp(logits_gt, axis=-1, keepdims=True) -
                     logsumexp(logits_pred, axis=-1, keepdims=True) - 1,
                     axis=-1
                     )

    return np.mean(is_dist, axis=0)

############################################################


def total_spherical(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return (
        excess_spherical_outer(
            logits_gt=logits_gt, logits_pred=logits_pred)
        + bayes_spherical_outer(
            logits_gt=logits_gt, logits_pred=logits_pred)
    )


def bayes_spherical_outer(logits_gt, logits_pred=None):
    prob_p = safe_softmax(logits_gt)
    return -np.mean(np.linalg.norm(prob_p, axis=-1), axis=0)


def bayes_spherical_inner(logits_gt, logits_pred=None):
    avg_predictions = posterior_predictive(logits_=logits_gt)[0]
    return -np.linalg.norm(avg_predictions, axis=-1)


def excess_spherical_outer(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_spherical(logits_gt, logits_pred), axis=(0, 1)
    )


def excess_spherical_inner(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    ppd = safe_softmax(logits_gt)
    pred = safe_softmax(logits_pred)

    ppd_normed = ppd / np.linalg.norm(ppd, ord=2, axis=-1, keepdims=True)
    pred_normed = pred / np.linalg.norm(pred, ord=2, axis=-1, keepdims=True)

    p_normed = np.linalg.norm(ppd, ord=2, axis=-1)
    dot_prods = np.sum(ppd_normed * pred_normed, axis=-1)

    spherical_divs = p_normed * (1 - dot_prods)

    return np.mean(spherical_divs, axis=0)

############################################################


def total_maxprob(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return (
        excess_maxprob_outer(
            logits_gt=logits_gt, logits_pred=logits_pred)
        + bayes_maxprob_outer(
            logits_gt=logits_gt, logits_pred=logits_pred)
    )


def bayes_maxprob_outer(logits_gt, logits_pred=None):
    prob_p = safe_softmax(logits_gt)
    return np.mean(1 - np.max(prob_p, axis=-1), axis=0)


def bayes_maxprob_inner(logits_gt, logits_pred=None):
    avg_predictions = posterior_predictive(logits_=logits_gt)[0]
    return 1 - np.max(avg_predictions, axis=-1)


def excess_maxprob_outer(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_prob_diff(
            logits_gt=logits_gt, logits_pred=logits_pred), axis=(0, 1)
    )


def excess_maxprob_inner(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    avg_predictions = posterior_predictive(logits_=logits_gt)
    prob_pred = safe_softmax(logits_pred)

    argmax_indices = np.argmax(prob_pred, axis=-1)
    objects_index = np.arange(prob_pred.shape[1])
    max_gt = avg_predictions[:, objects_index, argmax_indices][0]

    prob_divs = np.mean(np.max(avg_predictions, axis=-1) - max_gt, axis=0)

    return prob_divs


############################################################


def total_brier(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return (excess_brier_outer(logits_gt=logits_gt, logits_pred=logits_pred)
            + bayes_brier_outer(logits_gt=logits_gt, logits_pred=logits_pred))


def bayes_brier_outer(logits_gt, logits_pred=None):
    return 1 - np.mean(
        np.linalg.norm(logits_gt, ord=2, axis=-1), axis=0
    )


def bayes_brier_inner(logits_gt, logits_pred=None):
    avg_predictions = posterior_predictive(logits_=logits_gt)[0]
    return 1 - np.linalg.norm(avg_predictions, ord=2, axis=-1)


def excess_brier_outer(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_brier(logits_gt, logits_pred), axis=(0, 1)
    )


def excess_brier_inner(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    avg_predictions = posterior_predictive(logits_=logits_pred)
    pred_prob = safe_softmax(logits_gt)
    return np.mean(
        np.linalg.norm(avg_predictions - pred_prob, ord=2, axis=-1), axis=0
    )


############################################################

def total_logscore(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    """
    Expected Pairwise Cross Entropy
    """
    return np.mean(
        pairwise_ce(logits_gt, logits_pred), axis=(0, 1)
    )


def excess_logscore_outer(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    """
    Expected Pairwise Kullback Leibler
    """
    return np.mean(
        pairwise_kl(logits_gt, logits_pred), axis=(0, 1)
    )


def excess_logscore_inner(logits_gt, logits_pred):
    """
    Reverse mutual information, computed as Expected KL[ppd | pred]
    """
    prob_p = safe_softmax(logits_pred)
    avg_predictions = posterior_predictive(logits_=logits_gt)
    return np.mean(
        np.sum(
            avg_predictions * np.log(avg_predictions / prob_p), axis=-1
        ), axis=0)


def bayes_logscore_outer(logits_gt, logits_pred=None):
    prob_p = safe_softmax(logits_gt)
    return np.mean(entropy(prob_p), axis=0)


def bayes_logscore_inner(logits_gt, logits_pred=None):
    avg_predictions = posterior_predictive(logits_=logits_gt)[0]
    return entropy(avg_predictions)


if __name__ == '__main__':
    # Example usage
    N_members, N_objects, N_classes = 3, 40, 5  # Example dimensions
    A = np.random.randn(N_members, N_objects, N_classes)
    B = np.random.randn(N_members, N_objects, N_classes)

    # squared_dist_results = bayes_neglog_inner(A, B)
    # print(squared_dist_results.shape)

    for func_ in [
        total_brier,
        total_logscore,
        total_neglog,
        total_maxprob,
        total_spherical,
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
        excess_brier_inner,
        excess_brier_outer,
        excess_logscore_inner,
        excess_logscore_outer,
        excess_maxprob_inner,
        excess_maxprob_outer,
        excess_neglog_inner,
        excess_neglog_outer,
        excess_spherical_inner,
        excess_spherical_outer
    ]:
        try:
            squared_dist_results = func_(A, B)
            assert squared_dist_results.shape == (N_objects, )
        except Exception as ex:
            print(func_)
            print(ex)

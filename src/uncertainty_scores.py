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


def entropy_average(logits_pred, logits_gt=None):
    """HE -- entropy of an average prediction

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    avg_predictions = posterior_predictive(logits_=logits_pred)[0]
    return entropy(avg_predictions)


def average_entropy(logits_pred, logits_gt=None):
    """Average entropy

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prob_p = safe_softmax(logits_pred)
    individual_entropies = entropy(prob_p)
    average_entropy_individual = np.mean(individual_entropies, axis=0)
    return average_entropy_individual


def mutual_information(logits_pred, logits_gt=None):
    """Mutual information, computed as HE - EH

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    HE = entropy_average(logits_pred=logits_pred, logits_gt=None)
    EH = average_entropy(logits_pred=logits_pred, logits_gt=None)

    bald_scores = HE - EH

    return bald_scores


def mutual_information_avg_kl(logits_pred, logits_gt=None):
    """Mutual information, computed as Expected KL[pred | ppd]

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prob_p = safe_softmax(logits_pred)
    avg_predictions = posterior_predictive(logits_=logits_pred)
    return np.mean(
        np.sum(
            prob_p * np.log(prob_p / avg_predictions), axis=-1
        ), axis=0)


def reverse_mutual_information(logits_pred, logits_gt=None):
    """Reverse mutual information, computed as Expected KL[ppd | pred]

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prob_p = safe_softmax(logits_pred)
    avg_predictions = posterior_predictive(logits_=logits_pred)
    return np.mean(
        np.sum(
            avg_predictions * np.log(avg_predictions / prob_p), axis=-1
        ), axis=0)


def expected_pairwise_kl(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    """EPKL

    Args:
        logits_gt (np.ndarray): _description_
        logits_pred (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    return np.mean(
        pairwise_kl(logits_gt, logits_pred), axis=(0, 1)
    )


def expected_pairwise_brier(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_brier(logits_gt, logits_pred), axis=(0, 1)
    )


def expected_pairwise_spherical(
        logits_gt: np.ndarray,
        logits_pred: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_spherical(logits_gt, logits_pred), axis=(0, 1)
    )


def maxprob_average(logits_pred, logits_gt=None):
    """1 - max ppd

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    avg_predictions = posterior_predictive(logits_=logits_pred)[0]
    return 1 - np.max(avg_predictions, axis=-1)


def average_maxprob(logits_pred, logits_gt=None):
    """ Expected (1 - max pred)

    Args:
        logits_pred (_type_): _description_
        logits_gt (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    prob_p = safe_softmax(logits_pred)
    return np.mean(1 - np.max(prob_p, axis=-1), axis=0)


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


if __name__ == '__main__':
    # Example usage
    N_members, N_objects, N_classes = 3, 40, 5  # Example dimensions
    A = np.random.randn(N_members, N_objects, N_classes)
    B = np.random.randn(N_members, N_objects, N_classes)

    squared_dist_results = kl_model_variance_plus_mi(A, B)
    print(squared_dist_results.shape)

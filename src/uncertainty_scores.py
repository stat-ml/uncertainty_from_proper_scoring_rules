import numpy as np
from scipy.special import logsumexp


def safe_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def safe_kl_divergence(logits_p, logits_q):
    p_safe = safe_softmax(logits_p)
    return np.sum(
        p_safe * (logits_p - logits_q +
                  logsumexp(logits_q, axis=-1, keepdims=True)
                  - logsumexp(logits_p, axis=-1, keepdims=True)), axis=-1)


def pairwise_kl(logits_p, logits_q):
    # Expand P and Q to have shape
    # (N_members_A, N_members_B, N_objects, N_classes)
    # for broadcasting. This creates every combination
    #  of pairs between A and B.
    logits_p_exp = logits_p[:, np.newaxis, :, :]
    logits_q_exp = logits_q[np.newaxis, :, :, :]

    # Compute KL divergence for each combination of pairs
    kl_divs = safe_kl_divergence(logits_p_exp, logits_q_exp)

    # kl_divs now has shape (N_members_A, N_members_B, N_objects)
    # You might want to aggregate this further depending on your needs,
    # e.g., mean over objects.

    return kl_divs


def pairwise_brier(logits_p, logits_q):
    p_exp = safe_softmax(logits_p)[:, np.newaxis, :, :]
    q_exp = safe_softmax(logits_q)[np.newaxis, :, :, :]

    brier_divs = np.sum((p_exp - q_exp) ** 2, axis=-1)

    return brier_divs


def pairwise_spherical(logits_p, logits_q):
    p_exp = safe_softmax(logits_p)[:, np.newaxis, :, :]
    q_exp = safe_softmax(logits_q)[np.newaxis, :, :, :]

    p_exp_normed = p_exp / np.linalg.norm(p_exp, ord=2, axis=-1, keepdims=True)
    q_exp_normed = q_exp / np.linalg.norm(q_exp, ord=2, axis=-1, keepdims=True)

    p_normed = np.linalg.norm(p_exp, ord=2, axis=-1)
    pairwise_dot = np.sum(p_exp_normed * q_exp_normed, axis=-1)

    spherical_divs = p_normed * (1 - pairwise_dot)

    return spherical_divs


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=-1)


def entropy_average(logits_p, logits_q):
    prob_p = safe_softmax(logits_p)
    avg_predictions = np.mean(prob_p, axis=0)
    return entropy(avg_predictions)


def average_entropy(logits_p, logits_q):
    prob_p = safe_softmax(logits_p)
    individual_entropies = entropy(prob_p)
    average_entropy_individual = np.mean(individual_entropies, axis=0)
    return average_entropy_individual


def mutual_information(logits_p, logits_q):
    HE = entropy_average(logits_p, logits_p)
    EH = average_entropy(logits_p, logits_p)

    bald_scores = HE - EH

    return bald_scores


def reverse_mutual_information(logits_p, logits_q):
    prob_p = safe_softmax(logits_p)
    avg_predictions = np.mean(prob_p, axis=0, keepdims=True)
    return np.mean(
        np.sum(
            avg_predictions * np.log(avg_predictions / prob_p), axis=-1
        ), axis=0)


def expected_pairwise_kl(
        logits_p: np.ndarray,
        logits_q: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_kl(logits_p, logits_q), axis=(0, 1)
    )


def expected_pairwise_brier(
        logits_p: np.ndarray,
        logits_q: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_brier(logits_p, logits_q), axis=(0, 1)
    )


def expected_pairwise_spherical(
        logits_p: np.ndarray,
        logits_q: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_spherical(logits_p, logits_q), axis=(0, 1)
    )


def maxprob_average(logits_p, logits_1):
    prob_p = safe_softmax(logits_p)
    avg_predictions = np.mean(prob_p, axis=0)
    return 1 - np.max(avg_predictions, axis=-1)


def average_maxprob(logits_p, logits_1):
    prob_p = safe_softmax(logits_p)
    return np.mean(1 - np.max(prob_p, axis=-1), axis=0)


if __name__ == '__main__':
    # Example usage
    N_members, N_objects, N_classes = 3, 40, 5  # Example dimensions
    A = np.random.randn(N_members, N_objects, N_classes)
    B = np.random.randn(N_members, N_objects, N_classes)

    squared_dist_results = average_maxprob(A, B)
    print(squared_dist_results.shape)

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


def expected_pairwise_kl(
        logits_p: np.ndarray,
        logits_q: np.ndarray
) -> np.ndarray:
    return np.mean(
        pairwise_kl(logits_p, logits_q), axis=(0, 1)
    )


if __name__ == '__main__':
    # Example usage
    N_members, N_objects, N_classes = 3, 123, 5  # Example dimensions
    A = np.random.randn(N_members, N_objects, N_classes)
    B = np.random.randn(N_members, N_objects, N_classes)

    kl_results = pairwise_kl(A, B)
    print(kl_results.shape)  # This will show (N_members, N_members, N_objects)

    epkl_results = expected_pairwise_kl(A, B)
    # This will show (N_members, N_members, N_objects)
    print(epkl_results.shape)

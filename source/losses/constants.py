from enum import Enum


class LossName(Enum):
    CROSS_ENTROPY = "CrossEntropy"
    NLL = "NLLLoss"
    BRIER_SCORE = "BrierScore"
    SPHERICAL_SCORE = "SphericalScore"
    FOCAL_LOSS = "FocalLoss"
    BINARY_CROSS_ENTROPY = "BinaryCrossEntropy"
    # NEG_LOG_SCORE = "NegLog"

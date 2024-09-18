from enum import Enum


class LossName(Enum):
    CROSS_ENTROPY = "CrossEntropy"
    BRIER_SCORE = "BrierScore"
    SPHERICAL_SCORE = "SphericalScore"
    # NEG_LOG_SCORE = "NegLog"

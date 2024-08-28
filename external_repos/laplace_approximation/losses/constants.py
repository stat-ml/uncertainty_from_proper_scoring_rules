from enum import StrEnum


class LossName(StrEnum):
    CROSS_ENTROPY = "CrossEntropy"
    BRIER_SCORE = "BrierScore"
    SPHERICAL_SCORE = "SphericalScore"
    NEG_LOG_SCORE = "NegLog"
    
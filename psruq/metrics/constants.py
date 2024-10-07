from enum import Enum


class GName(Enum):
    LOG_SCORE = "LogScore"
    BRIER_SCORE = "BrierScore"
    ZERO_ONE_SCORE = "ZeroOneScore"
    SPHERICAL_SCORE = "SphericalScore"


class RiskType(Enum):
    TOTAL_RISK = "TotalRisk"
    EXCESS_RISK = "ExcessRisk"
    BAYES_RISK = "BayesRisk"


class ApproximationType(Enum):
    OUTER = "outer"
    INNER = "inner"
    CENTRAL = "central"

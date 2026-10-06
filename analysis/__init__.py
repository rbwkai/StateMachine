"""
analysis package for DWS-Bench.
"""

from .query_analysis import (
    QueryAnalysis,
    QuerySpec,
    analyze_trajectory,
)
from .failure_onset import (
    CurveFitResult,
    FailureProfile,
    LogitSlopeResult,
    MIN_CURVE_LEVELS,
    best_fitting_curve,
    compare_curves,
    compute_failure_onset,
    fit_exponential,
    fit_linear,
    fit_sigmoid,
    slope_logit,
)
from .first_error import (
    ErrorType,
    TrajectoryErrorAnalysis,
    analyze_first_error,
)
from .statistics import (
    CONFIDENCE_LEVEL,
    MAX_WILSON_HALF_WIDTH,
    MIN_SUCCESSES_FOR_REPORTING,
    McNemarResult,
    flag_low_success_cells,
    holm_correction,
    mcnemar_paired,
    mcnemar_test,
    wilson_interval,
)

__all__ = [
    "QueryAnalysis",
    "QuerySpec",
    "analyze_trajectory",
    "CurveFitResult",
    "FailureProfile",
    "compute_failure_onset",
    "fit_linear",
    "fit_exponential",
    "fit_sigmoid",
    "compare_curves",
    "best_fitting_curve",
    "MIN_CURVE_LEVELS",
    "LogitSlopeResult",
    "slope_logit",
    "ErrorType",
    "TrajectoryErrorAnalysis",
    "analyze_first_error",
    "CONFIDENCE_LEVEL",
    "MIN_SUCCESSES_FOR_REPORTING",
    "McNemarResult",
    "wilson_interval",
    "mcnemar_test",
    "flag_low_success_cells",
    "MAX_WILSON_HALF_WIDTH",
    "mcnemar_paired",
    "holm_correction",
]

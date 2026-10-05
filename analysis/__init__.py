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
    best_fitting_curve,
    compare_curves,
    compute_failure_onset,
    fit_exponential,
    fit_linear,
    fit_sigmoid,
)
from .first_error import (
    ErrorType,
    TrajectoryErrorAnalysis,
    analyze_first_error,
)
from .statistics import (
    CONFIDENCE_LEVEL,
    MIN_SUCCESSES_FOR_REPORTING,
    McNemarResult,
    flag_low_success_cells,
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
    "ErrorType",
    "TrajectoryErrorAnalysis",
    "analyze_first_error",
    "CONFIDENCE_LEVEL",
    "MIN_SUCCESSES_FOR_REPORTING",
    "McNemarResult",
    "wilson_interval",
    "mcnemar_test",
    "flag_low_success_cells",
]

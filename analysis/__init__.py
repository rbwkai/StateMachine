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
]

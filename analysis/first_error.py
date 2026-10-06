"""
analysis/first_error.py
=======================
First-Error and Trajectory Error Classification for DWS-Bench.

Implements §12 of the research plan:
Given a gold trajectory [S_0, S_1, ..., S_n] and predicted trajectory [Ŝ_0, Ŝ_1, ..., Ŝ_n]:
- Identify first error step: t = min { i : S_i != Ŝ_i }
- Classify error dynamics:
  1. LOCAL_ERROR: Model makes an incorrect transition but later recovers (Ŝ_j == S_j for some j > t).
  2. PROPAGATING_ERROR: One incorrect transition causes all subsequent states to be wrong.
  3. FINAL_ONLY_ERROR: All intermediate states are correct (Ŝ_i == S_i for i < n), but the final answer is wrong.
  4. CANCELLATION_ERROR: An intermediate error occurred (t < n), but a later operation accidentally restored the correct final answer (Ŝ_n == S_n).
  5. NO_ERROR: Complete agreement across all steps.
  6. MISSING: No observed step is wrong, but at least one step could not be
     parsed (prediction is None), so agreement cannot be confirmed.

A None prediction means "unparsed", not "wrong": it is never counted in
`step_errors`, the first-error onset is the first non-None wrong step, and the
number of unparsed steps is reported as `n_missing`. A predicted trajectory
shorter than the gold one is padded with None (missing tail).

FINAL_ONLY_ERROR asserts that every intermediate state is correct, so it is
assigned only when no step before the first error is unparsed
(`has_missing_before_first_error` is False). When the only observed error is
the final step but an earlier step is None, the intermediates cannot be
confirmed: the trajectory falls through to the LOCAL/PROPAGATING rules with
unparsed steps treated as not-recovered, which yields PROPAGATING_ERROR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ErrorType(Enum):
    NO_ERROR = auto()
    LOCAL_ERROR = auto()
    PROPAGATING_ERROR = auto()
    FINAL_ONLY_ERROR = auto()
    CANCELLATION_ERROR = auto()
    MISSING = auto()


@dataclass
class TrajectoryErrorAnalysis:
    error_type: ErrorType
    first_error_step: Optional[int]
    total_steps: int
    step_errors: List[int]
    gold_final: Any
    pred_final: Any
    final_is_correct: bool
    # Unparsed (None) steps, 1-indexed; they are excluded from step_errors.
    missing_steps: List[int] = field(default_factory=list)
    # True when an unparsed step precedes the first observed error, so
    # "all earlier states correct" cannot be confirmed.
    has_missing_before_first_error: bool = False

    @property
    def n_missing(self) -> int:
        return len(self.missing_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.name,
            "first_error_step": self.first_error_step,
            "total_steps": self.total_steps,
            "step_errors": self.step_errors,
            "gold_final": self.gold_final,
            "pred_final": self.pred_final,
            "final_is_correct": self.final_is_correct,
            "missing_steps": self.missing_steps,
            "n_missing": self.n_missing,
            "has_missing_before_first_error": self.has_missing_before_first_error,
        }


def analyze_first_error(
    gold_states: List[Any],
    pred_states: List[Any],
) -> TrajectoryErrorAnalysis:
    """
    Classify the trajectory error between gold step-wise states and model predictions.
    
    Note: gold_states[0] is the initial state (after first Put), while model's Step 1
    corresponds to the first Move. We align by skipping gold_states[0] for comparison.
    """
    # Align: gold_states[0] is initial state (after Put), pred_states[0] is model's Step 1 (first move)
    # So we compare pred_states[i] with gold_states[i+1]. A shorter prediction is
    # a missing tail (the model stopped early), padded with None; a longer one
    # cannot be aligned and is rejected.
    n = len(gold_states) - 1
    if n < 0 or len(pred_states) > n:
        raise ValueError(
            f"Trajectory length mismatch: gold={len(gold_states)}, pred={len(pred_states)}. "
            "Expected gold >= pred + 1 (gold includes initial state)"
        )
    preds: List[Any] = list(pred_states) + [None] * (n - len(pred_states))

    if n == 0:
        return TrajectoryErrorAnalysis(
            error_type=ErrorType.NO_ERROR,
            first_error_step=None,
            total_steps=0,
            step_errors=[],
            gold_final=None,
            pred_final=None,
            final_is_correct=True,
        )

    missing = [i for i in range(n) if preds[i] is None]
    step_errors = [
        i for i in range(n)
        if preds[i] is not None and gold_states[i + 1] != preds[i]
    ]
    final_is_correct = preds[-1] is not None and gold_states[-1] == preds[-1]
    missing_1 = [i + 1 for i in missing]

    if not step_errors:
        return TrajectoryErrorAnalysis(
            error_type=ErrorType.MISSING if missing else ErrorType.NO_ERROR,
            first_error_step=None,
            total_steps=n,
            step_errors=[],
            gold_final=gold_states[-1],
            pred_final=preds[-1],
            final_is_correct=final_is_correct,
            missing_steps=missing_1,
        )

    t = step_errors[0]  # 0-indexed in preds
    error_set = set(step_errors)
    missing_before = any(i < t for i in missing)

    # Final-only error: the final step is the first error AND every earlier
    # step was observed correct. An unparsed earlier step leaves the
    # intermediates unverified, so it falls through to the rules below.
    if t == n - 1 and not missing_before:
        error_type = ErrorType.FINAL_ONLY_ERROR

    # Check for cancellation error: intermediate error occurred, but final answer is correct
    elif final_is_correct:
        error_type = ErrorType.CANCELLATION_ERROR

    else:
        # Final answer is wrong or unparsed. Recovery needs an observed correct
        # step after t; an unparsed step is not evidence of recovery.
        recovered = any(
            preds[i] is not None and i not in error_set for i in range(t + 1, n)
        )
        if recovered:
            error_type = ErrorType.LOCAL_ERROR
        else:
            error_type = ErrorType.PROPAGATING_ERROR

    return TrajectoryErrorAnalysis(
        error_type=error_type,
        first_error_step=t + 1,  # 1-indexed for human-readable step number
        total_steps=n,
        step_errors=[e + 1 for e in step_errors],  # 1-indexed
        gold_final=gold_states[-1],
        pred_final=preds[-1],
        final_is_correct=final_is_correct,
        missing_steps=missing_1,
        has_missing_before_first_error=missing_before,
    )

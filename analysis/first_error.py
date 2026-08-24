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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class ErrorType(Enum):
    NO_ERROR = auto()
    LOCAL_ERROR = auto()
    PROPAGATING_ERROR = auto()
    FINAL_ONLY_ERROR = auto()
    CANCELLATION_ERROR = auto()


@dataclass
class TrajectoryErrorAnalysis:
    error_type: ErrorType
    first_error_step: Optional[int]
    total_steps: int
    step_errors: List[int]
    gold_final: Any
    pred_final: Any
    final_is_correct: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.name,
            "first_error_step": self.first_error_step,
            "total_steps": self.total_steps,
            "step_errors": self.step_errors,
            "gold_final": self.gold_final,
            "pred_final": self.pred_final,
            "final_is_correct": self.final_is_correct,
        }


def analyze_first_error(
    gold_states: List[Any],
    pred_states: List[Any],
) -> TrajectoryErrorAnalysis:
    """
    Classify the trajectory error between gold step-wise states and model predictions.
    """
    if len(gold_states) != len(pred_states):
        raise ValueError(
            f"Trajectory length mismatch: gold={len(gold_states)}, pred={len(pred_states)}"
        )

    n = len(gold_states)
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

    step_errors = [i for i in range(n) if gold_states[i] != pred_states[i]]
    final_is_correct = (gold_states[-1] == pred_states[-1])

    if not step_errors:
        return TrajectoryErrorAnalysis(
            error_type=ErrorType.NO_ERROR,
            first_error_step=None,
            total_steps=n,
            step_errors=[],
            gold_final=gold_states[-1],
            pred_final=pred_states[-1],
            final_is_correct=True,
        )

    t = step_errors[0]

    # Check for final-only error: only the final step is wrong
    if t == n - 1 and len(step_errors) == 1:
        error_type = ErrorType.FINAL_ONLY_ERROR

    # Check for cancellation error: intermediate error occurred, but final answer is correct
    elif final_is_correct:
        error_type = ErrorType.CANCELLATION_ERROR

    else:
        # Final answer is wrong.
        # Check if model recovered at any point after step t
        recovered = any(i not in step_errors for i in range(t + 1, n))
        if recovered:
            error_type = ErrorType.LOCAL_ERROR
        else:
            error_type = ErrorType.PROPAGATING_ERROR

    return TrajectoryErrorAnalysis(
        error_type=error_type,
        first_error_step=t,
        total_steps=n,
        step_errors=step_errors,
        gold_final=gold_states[-1],
        pred_final=pred_states[-1],
        final_is_correct=final_is_correct,
    )

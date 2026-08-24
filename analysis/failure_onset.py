"""
analysis/failure_onset.py
=========================
Curve Fitting and Failure Onset Analysis for DWS-Bench.

Implements §10 (Curve Analysis) and §11 (Failure Onset) of the research plan:
1. Fit candidate difficulty curves:
   - Linear: A(x) = a + b*x
   - Exponential: A(x) = a * exp(-b*x) + c
   - Sigmoid: A(x) = c + (a - c) / (1 + exp(b * (x - x_0)))
2. Model selection using R² and AIC (Akaike Information Criterion).
3. Formal failure onset determination:
   L_f = min{ x : A(x) < τ } (default τ = 0.70)
4. Multi-dimensional model failure profile: M = (L_T, L_D, L_V, L_E).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class CurveFitResult:
    """Results of fitting a candidate curve model to accuracy data."""
    model_type: str
    params: Dict[str, float]
    r_squared: float
    aic: float
    predictions: List[float]


@dataclass
class FailureProfile:
    """Per-model failure profile across reasoning dimensions."""
    model_name: str
    tau: float
    L_T: Optional[int] = None  # Temporal depth onset
    L_D: Optional[int] = None  # Distractor interference onset
    L_V: Optional[int] = None  # Revision complexity onset
    L_E: Optional[int] = None  # Multi-entity onset

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tau": self.tau,
            "L_T": self.L_T,
            "L_D": self.L_D,
            "L_V": self.L_V,
            "L_E": self.L_E,
        }


def compute_failure_onset(
    x_values: List[Union[int, float]],
    accuracies: List[float],
    tau: float = 0.70,
) -> Optional[Union[int, float]]:
    """
    Compute failure onset L_f = min { x : A(x) < tau }.
    Assumes x_values and accuracies are sorted in increasing order of difficulty x.
    """
    if len(x_values) != len(accuracies):
        raise ValueError("x_values and accuracies must have identical length")

    for x, acc in zip(x_values, accuracies):
        if acc < tau:
            return x
    return None


def fit_linear(
    x_values: List[float],
    y_values: List[float],
) -> CurveFitResult:
    """
    Fit linear model: A(x) = a + b * x
    """
    n = len(x_values)
    if n < 2:
        return CurveFitResult("linear", {"a": 0.0, "b": 0.0}, 0.0, float("inf"), y_values)

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    b = numerator / denominator if denominator != 0 else 0.0
    a = y_mean - b * x_mean

    preds = [a + b * x for x in x_values]
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    ss_res = sum((y - pred) ** 2 for y, pred in zip(y_values, preds))

    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    k = 2  # number of parameters (a, b)
    aic = n * math.log(max(ss_res / n, 1e-12)) + 2 * k

    return CurveFitResult("linear", {"a": a, "b": b}, r_squared, aic, preds)


def fit_exponential(
    x_values: List[float],
    y_values: List[float],
) -> CurveFitResult:
    """
    Fit exponential decay: A(x) = a * exp(-b * x) + c (with grid search / bounded optimization)
    """
    n = len(x_values)
    if n < 3:
        return fit_linear(x_values, y_values)

    best_res = float("inf")
    best_params = {"a": 1.0, "b": 0.1, "c": 0.0}
    best_preds = y_values

    # Grid search for robust initialization
    for c_val in [0.0, 0.1, 0.2, 0.25]:
        for b_val in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
            # a = (y - c) / exp(-b * x)
            exp_terms = [math.exp(-b_val * x) for x in x_values]
            denom = sum(e ** 2 for e in exp_terms)
            if denom == 0:
                continue
            a_val = sum((y - c_val) * e for y, e in zip(y_values, exp_terms)) / denom
            preds = [a_val * e + c_val for e in exp_terms]
            ss_res = sum((y - p) ** 2 for y, p in zip(y_values, preds))
            if ss_res < best_res:
                best_res = ss_res
                best_params = {"a": a_val, "b": b_val, "c": c_val}
                best_preds = preds

    y_mean = sum(y_values) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r_squared = 1.0 - (best_res / ss_tot) if ss_tot > 0 else 1.0
    k = 3  # parameters (a, b, c)
    aic = n * math.log(max(best_res / n, 1e-12)) + 2 * k

    return CurveFitResult("exponential", best_params, r_squared, aic, best_preds)


def fit_sigmoid(
    x_values: List[float],
    y_values: List[float],
) -> CurveFitResult:
    """
    Fit sigmoid: A(x) = c + (a - c) / (1 + exp(b * (x - x_0)))
    """
    n = len(x_values)
    if n < 4:
        return fit_linear(x_values, y_values)

    best_res = float("inf")
    best_params = {"a": 1.0, "c": 0.0, "b": 0.5, "x_0": sum(x_values) / n}
    best_preds = y_values

    # Grid search for (x_0, b, a, c)
    x_min, x_max = min(x_values), max(x_values)
    x0_candidates = [x_min + (x_max - x_min) * frac for frac in [0.25, 0.5, 0.75]]

    for x_0 in x0_candidates:
        for b_val in [0.2, 0.5, 1.0, 2.0]:
            for a_val in [1.0, 0.95]:
                for c_val in [0.0, 0.1, 0.2]:
                    preds = []
                    for x in x_values:
                        exponent = max(min(b_val * (x - x_0), 50.0), -50.0)
                        val = c_val + (a_val - c_val) / (1.0 + math.exp(exponent))
                        preds.append(val)
                    ss_res = sum((y - p) ** 2 for y, p in zip(y_values, preds))
                    if ss_res < best_res:
                        best_res = ss_res
                        best_params = {"a": a_val, "c": c_val, "b": b_val, "x_0": x_0}
                        best_preds = preds

    y_mean = sum(y_values) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r_squared = 1.0 - (best_res / ss_tot) if ss_tot > 0 else 1.0
    k = 4  # parameters (a, c, b, x_0)
    aic = n * math.log(max(best_res / n, 1e-12)) + 2 * k

    return CurveFitResult("sigmoid", best_params, r_squared, aic, best_preds)


def compare_curves(
    x_values: List[float],
    y_values: List[float],
) -> Dict[str, CurveFitResult]:
    """Fit linear, exponential, and sigmoid models and return all results."""
    return {
        "linear": fit_linear(x_values, y_values),
        "exponential": fit_exponential(x_values, y_values),
        "sigmoid": fit_sigmoid(x_values, y_values),
    }


def best_fitting_curve(
    x_values: List[float],
    y_values: List[float],
) -> Tuple[str, CurveFitResult]:
    """Select the best curve model according to AIC (lower is better)."""
    fits = compare_curves(x_values, y_values)
    best_name = min(fits, key=lambda k: fits[k].aic)
    return best_name, fits[best_name]

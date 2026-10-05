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
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


# Free parameters per candidate model; AICc is only defined for n > k + 1, so
# this count also decides which models `best_fitting_curve` may select.
MODEL_PARAM_COUNT: Dict[str, int] = {"linear": 2, "exponential": 3, "sigmoid": 4}

# AICc adds 2k(k+1)/(n-k-1). That denominator is 0 for the 4-parameter sigmoid
# on n=5 points, where the correction is undefined; it is floored so every
# reported AIC stays finite (SPEC §7 selects by R² and AIC, never inf).
AIC_DOF_FLOOR: int = 1

# Coarse grids seed a pattern search; on their own they returned the nearest
# grid point, so a generating rate of 0.15 came back as 0.2. 1/3 is the chance
# floor for the 3-choice answer space (SPEC §3), so the floor grids must reach it.
EXP_B_GRID: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)
SIGMOID_B_GRID: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
SIGMOID_X0_FRACTIONS: Tuple[float, ...] = (0.25, 0.5, 0.75)

# Bounds and initial step sizes for the pattern search. The midpoint may sit
# outside the sampled x range (the curve crosses tau there), so x_0 is searched
# across one full x-range either side.
SIGMOID_B_BOUNDS: Tuple[float, float] = (1e-3, 5.0)
EXP_B_BOUNDS: Tuple[float, float] = (1e-3, 2.0)
REFINE_ROUNDS: int = 6

# Accuracy is a probability in [0, 1], and its asymptote is the chance floor of the
# answer space, which is 1/3 for the 3-choice queries. An unconstrained offset
# solves to a negative "floor" on short noisy series, which is not a physical
# curve; these bounds keep the fit inside the accuracy range. Both belong in
# generator/constants.py next to FAILURE_THRESHOLD_TAU; they are duplicated here
# because analysis must not import generator (AGENTS.md §4 layering).
ACCURACY_BOUNDS: Tuple[float, float] = (0.0, 1.0)
CHANCE_FLOOR: float = 1.0 / 3.0
OFFSET_SOLVE_ROUNDS: int = 8

# The sigmoid asymptote is the chance floor, so it is capped at CHANCE_FLOOR; the
# exponential asymptote is a fitted constant, not the chance level, so it only
# has to stay inside [0, 1].
EXP_FLOOR_BOUNDS: Tuple[float, float] = (0.0, ACCURACY_BOUNDS[1])
SIGMOID_FLOOR_BOUNDS: Tuple[float, float] = (0.0, CHANCE_FLOOR)

# ss_res is floored before the log so a perfect fit yields a finite AIC
# instead of -inf (test_failsnow-adjacent: "a perfect fit produces a finite AIC").
SS_RES_FLOOR: float = 1e-12


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
    Returns the first x where accuracy drops below tau (per SPEC §7).
    """
    if len(x_values) != len(accuracies):
        raise ValueError("x_values and accuracies must have identical length")

    for x, acc in zip(x_values, accuracies):
        if acc < tau:
            return x
    
    return None


def _ss_total(y_values: Sequence[float]) -> float:
    """Total sum of squares about the mean."""
    n = len(y_values)
    y_mean = sum(y_values) / n
    return sum((y - y_mean) ** 2 for y in y_values)


def _r_squared(ss_res: float, ss_tot: float) -> float:
    """
    R² with the flat-series case made explicit.

    A constant series has ss_tot = 0, so the usual ratio is undefined. Reporting
    1.0 there claimed every model explained the data perfectly; 0.0 reports the
    truth, which is that a flat curve has no variance to explain.
    """
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _aic(ss_res: float, n: int, k: int) -> float:
    """
    AIC with the small-sample AICc correction, floored so it stays finite.

    The correction is 2k(k+1)/(n-k-1), which is undefined when n <= k + 1;
    AIC_DOF_FLOOR keeps the reported value finite, and `best_fitting_curve`
    excludes such models from selection instead.
    """
    aic = n * math.log(max(ss_res / n, SS_RES_FLOOR)) + 2 * k
    aic += 2 * k * (k + 1) / max(n - k - 1, AIC_DOF_FLOOR)
    return aic


def _aicc_is_defined(n: int, k: int) -> bool:
    """AICc needs n - k - 1 > 0; otherwise the model has no spare degrees of freedom."""
    return n - k - 1 > 0


def _bounded_offsets(
    basis_a: Sequence[float],
    basis_b: Sequence[float],
    y_values: Sequence[float],
    floor_bounds: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Solve y ≈ p * basis_a + q * basis_b under physical bounds on (p, q).

    Both coefficients are offsets of an accuracy curve, so `q` (the chance
    floor) stays inside `floor_bounds` and `p` stays between `q` and 1. Plain
    least squares ignores those bounds and answers a negative "chance floor" on
    short series, which is not a curve any model can produce. Coordinate
    descent on a two-variable box-constrained convex problem reaches the same
    optimum as a proper solver, with no new dependency.

    `floor_bounds[1]` may be None to leave the floor's upper bound open; the
    sigmoid floor is capped at the chance floor, the exponential floor is not,
    because the exponential's asymptote is not the chance level by construction.
    """
    floor_low, floor_high = floor_bounds
    amplitude = ACCURACY_BOUNDS[1]
    floor_value = min(floor_low, amplitude)
    if floor_high is not None:
        floor_value = max(floor_low, min(floor_high, floor_value))

    for _ in range(OFFSET_SOLVE_ROUNDS):
        denom_a = sum(u * u for u in basis_a)
        if denom_a == 0.0:
            return None
        amplitude = sum((y - floor_value * v) * u for y, u, v in zip(y_values, basis_a, basis_b)) / denom_a
        amplitude = max(floor_value, min(ACCURACY_BOUNDS[1], amplitude))

        denom_b = sum(v * v for v in basis_b)
        if denom_b == 0.0:
            return None
        floor_value = sum(y - amplitude * u for y, u in zip(y_values, basis_a)) / denom_b
        if floor_high is not None:
            floor_value = max(floor_low, min(floor_high, floor_value))
        floor_value = min(floor_value, amplitude)

    return amplitude, floor_value


def _pattern_search(
    objective: Callable[[Dict[str, float]], float],
    start: Dict[str, float],
    steps: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    rounds: int = REFINE_ROUNDS,
) -> Tuple[Dict[str, float], float]:
    """
    Deterministic coordinate pattern search.

    Coarse grids seed this because a grid alone can only return one of its own
    points, which biases the fitted shape parameters. Keys are visited in sorted
    order so the result depends on nothing but the inputs (AGENTS.md §6.5).
    """
    best = dict(start)
    best_value = objective(best)
    steps = dict(steps)

    for _ in range(rounds):
        improved = False
        for name in sorted(steps):
            for direction in (1.0, -1.0):
                low, high = bounds[name]
                candidate = best[name] + direction * steps[name]
                candidate = min(max(candidate, low), high)
                if candidate == best[name]:
                    continue
                trial = dict(best)
                trial[name] = candidate
                value = objective(trial)
                if value < best_value:
                    best, best_value = trial, value
                    improved = True
        if not improved:
            steps = {name: step / 2.0 for name, step in steps.items()}
            if max(steps.values()) < 1e-4:
                break

    return best, best_value


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
    ss_tot = _ss_total(y_values)
    ss_res = sum((y - pred) ** 2 for y, pred in zip(y_values, preds))

    return CurveFitResult(
        "linear",
        {"a": a, "b": b},
        _r_squared(ss_res, ss_tot),
        _aic(ss_res, n, MODEL_PARAM_COUNT["linear"]),
        preds,
    )


def fit_exponential(
    x_values: List[float],
    y_values: List[float],
) -> CurveFitResult:
    """
    Fit exponential decay: A(x) = a * exp(-b*x) + c.

    Given (b, c) the model is linear in a, so a is solved in closed form and the
    search runs over (b, c) only. c ranges up to the 1/3 chance floor of the
    3-choice answer space, not to 0.25, so a floor at 1/3 is reachable.
    """
    n = len(x_values)
    if n < 3:
        return fit_linear(x_values, y_values)

    def solve_offsets(b_val: float) -> Optional[Tuple[float, float]]:
        """Amplitude and floor are linear given the decay rate, so they are solved."""
        return _bounded_offsets(
            [math.exp(-b_val * x) for x in x_values],
            [1.0] * n,
            y_values,
            EXP_FLOOR_BOUNDS,
        )

    def objective(params: Dict[str, float]) -> float:
        solved = solve_offsets(params["b"])
        if solved is None:
            return float("inf")
        a_val, c_val = solved
        terms = [math.exp(-params["b"] * x) for x in x_values]
        return sum((y - (a_val * t + c_val)) ** 2 for y, t in zip(y_values, terms))

    best_res = float("inf")
    best_b = EXP_B_GRID[0]
    for b_val in EXP_B_GRID:
        res = objective({"b": b_val})
        if res < best_res:
            best_res, best_b = res, b_val

    x_span = max(x_values) - min(x_values) or 1.0
    best_shape, best_res = _pattern_search(
        objective,
        {"b": best_b},
        {"b": x_span / 8.0},
        {"b": EXP_B_BOUNDS},
    )
    best_b = best_shape["b"]

    solved = solve_offsets(best_b)
    best_amplitude, best_floor = solved if solved is not None else (1.0, 0.0)
    best_preds = [best_amplitude * math.exp(-best_b * x) + best_floor for x in x_values]

    params = {
        "a": best_amplitude,
        "b": best_b,
        "c": best_floor,
    }

    return CurveFitResult(
        "exponential",
        params,
        _r_squared(best_res, _ss_total(y_values)),
        _aic(best_res, n, MODEL_PARAM_COUNT["exponential"]),
        best_preds,
    )


def fit_sigmoid(
    x_values: List[float],
    y_values: List[float],
) -> CurveFitResult:
    """
    Fit sigmoid: A(x) = c + (a - c) / (1 + exp(b * (x - x_0)))

    Rewriting as a * s + c * (1 - s) with s = 1/(1 + exp(b*(x - x_0))) shows the
    model is linear in (a, c) once (b, x_0) are fixed, so only the shape
    parameters are searched. c reaches 1/3, the chance floor for the 3-choice
    answer space.
    """
    n = len(x_values)
    if n < 4:
        return fit_linear(x_values, y_values)

    def shape_basis(b_val: float, x_0: float) -> List[float]:
        """s(x) = 1/(1+exp(b*(x-x_0))); A(x) = a*s + c*(1-s)."""
        return [
            1.0 / (1.0 + math.exp(max(min(b_val * (x - x_0), 50.0), -50.0)))
            for x in x_values
        ]

    def objective(params: Dict[str, float]) -> float:
        basis = shape_basis(params["b"], params["x_0"])
        solved = _bounded_offsets(
            basis,
            [1.0 - s for s in basis],
            y_values,
            SIGMOID_FLOOR_BOUNDS,
        )
        if solved is None:
            return float("inf")
        a_val, c_val = solved
        return sum(
            (y - (a_val * s + c_val * (1.0 - s))) ** 2 for y, s in zip(y_values, basis)
        )

    x_min, x_max = min(x_values), max(x_values)
    x_span = (x_max - x_min) or 1.0
    x0_candidates = [x_min + x_span * frac for frac in SIGMOID_X0_FRACTIONS]

    best_res = float("inf")
    best_params = {"b": SIGMOID_B_GRID[0], "x_0": x0_candidates[0]}
    for b_val in SIGMOID_B_GRID:
        for x_0 in x0_candidates:
            candidate = {"b": b_val, "x_0": x_0}
            res = objective(candidate)
            if res < best_res:
                best_res, best_params = res, candidate

    best_params, best_res = _pattern_search(
        objective,
        best_params,
        {"b": x_span / 8.0, "x_0": x_span / 4.0},
        {"b": SIGMOID_B_BOUNDS, "x_0": (x_min - x_span, x_max + x_span)},
    )

    basis = shape_basis(best_params["b"], best_params["x_0"])
    a_val, c_val = _bounded_offsets(
        basis,
        [1.0 - s for s in basis],
        y_values,
        SIGMOID_FLOOR_BOUNDS,
    ) or (1.0, 0.0)
    best_preds = [a_val * s + c_val * (1.0 - s) for s in basis]

    return CurveFitResult(
        "sigmoid",
        {"a": a_val, "c": c_val, "b": best_params["b"], "x_0": best_params["x_0"]},
        _r_squared(best_res, _ss_total(y_values)),
        _aic(best_res, n, MODEL_PARAM_COUNT["sigmoid"]),
        best_preds,
    )


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
    """
    Select the best curve model by AICc (lower is better).

    Models whose AICc is undefined (n <= k + 1, no spare degrees of freedom) are
    excluded: with n = 5 the 4-parameter sigmoid can interpolate any series,
    including noise, so it would otherwise win on residual sum of squares alone.
    When every model is excluded the lowest raw AIC still decides, so a
    selection is always returned.
    """
    fits = compare_curves(x_values, y_values)
    n = len(x_values)

    eligible = {
        name: fit
        for name, fit in fits.items()
        if _aicc_is_defined(n, MODEL_PARAM_COUNT[fit.model_type])
    }
    if not eligible:
        eligible = fits

    best_name = min(eligible, key=lambda name: eligible[name].aic)
    return best_name, eligible[best_name]

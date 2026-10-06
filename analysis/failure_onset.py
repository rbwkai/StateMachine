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
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .statistics import wilson_interval


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

# CHANCE_FLOOR is only the default: callers fitting cells with a different
# answer space (count queries with 0..E+splits, split_chain's {1, 2}, more
# containers) pass `chance_floor` to fit_sigmoid / compare_curves /
# best_fitting_curve, typically the cell's mean eval.baselines.chance_level.
# The sigmoid asymptote is the chance floor, so it is capped at it; the
# exponential asymptote is a fitted constant, not the chance level, so it only
# has to stay inside [0, 1].
EXP_FLOOR_BOUNDS: Tuple[float, float] = (0.0, ACCURACY_BOUNDS[1])
SIGMOID_FLOOR_BOUNDS: Tuple[float, float] = (0.0, CHANCE_FLOOR)

# ss_res is floored before the log so a perfect fit yields a finite AIC
# instead of -inf (test_failsnow-adjacent: "a perfect fit produces a finite AIC").
SS_RES_FLOOR: float = 1e-12

# Fewer distinct difficulty levels than this cannot discriminate the three
# candidate shapes (the sigmoid alone has 4 parameters), so `best_fitting_curve`
# declines to select when called with min_levels=MIN_CURVE_LEVELS.
MIN_CURVE_LEVELS: int = 6

# Logistic-slope estimation: IRLS iteration cap and convergence tolerance, and a
# ridge term that keeps the Hessian invertible on separable bootstrap resamples.
LOGIT_MAX_ITER: int = 50
LOGIT_TOL: float = 1e-8
LOGIT_RIDGE: float = 1e-6


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


def _per_point(
    value: Union[None, float, int, Sequence[Any]],
    n: int,
    name: str,
) -> List[Any]:
    """Broadcast a scalar (or None) to n points, or check a per-point sequence."""
    if value is None or isinstance(value, (int, float)):
        return [value] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"{name} must be a scalar or have one entry per x value")
    return values


def compute_failure_onset(
    x_values: Sequence[Union[int, float]],
    accuracies: Sequence[float],
    tau: float = 0.70,
    *,
    sustained: bool = False,
    n_trials: Union[None, int, Sequence[int]] = None,
    chance: Union[None, float, Sequence[float]] = None,
) -> Optional[Union[int, float]]:
    """
    Compute failure onset L_f = min { x : A(x) < tau } (SPEC §7).

    Points are sorted by x internally, so callers need not pre-sort. The
    comparison is strict: A(x) == tau is not a failure.

    sustained: when True, return the smallest x such that x and every larger
        x are below tau (a dip followed by recovery is not an onset).
    n_trials: number of trials behind each accuracy (scalar or per point).
        When given, a point counts as below tau only if the upper bound of its
        two-sided 95% Wilson interval is below tau, i.e. the drop is not
        explained by sampling noise. successes = round(acc * n_trials).
    chance: chance accuracy (scalar or per point, each < 1). When given,
        accuracy is chance-normalised as (acc - c) / (1 - c) before the
        comparison, so tau is read as a fraction of the above-chance range;
        with n_trials the same transform is applied to the Wilson upper bound
        (the transform is monotone, so the order of the two steps is immaterial).

    Returns None when no point qualifies or the input is empty.
    """
    if len(x_values) != len(accuracies):
        raise ValueError("x_values and accuracies must have identical length")
    n = len(x_values)
    if n < 1:
        return None

    trials = _per_point(n_trials, n, "n_trials")
    chances = _per_point(chance, n, "chance")

    points = sorted(
        zip(x_values, accuracies, trials, chances), key=lambda point: point[0]
    )

    def is_below(acc: float, n_t: Optional[int], c: Optional[float]) -> bool:
        value = float(acc)
        if n_t is not None:
            n_t = int(n_t)
            if n_t <= 0:
                raise ValueError("n_trials must be positive")
            successes = min(n_t, max(0, int(round(value * n_t))))
            value = wilson_interval(successes, n_t)[1]
        if c is not None:
            if not c < 1.0:
                raise ValueError("chance must be below 1")
            value = (value - c) / (1.0 - c)
        return value < tau

    below = [is_below(acc, n_t, c) for _, acc, n_t, c in points]

    if not sustained:
        for (x, _, _, _), flag in zip(points, below):
            if flag:
                return x
        return None

    onset: Optional[Union[int, float]] = None
    for (x, _, _, _), flag in zip(reversed(points), reversed(below)):
        if not flag:
            break
        onset = x
    return onset


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
) -> Optional[CurveFitResult]:
    """
    Fit linear model: A(x) = a + b * x
    """
    n = len(x_values)
    if n < 2:
        # A line through fewer than two points is undefined and its AIC would
        # be infinite; report no fit rather than a non-finite result.
        return None

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
) -> Optional[CurveFitResult]:
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
    chance_floor: float = CHANCE_FLOOR,
) -> Optional[CurveFitResult]:
    """
    Fit sigmoid: A(x) = c + (a - c) / (1 + exp(b * (x - x_0)))

    Rewriting as a * s + c * (1 - s) with s = 1/(1 + exp(b*(x - x_0))) shows the
    model is linear in (a, c) once (b, x_0) are fixed, so only the shape
    parameters are searched. c is capped at `chance_floor` (default 1/3, the
    3-choice location answer space); pass the cell's uniform chance level when
    its answer space differs.
    """
    if not 0.0 <= chance_floor <= ACCURACY_BOUNDS[1]:
        raise ValueError("chance_floor must lie in [0, 1]")
    floor_bounds = (SIGMOID_FLOOR_BOUNDS[0], chance_floor)
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
            floor_bounds,
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
        floor_bounds,
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
    chance_floor: float = CHANCE_FLOOR,
) -> Dict[str, CurveFitResult]:
    """
    Fit linear, exponential, and sigmoid models and return all results.

    A model with no defined fit (fewer than two points) is omitted, so the
    result is empty for n < 2.
    """
    fits = {
        "linear": fit_linear(x_values, y_values),
        "exponential": fit_exponential(x_values, y_values),
        "sigmoid": fit_sigmoid(x_values, y_values, chance_floor=chance_floor),
    }
    return {name: fit for name, fit in fits.items() if fit is not None}


def best_fitting_curve(
    x_values: List[float],
    y_values: List[float],
    min_levels: Optional[int] = None,
    chance_floor: float = CHANCE_FLOOR,
) -> Optional[Tuple[str, CurveFitResult]]:
    """
    Select the best curve model by AICc (lower is better).

    Models whose AICc is undefined (n <= k + 1, no spare degrees of freedom) are
    excluded: with n = 5 the 4-parameter sigmoid can interpolate any series,
    including noise, so it would otherwise win on residual sum of squares alone.
    When every model is excluded the lowest raw AIC still decides.

    min_levels: when given (normally MIN_CURVE_LEVELS), return None if the data
        has fewer distinct x levels than this, because shape selection on so few
        levels defaults to "linear" regardless of the true shape. It is opt-in
        so existing callers that index the result keep working.

    Returns None when no model can be fitted (n < 2) or min_levels is not met.
    """
    if min_levels is not None and len(set(x_values)) < min_levels:
        return None

    fits = compare_curves(x_values, y_values, chance_floor=chance_floor)
    if not fits:
        return None
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


# ==== Logistic slope with cluster bootstrap ====


@dataclass
class LogitSlopeResult:
    """Slope of logit P(correct) on x, with a cluster-bootstrap percentile CI."""
    slope: float
    intercept: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_boot_failed: int
    n_clusters: int
    # Resamples dropped because the classes were (quasi-)completely separated
    # in x; also counted in n_boot_failed.
    n_boot_separated: int = 0
    # The full-data fit is (quasi-)separable: the MLE does not exist and
    # `slope` is set by the eta clamp and ridge, not by the data.
    separable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_boot": self.n_boot,
            "n_boot_failed": self.n_boot_failed,
            "n_clusters": self.n_clusters,
            "n_boot_separated": self.n_boot_separated,
            "separable": self.separable,
        }


def _is_separable(x: Sequence[float], y: Sequence[float]) -> bool:
    """
    True when the logistic MLE does not exist because x separates the classes.

    Complete or quasi-complete separation: max x of one class <= min x of the
    other (sharing only the boundary value is quasi-complete). A single-class
    sample is also separated (the intercept diverges). On such data IRLS walks
    until the eta clamp and ridge stop it, e.g. ([1,1,2,2], [1,1,0,0]) gave a
    slope of -20.7, which is an artefact, not an estimate.
    """
    ones = [xi for xi, yi in zip(x, y) if yi >= 0.5]
    zeros = [xi for xi, yi in zip(x, y) if yi < 0.5]
    if not ones or not zeros:
        return True
    return max(ones) <= min(zeros) or max(zeros) <= min(ones)


def _logit_irls(
    x: Sequence[float],
    y: Sequence[float],
) -> Optional[Tuple[float, float]]:
    """
    Fit logit P(y=1) = b0 + b1 * x by iteratively reweighted least squares.

    Pure Python (the system is 2x2) so `analysis` keeps importing without numpy;
    generator.probes imports this package. A tiny ridge term keeps the step
    defined on (quasi-)separable data, where the estimate grows large but stays
    finite. Returns None if x has no spread.
    """
    if not x or max(x) == min(x):
        return None
    b0 = b1 = 0.0
    for _ in range(LOGIT_MAX_ITER):
        # Ridge on the diagonal only (penalised log-likelihood).
        h00, h01, h11 = LOGIT_RIDGE, 0.0, LOGIT_RIDGE
        g0 = -LOGIT_RIDGE * b0
        g1 = -LOGIT_RIDGE * b1
        for xi, yi in zip(x, y):
            eta = max(-30.0, min(30.0, b0 + b1 * xi))
            p = 1.0 / (1.0 + math.exp(-eta))
            w = p * (1.0 - p)
            h00 += w
            h01 += w * xi
            h11 += w * xi * xi
            r = yi - p
            g0 += r
            g1 += r * xi
        det = h00 * h11 - h01 * h01
        if det <= 0.0:
            return None
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (h00 * g1 - h01 * g0) / det
        b0 += step0
        b1 += step1
        if max(abs(step0), abs(step1)) < LOGIT_TOL:
            break
    return b0, b1


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default method) on sorted data."""
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def slope_logit(
    xs: Sequence[float],
    ys_binary: Sequence[Union[bool, int]],
    clusters: Optional[Sequence[Any]] = None,
    n_boot: int = 1000,
    rng: Optional[random.Random] = None,
    confidence: float = 0.95,
) -> Optional[LogitSlopeResult]:
    """
    Logistic-regression slope of correctness on x with a cluster-bootstrap CI.

    clusters: one label per observation (e.g. instance template or seed group);
        the bootstrap resamples whole clusters with replacement, so correlated
        items sharing a cluster are not treated as independent. None means each
        observation is its own cluster.
    rng: explicit random.Random (AGENTS.md §6 rule 5); defaults to
        random.Random(0) so the CI is byte-for-byte reproducible.

    Resamples whose x has no spread cannot identify a slope and are counted in
    n_boot_failed. Resamples that are (quasi-)completely separated have no
    finite MLE; their clamp-driven slopes are excluded from the CI and counted
    in both n_boot_failed and n_boot_separated. The CI is the percentile
    interval of the successful resamples (NaN if none succeed). When the
    full-data fit is itself separable, `separable` is True and the reported
    slope is not a usable estimate. Returns None if the full-data fit is
    undefined (fewer than two distinct x values).
    """
    if len(xs) != len(ys_binary):
        raise ValueError("xs and ys_binary must have identical length")
    if clusters is not None and len(clusters) != len(xs):
        raise ValueError("clusters must have one label per observation")
    if rng is None:
        rng = random.Random(0)

    x = [float(v) for v in xs]
    y = [1.0 if bool(v) else 0.0 for v in ys_binary]
    point = _logit_irls(x, y)
    if point is None:
        return None

    labels = list(range(len(xs))) if clusters is None else list(clusters)
    # First-appearance order, not set order, so resampling is deterministic.
    members: Dict[Any, List[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(label, []).append(index)
    groups = list(members.values())

    slopes: List[float] = []
    failed = 0
    separated = 0
    for _ in range(n_boot):
        picked: List[int] = []
        for _ in range(len(groups)):
            picked.extend(groups[rng.randrange(len(groups))])
        bx = [x[i] for i in picked]
        by = [y[i] for i in picked]
        # Spread is checked first so a no-spread resample is not double-booked
        # as separated; the RNG draws above are unchanged either way.
        fit = _logit_irls(bx, by)
        if fit is None:
            failed += 1
        elif _is_separable(bx, by):
            failed += 1
            separated += 1
        else:
            slopes.append(fit[1])

    if slopes:
        alpha = (1.0 - confidence) / 2.0
        slopes.sort()
        ci_low = _quantile(slopes, alpha)
        ci_high = _quantile(slopes, 1.0 - alpha)
    else:
        ci_low = ci_high = float("nan")

    return LogitSlopeResult(
        slope=point[1],
        intercept=point[0],
        ci_low=ci_low,
        ci_high=ci_high,
        n_boot=n_boot,
        n_boot_failed=failed,
        n_clusters=len(groups),
        n_boot_separated=separated,
        separable=_is_separable(x, y),
    )

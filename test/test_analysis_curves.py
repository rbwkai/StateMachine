"""Checklist 10: analysis.

Covers curve fitting (recovery of known shapes, small-n fallbacks, constant
series, perfect fits, AIC model selection), failure-onset semantics, and the
missing statistical helpers (confidence intervals, McNemar, small-cell
reporting). Real defects are documented as plain failing `test_failsnow_*` tests.
"""
from __future__ import annotations

import collections
import math
import random

import pytest

from analysis.failure_onset import (
    best_fitting_curve,
    compare_curves,
    compute_failure_onset,
    fit_exponential,
    fit_linear,
    fit_sigmoid,
)


XS = [2.0, 4.0, 6.0, 8.0, 12.0]
LINEAR = [0.9 - 0.05 * x for x in XS]
EXPONENTIAL = [0.9 * math.exp(-0.15 * x) for x in XS]
SIGMOID = [1.0 / (1.0 + math.exp(0.5 * (x - 8.0))) for x in XS]


# ---------------------------------------------------------------------------
# Curve fitting.
# ---------------------------------------------------------------------------

def test_linear_fit_recovers_the_exact_parameters():
    fit = fit_linear(XS, LINEAR)
    assert fit.model_type == "linear"
    assert fit.params["a"] == pytest.approx(0.9)
    assert fit.params["b"] == pytest.approx(-0.05)
    assert fit.r_squared == pytest.approx(1.0)


def test_exponential_fit_tracks_a_known_exponential():
    fit = fit_exponential(XS, EXPONENTIAL)
    assert fit.model_type == "exponential"
    assert fit.r_squared > 0.95
    assert min(fit.predictions) == pytest.approx(min(EXPONENTIAL), abs=0.15)


def test_sigmoid_fit_tracks_a_known_sigmoid():
    fit = fit_sigmoid(XS, SIGMOID)
    assert fit.model_type == "sigmoid"
    assert fit.r_squared > 0.9
    assert max(fit.predictions) > min(fit.predictions)
    assert all(0.0 <= value <= 1.0 for value in fit.predictions)


def test_all_three_models_are_produced_for_a_full_size_series():
    fits = compare_curves(XS, LINEAR)
    assert set(fits) == {"linear", "exponential", "sigmoid"}
    for fit in fits.values():
        assert len(fit.predictions) == len(XS)
        assert math.isfinite(fit.aic)


@pytest.mark.parametrize("fit_fn,series", [
    (fit_exponential, [0.5, 0.4]),
    (fit_sigmoid, [0.5, 0.4]),
    (fit_sigmoid, [0.5, 0.4, 0.3]),
])
def test_small_series_fall_back_to_the_linear_model(fit_fn, series):
    """n<3 falls back for the exponential fit, n<4 for the sigmoid fit."""
    assert fit_fn([1.0, 2.0, 3.0][: len(series)], series).model_type == "linear"


def test_a_perfect_fit_produces_a_finite_aic():
    """`ss_res=0` must not send AIC to -inf."""
    fits = compare_curves(XS, LINEAR)
    assert all(math.isfinite(fit.aic) for fit in fits.values())
    assert fits["linear"].r_squared == pytest.approx(1.0)
    assert all(fit.r_squared > 0.9 for fit in fits.values())


def test_four_point_series_still_produces_every_model():
    xs = [4.0, 8.0, 12.0, 16.0]
    fits = compare_curves(xs, [0.95 - 0.04 * x for x in xs])
    assert set(fits) == {"linear", "exponential", "sigmoid"}
    assert all(len(fit.predictions) == 4 for fit in fits.values())


def test_aic_prefers_the_linear_model_on_clean_linear_data():
    name, fit = best_fitting_curve(XS, LINEAR)
    assert name == "linear"
    assert fit.model_type == "linear"


def test_failsnow_model_selection_stays_linear_under_small_noise():
    """[checklist 10] 'AIC prefers the simpler model on clean linear data.'

    [fails now] expected: AIC keeps picking `linear` for a linear series with
    realistic binomial noise at n=5. currently the 4-parameter sigmoid can
    interpolate the noise, so it wins about a third of the draws.
    """
    rng = random.Random(0)
    chosen = collections.Counter()
    for _ in range(200):
        noisy = [min(1.0, max(0.0, 0.95 - 0.04 * x + rng.gauss(0, 0.03))) for x in XS]
        chosen[best_fitting_curve(XS, noisy)[0]] += 1
    assert chosen["sigmoid"] == 0, dict(chosen)


def test_failsnow_constant_accuracy_is_not_a_perfect_fit():
    """[checklist 10] 'handles constant accuracy.'

    [fails now] expected: a flat series is reported as degenerate, not as a
    perfect fit. currently every model returns `r_squared=1.0` for constant
    input, so `best_fitting_curve` happily 'explains' noise-free data.
    """
    fits = compare_curves(XS, [0.5] * len(XS))
    for name, fit in fits.items():
        assert fit.r_squared < 1.0, f"{name} reports r_squared=1.0 for a flat series"


def test_failsnow_the_chance_floor_of_one_third_is_reachable():
    """[checklist 10] curve fitting against the real chance floor.

    [fails now] expected: a curve with an asymptote at 1/3 can be fitted, since
    that is the actual chance floor for these answer spaces. currently the grids
    cap the offset at 0.2 (sigmoid) and 0.25 (exponential), so a 1/3 floor is
    unreachable and the fitted parameters are biased.
    """
    xs = [2.0, 4.0, 6.0, 8.0, 12.0]
    sigmoid_truth = [0.33 + (0.98 - 0.33) / (1 + math.exp(0.6 * (x - 8.0))) for x in xs]
    fit = fit_sigmoid(xs, sigmoid_truth)
    assert fit.params["c"] == pytest.approx(0.33, abs=0.05), fit.params

    exponential_truth = [0.33 + (0.98 - 0.33) * math.exp(-0.2 * x) for x in xs]
    exp_fit = fit_exponential(xs, exponential_truth)
    assert exp_fit.params["c"] == pytest.approx(0.33, abs=0.05), exp_fit.params


def test_failsnow_fitted_parameters_match_the_generating_values():
    """[checklist 10] 'recovers known linear, exponential and sigmoid curves.'

    [fails now] expected: the recovered parameters equal the generating ones.
    currently the grid search returns the nearest grid point: the exponential
    decay rate comes back as 0.2 instead of 0.15, and the sigmoid midpoint as
    7.0 instead of 8.0.
    """
    assert fit_exponential(XS, EXPONENTIAL).params["b"] == pytest.approx(0.15, abs=0.02)
    assert fit_sigmoid(XS, SIGMOID).params["x_0"] == pytest.approx(8.0, abs=1.0)


# ---------------------------------------------------------------------------
# Failure onset.
# ---------------------------------------------------------------------------

def test_onset_is_the_first_depth_below_tau():
    assert compute_failure_onset([2, 4, 6, 8, 12], [0.95, 0.9, 0.8, 0.6, 0.4]) == 8


def test_onset_on_an_immediate_dip_is_the_first_depth():
    assert compute_failure_onset([2, 4, 6], [0.1, 0.9, 0.95]) == 2


def test_a_never_failing_curve_returns_none():
    assert compute_failure_onset([2, 4, 6], [0.95, 0.9, 0.75], tau=0.7) is None


def test_accuracy_exactly_equal_to_tau_is_not_a_failure():
    """The contract is strict: onset is min { x : A(x) < tau } (SPEC §7)."""
    assert compute_failure_onset([2, 4, 6], [0.9, 0.7, 0.7], tau=0.7) is None
    assert compute_failure_onset([2, 4, 6], [0.9, 0.69, 0.7], tau=0.7) == 4


def test_a_non_monotone_curve_reports_only_the_first_dip():
    """Recovery after the dip is ignored by design: onset is the first crossing."""
    assert compute_failure_onset([3, 4, 6, 8, 12], [0.95, 0.40, 0.92, 0.90, 0.35]) == 4


def test_onset_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_failure_onset([1, 2, 3], [0.5, 0.4])


# ---------------------------------------------------------------------------
# Statistics.
# ---------------------------------------------------------------------------

def _helper(name):
    import analysis

    return getattr(analysis, name, None)


def test_failsnow_confidence_intervals_are_reported():
    """[checklist 10] 'confidence intervals (n=50 is about +-14 points).'

    [fails now] expected: the analysis package exposes a binomial interval so a
    cell accuracy comes with an error bar. currently no such helper exists, so
    every reported accuracy is a bare point estimate.
    """
    interval = _helper("wilson_interval") or _helper("binomial_interval")
    assert interval is not None, "analysis exposes no confidence-interval helper"
    low, high = interval(25, 50)
    assert high - low == pytest.approx(0.28, abs=0.03)


def test_failsnow_mcnemar_test_is_available_for_direct_versus_cot():
    """[checklist 10] 'paired tests (McNemar) for direct vs CoT.'"""
    mcnemar = _helper("mcnemar_test")
    assert mcnemar is not None, "analysis exposes no McNemar helper"
    # 30 discordant pairs, all in one direction: p is tiny.
    result = mcnemar(b=30, c=0)
    assert result["p_value"] < 0.001


def test_failsnow_cells_with_fewer_than_fifty_successes_are_flagged():
    """[checklist 10] small-cell reporting.

    The literal "fewer than 50 successes" rule flagged every cell below 100% at
    n=50; the rule is now minority count < 10 or Wilson half-width > 0.12.
    """
    reporter = _helper("flag_low_success_cells")
    assert reporter is not None, "analysis exposes no low-success reporter"
    flagged = reporter([
        {"condition": "basic_chain_T4", "correct": 45, "total": 50},
        {"condition": "basic_chain_T8", "correct": 12, "total": 50},
    ])
    assert [cell["condition"] for cell in flagged] == ["basic_chain_T4"]

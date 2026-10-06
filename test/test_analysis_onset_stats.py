"""Failure onset, first-error MISSING handling, logistic slope and paired statistics.

Covers the fixes to analysis/first_error.py (unparsed steps are not errors),
analysis/failure_onset.py (sorting, sustained onset, Wilson and chance-aware
comparisons, small-n curve fits, opt-in level gate, logistic slope) and
analysis/statistics.py (small-cell rule, paired McNemar, Holm).
"""
from __future__ import annotations

import math
import random

import pytest

import analysis
from analysis.failure_onset import (
    MIN_CURVE_LEVELS,
    best_fitting_curve,
    compare_curves,
    compute_failure_onset,
    fit_exponential,
    fit_linear,
    fit_sigmoid,
    slope_logit,
)
from analysis.first_error import ErrorType, analyze_first_error
from analysis.statistics import (
    flag_low_success_cells,
    holm_correction,
    mcnemar_paired,
    wilson_interval,
)


GOLD = ["A", "B", "C", "D", "A"]


# ==== first_error: unparsed steps ====


def test_none_step_is_missing_not_an_error():
    result = analyze_first_error(GOLD, ["B", None, "D", "A"])
    assert result.step_errors == []
    assert result.first_error_step is None
    assert result.n_missing == 1
    assert result.missing_steps == [2]
    assert result.error_type is ErrorType.MISSING
    assert result.final_is_correct is True


def test_onset_is_the_first_non_none_wrong_step():
    result = analyze_first_error(GOLD, [None, None, "X", "Z"])
    assert result.first_error_step == 3
    assert result.step_errors == [3, 4]
    assert result.n_missing == 2
    assert result.error_type is ErrorType.PROPAGATING_ERROR


def test_missing_step_is_not_evidence_of_recovery():
    result = analyze_first_error(GOLD, ["X", None, None, "Z"])
    assert result.error_type is ErrorType.PROPAGATING_ERROR
    recovered = analyze_first_error(GOLD, ["X", None, "D", "Z"])
    assert recovered.error_type is ErrorType.LOCAL_ERROR


def test_short_prediction_is_padded_as_missing_tail():
    result = analyze_first_error(GOLD, ["B", "X"])
    assert result.total_steps == 4
    assert result.missing_steps == [3, 4]
    assert result.first_error_step == 2
    assert result.pred_final is None
    assert result.final_is_correct is False
    assert result.error_type is ErrorType.PROPAGATING_ERROR


def test_all_missing_prediction_is_missing():
    result = analyze_first_error(GOLD, [])
    assert result.error_type is ErrorType.MISSING
    assert result.n_missing == 4
    assert result.final_is_correct is False


def test_long_prediction_still_raises():
    with pytest.raises(ValueError):
        analyze_first_error(GOLD, ["B", "C", "D", "A", "B"])


def test_to_dict_reports_missing_fields_and_keeps_old_keys():
    payload = analyze_first_error(GOLD, ["B", None, "D", "A"]).to_dict()
    for key in ("error_type", "first_error_step", "total_steps", "step_errors",
                "gold_final", "pred_final", "final_is_correct"):
        assert key in payload
    assert payload["n_missing"] == 1
    assert payload["missing_steps"] == [2]
    assert payload["error_type"] == "MISSING"


def test_fully_parsed_trajectory_is_unchanged():
    assert analyze_first_error(GOLD, ["B", "C", "D", "A"]).error_type is ErrorType.NO_ERROR
    assert analyze_first_error(GOLD, ["B", "C", "D", "Z"]).error_type is ErrorType.FINAL_ONLY_ERROR
    assert analyze_first_error(GOLD, ["B", "X", "D", "A"]).error_type is ErrorType.CANCELLATION_ERROR


def test_l1_final_only_error_requires_observed_intermediates():
    # Regression (L1): only the final step is an observed error, but step 2 is
    # unparsed, so "all intermediate states correct" is unverified.
    result = analyze_first_error(GOLD, ["B", None, "D", "Z"])
    assert result.first_error_step == 4
    assert result.has_missing_before_first_error is True
    assert result.error_type is not ErrorType.FINAL_ONLY_ERROR
    assert result.error_type is ErrorType.PROPAGATING_ERROR
    assert result.to_dict()["has_missing_before_first_error"] is True
    clean = analyze_first_error(GOLD, ["B", "C", "D", "Z"])
    assert clean.error_type is ErrorType.FINAL_ONLY_ERROR
    assert clean.has_missing_before_first_error is False


# ==== compute_failure_onset ====


def test_onset_sorts_unsorted_input():
    assert compute_failure_onset([8, 2, 4], [0.5, 0.9, 0.6]) == 4


def test_onset_empty_input_is_none():
    assert compute_failure_onset([], []) is None


def test_sustained_onset_ignores_a_recovered_dip():
    xs = [3, 4, 6, 8, 12]
    acc = [0.95, 0.40, 0.92, 0.60, 0.35]
    assert compute_failure_onset(xs, acc) == 4
    assert compute_failure_onset(xs, acc, sustained=True) == 8


def test_sustained_onset_is_none_when_last_point_passes():
    assert compute_failure_onset([2, 4, 6], [0.5, 0.5, 0.9], sustained=True) is None


def test_sustained_onset_sorts_too():
    assert compute_failure_onset([12, 2, 8, 4], [0.3, 0.9, 0.5, 0.5], sustained=True) == 4


def test_n_trials_requires_wilson_upper_bound_below_tau():
    # 0.6 at n=10 is within noise of 0.7; at n=1000 it is not.
    assert compute_failure_onset([2, 4], [0.9, 0.6], n_trials=10) is None
    assert compute_failure_onset([2, 4], [0.9, 0.6], n_trials=1000) == 4
    assert compute_failure_onset([2, 4], [0.9, 0.6], n_trials=[1000, 10]) is None
    assert wilson_interval(6, 10)[1] > 0.7


def test_chance_normalisation_is_applied_before_comparison():
    # (0.75 - 1/3) / (2/3) = 0.625 < 0.7
    assert compute_failure_onset([2, 4], [0.95, 0.75]) is None
    assert compute_failure_onset([2, 4], [0.95, 0.75], chance=1.0 / 3.0) == 4
    assert compute_failure_onset([2, 4], [0.95, 0.75], chance=[0.0, 0.0]) is None


def test_onset_rejects_bad_per_point_lengths_and_chance():
    with pytest.raises(ValueError):
        compute_failure_onset([2, 4], [0.9, 0.6], n_trials=[10])
    with pytest.raises(ValueError):
        compute_failure_onset([2, 4], [0.9, 0.6], chance=1.0)


# ==== curve fits ====


def test_curve_fits_with_fewer_than_two_points_return_none():
    assert fit_linear([1.0], [0.5]) is None
    assert fit_exponential([1.0], [0.5]) is None
    assert fit_sigmoid([], []) is None
    assert compare_curves([1.0], [0.5]) == {}
    assert best_fitting_curve([1.0], [0.5]) is None


def test_best_fitting_curve_level_gate_is_opt_in():
    xs = [2.0, 4.0, 6.0, 8.0, 12.0]
    ys = [0.9 - 0.05 * x for x in xs]
    assert best_fitting_curve(xs, ys)[0] == "linear"
    assert best_fitting_curve(xs, ys, min_levels=MIN_CURVE_LEVELS) is None
    # Repeated x values do not count as extra levels.
    assert best_fitting_curve(xs + xs, ys + ys, min_levels=MIN_CURVE_LEVELS) is None
    xs6 = xs + [16.0]
    ys6 = [0.9 - 0.05 * x for x in xs6]
    assert best_fitting_curve(xs6, ys6, min_levels=MIN_CURVE_LEVELS) is not None


# ==== slope_logit ====


def _logistic_sample(slope: float, intercept: float, n_per_x: int, seed: int):
    rng = random.Random(seed)
    xs, ys = [], []
    for x in range(0, 10):
        p = 1.0 / (1.0 + math.exp(-(intercept + slope * x)))
        for _ in range(n_per_x):
            xs.append(float(x))
            ys.append(rng.random() < p)
    return xs, ys


def test_slope_logit_recovers_a_known_slope():
    xs, ys = _logistic_sample(-0.5, 2.0, 200, seed=1)
    result = slope_logit(xs, ys, n_boot=200, rng=random.Random(0))
    assert result.slope == pytest.approx(-0.5, abs=0.06)
    assert result.intercept == pytest.approx(2.0, abs=0.3)
    assert result.ci_low < result.slope < result.ci_high
    assert result.ci_low < -0.5 < result.ci_high


def test_slope_logit_is_deterministic_given_rng():
    xs, ys = _logistic_sample(-0.3, 1.0, 20, seed=2)
    first = slope_logit(xs, ys, n_boot=100, rng=random.Random(7)).to_dict()
    second = slope_logit(xs, ys, n_boot=100, rng=random.Random(7)).to_dict()
    assert first == second
    default = slope_logit(xs, ys, n_boot=100).to_dict()
    assert default == slope_logit(xs, ys, n_boot=100, rng=random.Random(0)).to_dict()


def test_cluster_bootstrap_widens_the_interval_for_duplicated_items():
    xs, ys = _logistic_sample(-0.3, 1.0, 10, seed=3)
    # Each item copied 10 times: an i.i.d. bootstrap overstates precision.
    xs_rep = [x for x in xs for _ in range(10)]
    ys_rep = [y for y in ys for _ in range(10)]
    clusters = [i for i in range(len(xs)) for _ in range(10)]
    naive = slope_logit(xs_rep, ys_rep, n_boot=200, rng=random.Random(0))
    clustered = slope_logit(xs_rep, ys_rep, clusters=clusters, n_boot=200, rng=random.Random(0))
    assert clustered.n_clusters == len(xs)
    assert naive.slope == pytest.approx(clustered.slope)
    assert (clustered.ci_high - clustered.ci_low) > 2.0 * (naive.ci_high - naive.ci_low)


def test_slope_logit_handles_separable_and_degenerate_data():
    separable = slope_logit([0, 1, 2, 3], [True, True, False, False], n_boot=50)
    assert math.isfinite(separable.slope) and separable.slope < 0
    assert separable.separable is True
    assert slope_logit([1, 1, 1], [True, False, True]) is None
    with pytest.raises(ValueError):
        slope_logit([1, 2], [True])


def test_m4_separation_is_detected_including_quasi_complete():
    from analysis.failure_onset import _is_separable
    # Regression (M4): this input gave a clamp-driven (30.8, -20.7) fit.
    assert _is_separable([1, 1, 2, 2], [1, 1, 0, 0]) is True
    assert _is_separable([1, 2, 2, 3], [1, 1, 0, 0]) is True   # quasi: shares x=2
    assert _is_separable([1, 1, 1, 1], [1, 1, 1, 1]) is True   # single class
    assert _is_separable([1, 2, 3, 4], [1, 0, 1, 0]) is False  # overlap


def test_m4_separated_resamples_are_excluded_from_the_ci():
    xs = [1, 1, 2, 2]
    ys = [True, True, False, False]
    result = slope_logit(xs, ys, n_boot=200, rng=random.Random(0))
    assert result.separable is True
    # Every resample of a separated sample is separated or has no x spread.
    assert result.n_boot_failed == 200
    assert result.n_boot_separated > 0
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)
    payload = result.to_dict()
    assert payload["separable"] is True
    assert payload["n_boot_separated"] == result.n_boot_separated


def test_m4_overlapping_data_is_not_flagged_separable():
    xs, ys = _logistic_sample(-0.5, 2.0, 50, seed=4)
    result = slope_logit(xs, ys, n_boot=100, rng=random.Random(0))
    assert result.separable is False
    assert result.n_boot_separated == 0


def test_sigmoid_chance_floor_is_a_parameter():
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    ys = [0.95, 0.95, 0.9, 0.8, 0.6, 0.55, 0.5, 0.5]  # floor near 1/2
    default = fit_sigmoid(xs, ys)
    assert default.params["c"] <= 1.0 / 3.0 + 1e-9
    binary = fit_sigmoid(xs, ys, chance_floor=0.5)
    assert binary.params["c"] <= 0.5 + 1e-9
    assert binary.params["c"] > 1.0 / 3.0
    best = best_fitting_curve(xs, ys, chance_floor=0.5)
    assert best is not None
    with pytest.raises(ValueError):
        fit_sigmoid(xs, ys, chance_floor=1.5)


# ==== statistics ====


def test_low_success_rule_uses_minority_count_and_half_width():
    cells = [
        {"condition": "perfect", "correct": 50, "total": 50},   # minority 0
        {"condition": "near", "correct": 45, "total": 50},      # minority 5
        {"condition": "mid", "correct": 25, "total": 50},       # half-width ~0.13
        {"condition": "ok", "correct": 40, "total": 50},        # minority 10, hw ~0.11
        {"condition": "big", "correct": 500, "total": 1000},
        {"condition": "empty", "correct": 0, "total": 0},
    ]
    flagged = {cell["condition"]: cell for cell in flag_low_success_cells(cells)}
    assert set(flagged) == {"perfect", "near", "mid", "empty"}
    assert flagged["near"]["flag_reason"] == ["minority_count"]
    assert flagged["mid"]["flag_reason"] == ["wide_interval"]
    assert flagged["near"]["minority_count"] == 5
    assert "low_success" not in cells[0]


def test_mcnemar_paired_exact_branch():
    a = [True] * 5 + [False] * 3 + [True] * 10
    b = [False] * 5 + [True] * 3 + [True] * 10
    b01, b10, p = mcnemar_paired(a, b)
    assert (b01, b10) == (3, 5)
    expected = 2.0 * sum(math.comb(8, i) for i in range(4)) / 2 ** 8
    assert p == pytest.approx(min(1.0, expected))


def test_mcnemar_paired_is_exact_at_large_discordant_counts():
    a = [True] * 30
    b = [False] * 30
    b01, b10, p = mcnemar_paired(a, b)
    assert (b01, b10) == (0, 30)
    assert p == pytest.approx(2.0 / 2 ** 30)


def test_l2_mcnemar_paired_agrees_with_mcnemar_test():
    from analysis.statistics import mcnemar_test
    # Regression (L2): b=20, c=10 gave 0.0987 (exact) vs 0.1003 (chi-square).
    a = [False] * 20 + [True] * 10
    b = [True] * 20 + [False] * 10
    b01, b10, p = mcnemar_paired(a, b)
    assert (b01, b10) == (20, 10)
    assert p == pytest.approx(mcnemar_test(20, 10)["p_value"])
    assert p == pytest.approx(0.0987, abs=5e-4)
    # Exact stays finite and correct far past float range of 2**n.
    big_a = [False] * 700 + [True] * 700
    big_b = [True] * 700 + [False] * 700
    assert mcnemar_paired(big_a, big_b)[2] == pytest.approx(1.0)


def test_mcnemar_paired_no_discordant_pairs_and_length_check():
    assert mcnemar_paired([True, False], [True, False]) == (0, 0, 1.0)
    with pytest.raises(ValueError):
        mcnemar_paired([True], [True, False])


def test_holm_correction_matches_hand_computation():
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert holm_correction([]) == []
    with pytest.raises(ValueError):
        holm_correction([1.5])


def test_new_helpers_are_exported():
    for name in ("mcnemar_paired", "holm_correction", "slope_logit", "MIN_CURVE_LEVELS"):
        assert name in analysis.__all__
        assert getattr(analysis, name) is not None

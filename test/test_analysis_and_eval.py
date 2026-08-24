"""
test/test_analysis_and_eval.py
==============================
Unit tests for analysis (failure_onset, first_error) and eval (eval_harness, models).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.failure_onset import (
    best_fitting_curve,
    compare_curves,
    compute_failure_onset,
    fit_exponential,
    fit_linear,
    fit_sigmoid,
)
from analysis.first_error import (
    ErrorType,
    analyze_first_error,
)
from eval.eval_harness import (
    evaluate_predictions,
    extract_answer,
    format_prompt,
)
from eval.models import CORE_MODELS, OPTIONAL_MODELS


# ============================================================
# 1. FAILURE ONSET COMPUTATION
# ============================================================

print("=" * 70)
print("1. FAILURE ONSET")
print("=" * 70)

x_vals = [2, 4, 6, 8, 12, 16]
# Example from §11: 96%, 94%, 91%, 84%, 69%, 48% with tau=0.70 -> onset at 12
accs = [0.96, 0.94, 0.91, 0.84, 0.69, 0.48]

onset = compute_failure_onset(x_vals, accs, tau=0.70)
assert onset == 12, f"Expected onset at 12, got {onset}"

# Always passing -> None
assert compute_failure_onset(x_vals, [0.99] * 6, tau=0.70) is None

# Immediate failure -> 2
assert compute_failure_onset(x_vals, [0.50] * 6, tau=0.70) == 2

print(f"Onset for [96%, 94%, 91%, 84%, 69%, 48%] with tau=0.70: L_f = {onset} PASS")


# ============================================================
# 2. CURVE FITTING & MODEL SELECTION
# ============================================================

print()
print("=" * 70)
print("2. CURVE FITTING")
print("=" * 70)

fits = compare_curves([float(x) for x in x_vals], accs)
assert "linear" in fits
assert "exponential" in fits
assert "sigmoid" in fits

for name, fit in fits.items():
    print(f"  {name:12s} : R^2 = {fit.r_squared:.4f}, AIC = {fit.aic:.2f}")

best_name, best_fit = best_fitting_curve([float(x) for x in x_vals], accs)
print(f"  Best fit by AIC: {best_name}")
assert best_fit.r_squared > 0.80
print("PASS")


# ============================================================
# 3. FIRST-ERROR ANALYSIS
# ============================================================

print()
print("=" * 70)
print("3. FIRST-ERROR CLASSIFICATION")
print("=" * 70)

# NO_ERROR
gold = ["c0", "c1", "c2", "c0"]
pred_perfect = ["c0", "c1", "c2", "c0"]
res_perfect = analyze_first_error(gold, pred_perfect)
assert res_perfect.error_type == ErrorType.NO_ERROR
assert res_perfect.first_error_step is None
assert res_perfect.final_is_correct is True

# PROPAGATING_ERROR: error at step 1 and all subsequent are wrong
pred_prop = ["c0", "c2", "c3", "c1"]
res_prop = analyze_first_error(gold, pred_prop)
assert res_prop.error_type == ErrorType.PROPAGATING_ERROR
assert res_prop.first_error_step == 1
assert res_prop.final_is_correct is False

# LOCAL_ERROR: error at step 1, recovers at step 2, but final is wrong
pred_local = ["c0", "c2", "c2", "c3"]
res_local = analyze_first_error(gold, pred_local)
assert res_local.error_type == ErrorType.LOCAL_ERROR
assert res_local.first_error_step == 1
assert res_local.final_is_correct is False

# CANCELLATION_ERROR: error at step 1 and 2, but final matches
pred_cancel = ["c0", "c2", "c3", "c0"]
res_cancel = analyze_first_error(gold, pred_cancel)
assert res_cancel.error_type == ErrorType.CANCELLATION_ERROR
assert res_cancel.final_is_correct is True
assert res_cancel.first_error_step == 1

# FINAL_ONLY_ERROR: intermediate correct, final wrong
pred_final_only = ["c0", "c1", "c2", "c3"]
res_final_only = analyze_first_error(gold, pred_final_only)
assert res_final_only.error_type == ErrorType.FINAL_ONLY_ERROR
assert res_final_only.first_error_step == 3
assert res_final_only.final_is_correct is False

print("All 5 error types classified correctly: PASS")


# ============================================================
# 4. EVALUATION HARNESS & ANSWER EXTRACTION
# ============================================================

print()
print("=" * 70)
print("4. EVAL HARNESS & ANSWER EXTRACTION")
print("=" * 70)

# Format prompt test
prompt = format_prompt(
    context="A key was moved to the wooden shelf.",
    question="Where is the key now?",
    system_prompt="Answer with the container.",
)
assert "Narrative:" in prompt
assert "Where is the key now?" in prompt

# Answer extraction test
assert extract_answer("The final location is the wooden shelf.", candidate_containers=["wooden shelf", "glass box"]) == "wooden shelf"
assert extract_answer("Answer: True") == "True"
assert extract_answer("Answer: False") == "False"

# Evaluation runner test
mock_instances = [
    {
        "instance_id": "inst_001",
        "family": "basic_chain",
        "requested_factors": {"T": 4, "D": 0},
        "gold_answer": "wooden shelf",
        "step_wise_gold": ["c0", "c1", "c2", "c0"],
        "final_state": {"containers": ["wooden shelf", "glass crate"]},
    },
    {
        "instance_id": "inst_002",
        "family": "basic_chain",
        "requested_factors": {"T": 4, "D": 0},
        "gold_answer": "glass crate",
        "step_wise_gold": ["c0", "c1"],
        "final_state": {"containers": ["wooden shelf", "glass crate"]},
    },
]

mock_predictions = [
    {
        "instance_id": "inst_001",
        "pred_answer": "wooden shelf",
        "pred_trajectory": ["c0", "c1", "c2", "c0"],
    },
    {
        "instance_id": "inst_002",
        "pred_answer": "wooden shelf",  # wrong
        "pred_trajectory": ["c0", "c0"],  # propagating error
    },
]

eval_out = evaluate_predictions(mock_instances, mock_predictions)
assert eval_out["overall_total"] == 2
assert eval_out["overall_correct"] == 1
assert eval_out["overall_accuracy"] == 0.5
assert "basic_chain_T4_D0" in eval_out["condition_summaries"]

print("Evaluation harness test: PASS")


# ============================================================
# 5. CORE MODELS REGISTRY
# ============================================================

print()
print("=" * 70)
print("5. CORE MODELS REGISTRY")
print("=" * 70)

assert len(CORE_MODELS) == 5, f"Expected 5 core models, got {len(CORE_MODELS)}"
for model_key, cfg in CORE_MODELS.items():
    assert cfg.temperature == 0.0, f"{model_key} should have temperature=0"
    assert cfg.do_sample is False
    print(f"  [{model_key}] {cfg.hf_model_id} ({cfg.parameter_count_b}B)")

print("All 5 core models registered: PASS")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ALL ANALYSIS AND EVAL TESTS PASSED")
print("=" * 70)

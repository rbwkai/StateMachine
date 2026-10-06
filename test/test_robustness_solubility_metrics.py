"""
test/test_robustness_solubility_metrics.py
==========================================
Regression tests for two metric bugs:

- eval/robustness.py: prompt sensitivity is the spread over all variants (a
  wrong baseline with a right variant is sensitivity), paraphrase gains and
  drops are reported separately and do not cancel, and a gain is not robust.
- analysis/solubility.py: count records are judged on replay-pass consistency
  rather than on the integer gold appearing in the text, and location records
  are not penalised for the gold not being the most-mentioned container.
"""

from __future__ import annotations

import copy

import pytest

from analysis.solubility import check_answer_uniqueness, run_solubility_audit
from eval.robustness import (
    MINIMAL_INSTRUCTIONS,
    evaluate_paraphrase_robustness,
    evaluate_prompt_sensitivity,
    run_robustness_suite,
)
from test.conftest import build_record, quiet


def _loc_instance(iid: str = "r-1") -> dict:
    return {
        "instance_id": iid,
        "family": "basic_chain",
        "spec": {"query_type": "location"},
        "gold_answer": "the blue bag",
        "context": "A key was placed in the red box. The key was moved to the blue bag.",
        "question": "Where is the key now?",
        "sentences": [
            "A key was placed in the red box.",
            "The key was moved to the blue bag.",
        ],
        "step_wise_gold_answers": ["the red box", "the blue bag"],
        "final_state": {
            "container_names": {"c0": "the red box", "c1": "the blue bag"},
            "container_display_names": {"c0": "the red box", "c1": "the blue bag"},
        },
    }


RIGHT = "Final Answer: the blue bag"
WRONG = "Final Answer: the red box"


def _is_paraphrased(prompt: str) -> bool:
    return "is put into" in prompt


# ============================================================
# Prompt sensitivity
# ============================================================

def test_sensitivity_detected_when_baseline_wrong_variant_right() -> None:
    inst = _loc_instance()
    minimal = MINIMAL_INSTRUCTIONS["location"]

    def predict(prompt: str) -> str:
        return RIGHT if minimal in prompt else WRONG

    result = evaluate_prompt_sensitivity(inst, predict)
    assert result.variant_accuracies["v2_standard"] == 0.0
    assert result.variant_accuracies["v3_minimal"] == 1.0
    assert result.baseline_accuracy == 0.0
    assert result.gap == 1.0
    assert result.is_robust is False


def test_sensitivity_zero_when_all_variants_agree_wrong() -> None:
    result = evaluate_prompt_sensitivity(_loc_instance(), lambda p: WRONG)
    assert result.gap == 0.0
    assert result.is_robust is True


# ============================================================
# Paraphrase robustness
# ============================================================

def test_paraphrase_improvement_is_not_robust() -> None:
    def predict(prompt: str) -> str:
        return RIGHT if _is_paraphrased(prompt) else WRONG

    result = evaluate_paraphrase_robustness(_loc_instance(), predict)
    assert result.gap == -1.0
    assert result.is_robust is False


def test_paraphrase_gains_and_drops_do_not_cancel() -> None:
    gain_inst = _loc_instance("gain")
    drop_inst = _loc_instance("drop")

    # Instance identity is not in the prompt; differentiate by question text.
    drop_inst["question"] = "Where is the key located now?"

    def router(prompt: str) -> str:
        para = _is_paraphrased(prompt)
        if "located" in prompt:
            return WRONG if para else RIGHT  # drop
        return RIGHT if para else WRONG  # gain

    out = run_robustness_suite([gain_inst, drop_inst], router)
    agg = out["paraphrase_robustness"]
    assert agg["mean_gap"] == 0.0  # signed: documented to cancel
    assert agg["drop_rate"] == 0.5
    assert agg["gain_rate"] == 0.5
    assert agg["mean_abs_gap"] == 1.0
    assert agg["max_gap"] == 1.0
    assert agg["robust_rate"] == 0.0


def test_suite_prompt_aggregate_has_sensitivity() -> None:
    out = run_robustness_suite([_loc_instance()], lambda p: RIGHT)
    agg = out["prompt_sensitivity"]
    assert agg["mean_sensitivity"] == agg["mean_gap"] == 0.0
    assert agg["robust_rate"] == 1.0


# ============================================================
# Solubility
# ============================================================

COUNT_FAMILIES = [
    ("merge_chain", 2, 4),
    ("swap_chain", 2, 4),
    ("undo_chain", 1, 4),
    ("split_chain", 2, 4),
]


def _records(family: str, entity_count: int, target_updates: int, seeds=range(8)):
    out = []
    with quiet():
        for s in seeds:
            r = build_record(family, s, entity_count=entity_count, target_updates=target_updates)
            if r is not None:
                out.append(r)
    return out


@pytest.mark.parametrize("family,E,T", COUNT_FAMILIES)
def test_generated_count_records_are_soluble(family: str, E: int, T: int) -> None:
    records = _records(family, E, T)
    assert records, f"no {family} records generated"
    audit = run_solubility_audit(records)
    bad = [r for r in audit["per_instance"] if not r["is_soluble"]]
    assert audit["solubility_rate"] == 1.0, bad[:2]
    assert all(r["query_type"] == "count" for r in audit["per_instance"])


def test_generated_revision_records_are_soluble() -> None:
    records = _records("revision", 1, 4)
    assert records
    audit = run_solubility_audit(records)
    bad = [r for r in audit["per_instance"] if not r["is_soluble"]]
    assert audit["solubility_rate"] == 1.0, bad[:2]


def test_location_gold_not_most_mentioned_is_warning_only() -> None:
    inst = _loc_instance()
    inst["context"] = (
        "A key was placed in the red box. The key was moved to the green cup. "
        "The key was moved to the red box. The key was moved to the green cup. "
        "The key was moved to the blue bag."
    )
    inst["step_wise_gold_answers"] = [
        "the red box", "the green cup", "the red box", "the green cup", "the blue bag",
    ]
    inst["final_state"]["container_names"]["c2"] = "the green cup"
    result = check_answer_uniqueness(inst)
    assert result.is_soluble is True
    assert result.issues == []
    assert any("most-mentioned" in w for w in result.warnings)
    assert result.ambiguity_score > 0.0  # mention-share diagnostic retained


def test_inconsistent_count_record_is_flagged() -> None:
    records = _records("merge_chain", 2, 4, seeds=range(3))
    assert records
    bad = copy.deepcopy(records[0])
    final = int(bad["step_wise_gold_answers"][-1])
    bad["gold_answer"] = str(final + 1)
    result = check_answer_uniqueness(bad)
    assert result.is_soluble is False
    assert any("disagrees with final step-wise answer" in i for i in result.issues)


def test_non_integer_count_gold_is_flagged() -> None:
    records = _records("swap_chain", 2, 4, seeds=range(3))
    assert records
    bad = copy.deepcopy(records[0])
    bad["gold_answer"] = "the red crate"
    result = check_answer_uniqueness(bad)
    assert result.is_soluble is False
    assert any("not a non-negative integer" in i for i in result.issues)


def test_count_question_container_missing_from_context_is_flagged() -> None:
    records = _records("swap_chain", 2, 4, seeds=range(3))
    assert records
    bad = copy.deepcopy(records[0])
    gc = bad["gold_container"]
    name = bad["final_state"]["container_names"][gc]
    bad["context"] = bad["context"].replace(name, "somewhere")
    result = check_answer_uniqueness(bad)
    assert result.is_soluble is False
    assert any("not mentioned in context" in i for i in result.issues)

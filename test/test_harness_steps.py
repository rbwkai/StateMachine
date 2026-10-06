"""Harness step-trajectory regressions (review finding C1).

A perfect chain-of-thought response -- ``Step k: step_wise_gold_answers[k-1]``
for every narrated sentence plus the correct final answer -- must be scored
NO_ERROR with exact alignment. Before the fix it was scored PROPAGATING_ERROR
with every step wrong: the parsed display names were compared with container
ids in ``step_wise_gold``, shifted by one, and count cells were compared with
locations instead of counts.
"""

from __future__ import annotations

import pytest

from eval.eval_harness import evaluate_predictions
from eval.scoring import candidate_answers, normalize_text
from test.conftest import build_record, quiet


def _records():
    with quiet():
        cells = {
            "location": build_record("basic_chain", 3, target_updates=4),
            "distractors": build_record(
                "basic_chain", 4, target_updates=4, textual_distractors=3,
            ),
            "count": build_record("split_chain", 0, entity_count=2, target_updates=6),
        }
    for name, record in cells.items():
        assert record is not None, f"{name} cell must generate"
    return cells


RECORDS = _records()


def _cot(steps, final):
    lines = [f"Step {i + 1}: {value}" for i, value in enumerate(steps)]
    return "\n".join(lines + [f"Final Answer: {final}"])


def _evaluate(record, raw):
    report = evaluate_predictions(
        [record],
        [{"instance_id": record["instance_id"], "raw_prediction": raw}],
        chain_of_thought=True,
    )
    return report["instance_results"][0]


def _wrong_value(record, gold_value):
    """A parseable answer for this cell that differs from ``gold_value``."""
    if normalize_text(record["gold_answer"]).isdigit():
        return str(int(gold_value) + 5)
    for cand in candidate_answers(record):
        if normalize_text(cand) != normalize_text(gold_value):
            return cand
    pytest.skip("cell offers no alternative candidate")


def test_distractor_cell_has_one_gold_step_per_sentence():
    record = RECORDS["distractors"]
    assert record["measured_factors"].get("N_actual", 0) > 0
    assert len(record["step_wise_gold_answers"]) == len(record["sentences"])


def test_count_cell_steps_are_counts():
    record = RECORDS["count"]
    assert all(str(v).isdigit() for v in record["step_wise_gold_answers"])


@pytest.mark.parametrize("cell", sorted(RECORDS))
def test_perfect_cot_is_no_error(cell):
    record = RECORDS[cell]
    golds = record["step_wise_gold_answers"]
    result = _evaluate(record, _cot(golds, record["gold_answer"]))
    assert result["is_correct"] is True
    assert [normalize_text(p) for p in result["pred_trajectory"]] == [
        normalize_text(g) for g in golds
    ]
    assert result["gold_trajectory"] == golds
    analysis = result["error_analysis"]
    assert analysis["error_type"] == "NO_ERROR"
    assert analysis["step_errors"] == []
    assert analysis["first_error_step"] is None
    assert analysis["total_steps"] == len(golds)
    assert result["step_alignment"] == "exact"


@pytest.mark.parametrize("cell", sorted(RECORDS))
def test_one_wrong_mid_step_is_the_first_error(cell):
    record = RECORDS[cell]
    golds = list(record["step_wise_gold_answers"])
    assert len(golds) >= 3
    mid = len(golds) // 2  # 0-indexed; reported 1-indexed
    steps = list(golds)
    steps[mid] = _wrong_value(record, golds[mid])
    result = _evaluate(record, _cot(steps, record["gold_answer"]))
    analysis = result["error_analysis"]
    assert analysis["first_error_step"] == mid + 1
    assert analysis["step_errors"] == [mid + 1]
    # Later steps are correct again: an intermediate error with a correct final.
    assert analysis["error_type"] == "CANCELLATION_ERROR"
    assert result["step_alignment"] == "exact"


def test_missing_last_step_is_missing_and_under_stepped():
    record = RECORDS["location"]
    golds = record["step_wise_gold_answers"]
    result = _evaluate(record, _cot(golds[:-1], record["gold_answer"]))
    analysis = result["error_analysis"]
    # A padded step is unparsed, not wrong (analysis.first_error MISSING).
    assert analysis["error_type"] == "MISSING"
    assert analysis["step_errors"] == []
    assert analysis["missing_steps"] == [len(golds)]
    assert result["step_alignment"] == "under_stepped"


def test_extra_step_is_over_stepped_and_dropped():
    record = RECORDS["location"]
    golds = record["step_wise_gold_answers"]
    result = _evaluate(record, _cot(list(golds) + [golds[0]], record["gold_answer"]))
    assert result["step_alignment"] == "over_stepped"
    assert len(result["pred_trajectory"]) == len(golds)
    assert result["error_analysis"]["error_type"] == "NO_ERROR"


def test_markerless_response_has_no_trajectory():
    record = RECORDS["location"]
    result = _evaluate(record, f"Final Answer: {record['gold_answer']}")
    assert result["error_analysis"] is None
    assert result["step_alignment"] is None

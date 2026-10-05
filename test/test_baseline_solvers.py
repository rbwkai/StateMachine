"""Checklist 7: baseline solvers.

Covers the stateless and most-frequent-class (MFC) baselines: agreement with a
hand-solved reference, chance derived from the real answer space, the empty
prediction rate, the split-family premise behind the stateless baseline, and the
MFC condition key. Real defects are documented as plain failing `test_failsnow_*`
tests.
"""
from __future__ import annotations

from collections import Counter

import pytest

from eval.baselines import (
    BaselineResult,
    compute_mfc_baseline,
    compute_stateless_baseline,
    run_all_baselines,
    query_type_of,
)
from eval.scoring import candidate_answers, normalize_text, score_prediction
from test.conftest import build_record, quiet


LOCATION_CELLS = [
    ("basic_chain", 1, 4, 0, 0),
    ("interleaved_chain", 2, 4, 1, 0),
    ("revision", 1, 4, 0, 0),
    ("undo_redo_chain", 1, 8, 0, 0),
]
SEEDS = range(10)


def _records(cell, seeds=SEEDS):
    family, e, t, d, n = cell[:5]
    num_containers = cell[5] if len(cell) > 5 else 3
    with quiet():
        records = [
            build_record(family, seed, entity_count=e, target_updates=t,
                         distractor_updates=d, textual_distractors=n,
                         num_containers=num_containers, max_attempts=20)
            for seed in seeds
        ]
    return [record for record in records if record]


def _accuracy(results):
    return sum(1 for result in results if result.is_correct) / len(results)


# ---------------------------------------------------------------------------
# Chance is derived from the answer space, never hard-coded.
# ---------------------------------------------------------------------------

def chance_level(record):
    """1/size of the answer space this instance actually offers."""
    if query_type_of(record) == "count":
        splits = sum(1 for op in record["canonical_trace"] if op["op_type"] == "SPLIT")
        entity_count = record["requested_factors"]["E"]
        return 1.0 / (entity_count + splits + 1)
    return 1.0 / len(record["final_state"]["containers"])


@pytest.mark.parametrize("cell", [
    ("basic_chain", 1, 4, 0, 0),
    ("interleaved_chain", 2, 8, 1, 0, 4),  # four containers
    ("split_chain", 2, 6, 0, 0),
])
def test_chance_level_follows_the_answer_space_not_a_fixed_third(cell):
    """Chance is 1/3 for three containers, 1/4 for four, and 1/(E+splits+1) for
    count cells. A deterministic cycle over the candidate set must land on it."""
    records = _records(cell)
    assert records
    record = records[0]
    containers = len(record["final_state"]["containers"])
    if query_type_of(record) == "count":
        splits = sum(1 for op in record["canonical_trace"] if op["op_type"] == "SPLIT")
        assert chance_level(record) == pytest.approx(1.0 / (record["requested_factors"]["E"] + splits + 1))
    else:
        assert chance_level(record) == pytest.approx(1.0 / containers)

    assert chance_level(record) != pytest.approx(1.0 / 3.0) or containers == 3

    correct = 0
    for index, current in enumerate(records):
        space = _answer_space(current)
        guess = space[index % len(space)]
        scored = score_prediction(
            {"instance_id": current["instance_id"], "raw_prediction": f"Final Answer: {guess}"},
            current,
        )
        correct += scored["strict_correct"]
    assert correct / len(records) == pytest.approx(chance_level(record), abs=0.25)


def _answer_space(record):
    """Every answer this instance could accept, in a stable order."""
    if query_type_of(record) == "count":
        splits = sum(1 for op in record["canonical_trace"] if op["op_type"] == "SPLIT")
        return [str(value) for value in range(record["requested_factors"]["E"] + splits + 1)]
    names = record["final_state"]["container_display_names"]
    return [names[cid] for cid in sorted(record["final_state"]["containers"])]


# ---------------------------------------------------------------------------
# Hand-solved references.
# ---------------------------------------------------------------------------

def _initial_container_display(record):
    for op in record["canonical_trace"]:
        if op["op_type"] == "PUT" and op.get("obj_id") == record["query_entity"]:
            names = record["final_state"]["container_display_names"]
            return names.get(op["container"], op["container"])
    return ""


@pytest.mark.parametrize("cell", LOCATION_CELLS)
def test_stateless_matches_a_hand_solved_reference(cell):
    """Reference: the display name of the container holding the target's own
    first Put, solved by reading the trace, not by calling the baseline."""
    records = _records(cell)
    assert records
    results = {result.instance_id: result for result in compute_stateless_baseline(records)}
    assert len(results) == len(records)
    for record in records:
        expected = _initial_container_display(record)
        result = results[record["instance_id"]]
        assert result.pred_answer == expected, record["instance_id"]
        assert result.is_correct is (expected == record["gold_answer"])


def test_failsnow_stateless_predicts_zero_for_count_cells():
    """[checklist 7] the stateless baseline predicts the initial count for a
    count cell.

    [fails now] expected: pred_answer is the string "0". currently it is the
    empty string: "0" is not in the candidate set (only the gold value is), so
    the numeric extraction defect in checklist 6 swallows the baseline's guess.
    """
    records = _records(("split_chain", 2, 6, 0, 0))
    assert records
    for result in compute_stateless_baseline(records):
        assert result.pred_answer == "0", (
            f"{result.instance_id}: stateless count prediction extracted as "
            f"{result.pred_answer!r}"
        )


def test_mfc_matches_a_hand_solved_reference():
    """Reference: the most common normalised gold inside the condition group the
    baseline builds itself (family, T, D)."""
    records = _records(("basic_chain", 1, 4, 0, 0))
    assert records
    groups = {}
    for record in records:
        factors = record["requested_factors"]
        key = f"{record['family']}_T{factors['T']}_D{factors.get('D', 0)}"
        groups.setdefault(key, []).append(record)
    reference = {
        key: Counter(normalize_text(r["gold_answer"]) for r in group).most_common(1)[0][0]
        for key, group in groups.items()
    }
    by_id = {record["instance_id"]: record for record in records}
    for result in compute_mfc_baseline(records):
        record = by_id[result.instance_id]
        factors = record["requested_factors"]
        key = f"{record['family']}_T{factors['T']}_D{factors.get('D', 0)}"
        mode = reference[key]
        # Extraction re-renders the mode as the instance's own candidate string,
        # or drops it entirely when the pooled name is not in the candidate set.
        expected = next(
            (c for c in candidate_answers(record) if normalize_text(c) == mode), ""
        )
        assert result.pred_answer == expected, result.instance_id


def test_mfc_reports_per_condition_accuracy():
    records = _records(("basic_chain", 1, 4, 0, 0))
    summary = run_all_baselines(records)["mfc"]["per_condition"]
    assert list(summary) == ["basic_chain_T4_D0"]
    assert summary["basic_chain_T4_D0"]["total"] == len(records)


def test_query_type_is_read_from_the_question_when_the_spec_omits_it():
    location = _records(("basic_chain", 1, 4, 0, 0))[0]
    count = _records(("split_chain", 2, 6, 0, 0))[0]
    assert query_type_of(location) == "location"
    assert query_type_of(count) == "count"


# ---------------------------------------------------------------------------
# Known defects.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", LOCATION_CELLS)
def test_failsnow_empty_prediction_rate_is_zero(cell):
    """[checklist 7] 'The empty-prediction rate is 0.'

    [fails now] expected: both baselines always emit a parsable answer string.
    currently MFC pools the per-instance random *display names*, so its pick is
    outside the instance's candidate set and extracts as the empty string
    (18 of 20 on basic_chain T=4); only stateless is always non-empty.
    """
    records = _records(cell)
    results = compute_stateless_baseline(records) + compute_mfc_baseline(records)
    empty = [(r.baseline_type, r.instance_id) for r in results if not r.pred_answer]
    assert not empty, f"{len(empty)}/{len(results)} baseline predictions are empty: {empty[:4]}"


@pytest.mark.parametrize("cell", LOCATION_CELLS)
def test_failsnow_mfc_stays_at_or_below_five_percent(cell):
    """[checklist 7] 'MFC scores about 0-5% on location families.'

    [fails now] expected: at or below the 0.05 floor. currently the display-name
    pooling defect leaves MFC at 0.05-0.15 depending on the cell.
    """
    records = _records(cell)
    accuracy = _accuracy(compute_mfc_baseline(records))
    assert accuracy <= 0.05, f"MFC accuracy {accuracy:.2f} on {cell}"


@pytest.mark.parametrize("cell", LOCATION_CELLS)
def test_failsnow_stateless_stays_at_or_below_five_percent(cell):
    """[checklist 7] 'stateless scores about 0-5% on location families because it
    emits container IDs that extraction rejects.'

    [fails now] expected: at or below the 0.05 floor, so any cell above it
    signals either an extraction rejection or a shortcut the baseline should not
    enjoy. currently the baseline emits *display names*, not IDs, so extraction
    accepts them and accuracy is 0.20-1.00 (revision is a perfect shortcut: the
    target's Put is its final location).
    """
    records = _records(cell)
    accuracy = _accuracy(compute_stateless_baseline(records))
    assert accuracy <= 0.05, f"stateless accuracy {accuracy:.2f} on {cell}"


def test_failsnow_split_puts_the_target_before_the_others():
    """[checklist 7] 'Stateless works for split (the target's Put is not the
    first Put).'

    [fails now] expected: the target's Put is preceded by another object's Put,
    so reading the first Put cannot stand in for the target's location. currently
    the target's Put is the first Put in every sampled split_chain trace, so
    this family gives the stateless baseline the same signal as the narrative
    order rather than the initial-state shortcut it is meant to model.
    """
    records = _records(("split_chain", 2, 6, 0, 0))
    assert records
    offenders = []
    for record in records:
        puts = [op for op in record["canonical_trace"] if op["op_type"] == "PUT"]
        if puts and puts[0].get("obj_id") == record["query_entity"]:
            offenders.append(record["instance_id"])
    assert not offenders, f"target's Put is first in {len(offenders)}/{len(records)} traces"


def test_failsnow_mfc_groups_by_entity_count_and_textual_distractors():
    """[checklist 7] 'the grouping key includes family, T, D, E and N.'

    [fails now] expected: MFC is computed per (family, T, D, E, N). currently the
    key is family/T/D only, so two cells that differ only in E and N are pooled.
    """
    cell_a = _records(("basic_chain", 1, 4, 0, 0))
    cell_b = _records(("basic_chain", 2, 4, 0, 4))
    assert cell_a and cell_b
    assert cell_a[0]["family"] == cell_b[0]["family"]
    for record in cell_a + cell_b:
        assert record["requested_factors"]["T"] == 4

    results = compute_mfc_baseline(cell_a + cell_b)
    assert len(results) == len(cell_a) + len(cell_b)
    assert _accuracy(results) == 1.0, (
        "each cell is homogeneous, so a per-cell MFC must be right every time; "
        f"pooling gives {_accuracy(results):.2f}"
    )

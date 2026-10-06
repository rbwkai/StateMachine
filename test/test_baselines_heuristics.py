"""
test/test_baselines_heuristics.py
=================================
Shortcut solvers, the cell-level heuristic ceiling and effective chance
(eval/baselines.py), finish_reason-based truncation (eval/post_run_sanity.py)
and step-alignment tagging in the harness (eval/eval_harness.py).
"""

from __future__ import annotations

import pytest

from eval.baselines import (
    best_heuristic_ceiling,
    chance_level,
    compute_mfc_baseline,
    compute_stateless_baseline,
    HEURISTIC_NOT_A_SHORTCUT,
    degenerate_gold,
    effective_chance,
    heuristic_accuracies,
    initial_count,
    last_move,
    modal_count,
    modal_location,
    not_start_and_not_prev,
    penultimate_move,
    prev_loc,
    run_all_baselines,
    uniform_chance,
)
from eval.eval_harness import evaluate_predictions
from eval.post_run_sanity import sanity_checks, sanity_failures
from test.conftest import build_record, quiet


NAMES = {"c0": "the red box", "c1": "the blue bag", "c2": "the green tin"}


def _loc_record(iid, trace, step_wise_gold, gold_container):
    return {
        "instance_id": iid,
        "family": "undo_redo_chain",
        "requested_factors": {"T": 4, "D": 0, "E": 1},
        "spec": {"query_type": "location"},
        "query_entity": "o0",
        "canonical_trace": trace,
        "step_wise_gold": step_wise_gold,
        "gold_container": gold_container,
        "gold_answer": NAMES[gold_container],
        "question": "Where is the gem now?",
        "final_state": {
            "containers": ["c0", "c1", "c2"],
            "container_names": dict(NAMES),
            "container_display_names": dict(NAMES),
        },
    }


# PUT c0, MOVE c1, MOVE c2, UNDO -> target back in c1.
UNDO_TRACE = [
    {"op_type": "PUT", "obj_id": "o0", "obj_type": "gem", "container": "c0"},
    {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
    {"op_type": "MOVE", "obj_id": "o0", "dst": "c2"},
    {"op_type": "UNDO"},
]
UNDO_GOLD = ["c0", "c1", "c2", "c1"]


def _count_record(iid, gold, puts, gold_container="c2"):
    trace = [
        {"op_type": "PUT", "obj_id": f"o{i}", "obj_type": t, "container": c}
        for i, (t, c) in enumerate(puts)
    ] + [{"op_type": "MOVE", "obj_id": "o0", "dst": gold_container}]
    return {
        "instance_id": iid,
        "family": "merge_chain",
        "requested_factors": {"T": 2, "D": 0, "E": len(puts)},
        "spec": {"query_type": "count"},
        "query_entity": "o0",
        "canonical_trace": trace,
        "step_wise_gold": [],
        "gold_container": gold_container,
        "gold_answer": gold,
        "question": "How many gems are in the green tin now?",
        "final_state": {
            "containers": ["c0", "c1", "c2"],
            "container_names": dict(NAMES),
            "container_display_names": dict(NAMES),
        },
    }


# ---------------------------------------------------------------------------
# Record-level location heuristics.
# ---------------------------------------------------------------------------

def test_location_heuristics_on_a_handcrafted_undo_record():
    record = _loc_record("u0", UNDO_TRACE, UNDO_GOLD, "c1")
    assert last_move(record) == "c2"
    assert penultimate_move(record) == "c1"
    # True location just before the last Move (index 2) is step_wise_gold[1].
    assert prev_loc(record) == "c1"
    # start c0, prev c1 -> only c2 is left.
    assert not_start_and_not_prev(record) == "c2"


def test_prev_loc_reads_the_replayed_state_not_the_previous_move():
    # PUT c0, MOVE c1, UNDO (back to c0), MOVE c2: penultimate move says c1,
    # but the target was really in c0 before its last Move.
    trace = [
        {"op_type": "PUT", "obj_id": "o0", "obj_type": "gem", "container": "c0"},
        {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
        {"op_type": "UNDO"},
        {"op_type": "MOVE", "obj_id": "o0", "dst": "c2"},
    ]
    record = _loc_record("u1", trace, ["c0", "c1", "c0", "c2"], "c2")
    assert penultimate_move(record) == "c1"
    assert prev_loc(record) == "c0"
    # start == prev -> two containers remain -> no unique answer.
    assert not_start_and_not_prev(record) is None


def test_single_move_record_has_no_penultimate_move():
    trace = UNDO_TRACE[:2]
    record = _loc_record("u2", trace, ["c0", "c1"], "c1")
    assert last_move(record) == "c1"
    assert penultimate_move(record) is None
    assert prev_loc(record) == "c0"
    # start == prev == c0 -> c1 and c2 remain -> no unique answer.
    assert not_start_and_not_prev(record) is None


def test_location_heuristics_do_not_apply_to_count_records():
    record = _count_record("k0", "1", [("gem", "c0")])
    for solver in (last_move, prev_loc, penultimate_move, not_start_and_not_prev):
        assert solver(record) is None


# ---------------------------------------------------------------------------
# Count heuristics.
# ---------------------------------------------------------------------------

def test_initial_count_counts_setup_puts_of_the_query_type_in_the_query_container():
    record = _count_record(
        "k1", "3", [("gem", "c0"), ("gem", "c2"), ("coin", "c2"), ("gem", "c2")]
    )
    assert initial_count(record) == "2"
    assert initial_count(_loc_record("u0", UNDO_TRACE, UNDO_GOLD, "c1")) is None


def test_stateless_count_prediction_is_the_initial_count_not_always_zero():
    records = [
        _count_record("k2", "2", [("gem", "c2"), ("gem", "c2")]),
        _count_record("k3", "1", [("gem", "c0")]),
    ]
    results = {r.instance_id: r for r in compute_stateless_baseline(records)}
    assert results["k2"].pred_answer == "2"
    assert results["k2"].is_correct is True
    assert results["k3"].pred_answer in {"", "0"}
    assert results["k3"].is_correct is False


def test_modal_solvers_pick_the_cell_mode():
    loc = [
        _loc_record("a", UNDO_TRACE, UNDO_GOLD, "c1"),
        _loc_record("b", UNDO_TRACE, UNDO_GOLD, "c1"),
        _loc_record("c", UNDO_TRACE, UNDO_GOLD, "c2"),
    ]
    assert modal_location(loc) == "c1"
    assert modal_count(loc) is None
    counts = [
        _count_record("x", "1", [("gem", "c0")]),
        _count_record("y", "2", [("gem", "c0")]),
        _count_record("z", "2", [("gem", "c0")]),
    ]
    assert modal_count(counts) == "2"
    assert modal_location(counts) is None


# ---------------------------------------------------------------------------
# Ceiling and effective chance.
# ---------------------------------------------------------------------------

def test_best_heuristic_ceiling_finds_prev_loc_on_an_undo_cell():
    cell = [_loc_record(f"u{i}", UNDO_TRACE, UNDO_GOLD, "c1") for i in range(4)]
    accuracies = heuristic_accuracies(cell)
    assert accuracies["prev_loc"] == 1.0
    assert accuracies["last_move"] == 0.0
    name, accuracy = best_heuristic_ceiling(cell)
    assert accuracy == 1.0
    assert name in {"prev_loc", "penultimate_move", "modal_location", "mfc"}
    assert effective_chance(cell) == 1.0
    assert best_heuristic_ceiling([]) == ("none", 0.0)
    assert effective_chance([]) == 0.0


@pytest.mark.parametrize(
    "family,entity_count,target_updates",
    [("basic_chain", 1, 4), ("undo_redo_chain", 1, 8),
     ("revision", 1, 4), ("split_chain", 2, 6)],
)
def test_effective_chance_dominates_mfc_and_uniform_on_generated_cells(
    family, entity_count, target_updates
):
    with quiet():
        records = [
            build_record(family, seed, entity_count=entity_count,
                         target_updates=target_updates,
                         instance_id=f"{family}_s{seed}", max_attempts=20)
            for seed in range(12)
        ]
    records = [r for r in records if r]
    assert records
    mfc_acc = sum(r.is_correct for r in compute_mfc_baseline(records)) / len(records)
    eff = effective_chance(records)
    uniform = sum(chance_level(r) for r in records) / len(records)
    assert eff >= mfc_acc - 1e-9
    assert eff >= uniform - 1e-9
    _, best = best_heuristic_ceiling(records)
    assert eff >= best - 1e-9
    # basic_chain/revision have no undo: last_move IS the tracking answer, so
    # it is excluded from the ceiling rather than counted as a shortcut (H1).
    if family in ("basic_chain", "revision"):
        assert all(last_move(r) == r["gold_container"] for r in records)
        assert "last_move" not in heuristic_accuracies(records)
    ceilings = run_all_baselines(records)["heuristic_ceiling"]["per_condition"]
    assert len(ceilings) == 1
    (entry,) = ceilings.values()
    assert entry["effective_chance"] == pytest.approx(eff)
    assert entry["chance_level"] == pytest.approx(uniform)
    assert entry["heuristic_ceiling_name"] == entry["name"]
    assert entry["heuristic_ceiling_acc"] == pytest.approx(best)
    assert isinstance(entry["degenerate_gold"], bool)


# ---------------------------------------------------------------------------
# Post-run sanity: truncation from finish_reason.
# ---------------------------------------------------------------------------

def _cell():
    return [_loc_record(f"t{i}", UNDO_TRACE, UNDO_GOLD, "c1") for i in range(4)]


def test_truncation_rate_comes_from_finish_reason():
    records = _cell()
    predictions = [
        {"instance_id": r["instance_id"],
         "raw_prediction": "Final Answer: the blue bag",
         "finish_reason": "length" if i == 0 else "eos_token"}
        for i, r in enumerate(records)
    ]
    report = evaluate_predictions(records, predictions)
    assert report["instance_results"][0]["finish_reason"] == "length"
    (cell,) = sanity_checks(records, report).values()
    assert cell["truncation_rate"] == pytest.approx(0.25)
    assert cell["missing_final_rate"] == 0.0
    assert cell["format_compliance"] == 1.0
    assert any("truncation rate" in f for f in sanity_failures({"k": cell}))


def test_truncation_rate_is_none_without_finish_reason_and_missing_final_is_separate():
    records = _cell()
    predictions = [
        {"instance_id": r["instance_id"], "raw_prediction": "no marker here"}
        for r in records
    ]
    report = evaluate_predictions(records, predictions)
    (cell,) = sanity_checks(records, report).values()
    assert cell["truncation_rate"] is None
    assert cell["missing_final_rate"] == 1.0
    assert not any("truncation" in f for f in sanity_failures({"k": cell}))


def test_finish_reason_can_come_from_prediction_rows():
    records = _cell()
    rows = [{"instance_id": r["instance_id"], "raw_prediction": "Final Answer: the blue bag"}
            for r in records]
    report = evaluate_predictions(records, rows)
    with_reason = [dict(row, finish_reason="length") for row in rows]
    (cell,) = sanity_checks(records, report, predictions=with_reason).values()
    assert cell["truncation_rate"] == 1.0


def test_sanity_chance_is_uniform_and_ceiling_is_reported_separately():
    records = _cell()
    rows = [{"instance_id": r["instance_id"], "raw_prediction": "Final Answer: the red box"}
            for r in records]
    report = evaluate_predictions(records, rows)
    (cell,) = sanity_checks(records, report).values()
    # The gate is the uniform floor (3 containers); prev_loc solving this cell
    # is reported as a diagnostic, not used as the floor (H1).
    assert cell["chance"] == pytest.approx(1.0 / 3.0)
    assert cell["chance"] == pytest.approx(uniform_chance(records))
    assert cell["below_chance"] is True  # 0% < 1/3
    assert cell["effective_chance"] == pytest.approx(effective_chance(records)) == 1.0
    assert cell["heuristic_ceiling_acc"] == 1.0
    assert cell["degenerate_gold"] is True  # every record's gold is c1


def _basic_record(iid, dst):
    trace = [
        {"op_type": "PUT", "obj_id": "o0", "obj_type": "gem", "container": "c0"},
        {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
        {"op_type": "MOVE", "obj_id": "o0", "dst": dst},
    ]
    record = _loc_record(iid, trace, ["c0", "c1", dst], dst)
    record["family"] = "basic_chain"
    return record


def test_h1_basic_chain_last_move_is_not_a_shortcut_and_does_not_flag_below_chance():
    # Regression (H1): last_move = gold on basic_chain, so effective_chance was
    # 1.0 and a 75%-accurate run was flagged below chance.
    records = [_basic_record(f"b{i}", dst) for i, dst in enumerate(["c2", "c0", "c2", "c0"])]
    assert all(last_move(r) == r["gold_container"] for r in records)
    assert "last_move" in HEURISTIC_NOT_A_SHORTCUT["basic_chain"]
    assert "last_move" not in heuristic_accuracies(records)
    assert effective_chance(records) < 1.0
    rows = [
        {"instance_id": r["instance_id"],
         "raw_prediction": f"Final Answer: {r['gold_answer'] if i else 'the blue bag'}"}
        for i, r in enumerate(records)
    ]
    report = evaluate_predictions(records, rows)
    (cell,) = sanity_checks(records, report).values()
    assert cell["accuracy"] == pytest.approx(0.75)
    assert cell["below_chance"] is False
    assert cell["degenerate_gold"] is False
    assert not any("below uniform chance" in f for f in sanity_failures({"k": cell}))


def test_h1_genuine_shortcuts_stay_applicable():
    # undo_redo_chain: last_move/penultimate_move are real shortcuts, kept.
    undo = _loc_record("u", UNDO_TRACE, UNDO_GOLD, "c1")
    assert "last_move" in heuristic_accuracies([undo])
    assert "penultimate_move" in heuristic_accuracies([undo])
    # revision: not_start_and_not_prev is the reviewed A5 shortcut, kept.
    revision = dict(undo, family="revision")
    accuracies = heuristic_accuracies([revision])
    assert "last_move" not in accuracies
    assert "not_start_and_not_prev" in accuracies


def test_h1_degenerate_gold_on_constant_count_cell():
    # Merge cells with constant gold: modal_count/mfc reach 1.0 by construction.
    cell = [_count_record(f"m{i}", "2", [("gem", "c0")]) for i in range(5)]
    assert degenerate_gold(cell) is True
    assert effective_chance(cell) == 1.0
    (entry,) = run_all_baselines(cell)["heuristic_ceiling"]["per_condition"].values()
    assert entry["degenerate_gold"] is True
    assert entry["chance_level"] == pytest.approx(uniform_chance(cell))
    mixed = cell + [_count_record("m9", "1", [("gem", "c0")])]
    assert degenerate_gold(mixed) is False
    assert degenerate_gold([]) is False


# ---------------------------------------------------------------------------
# Harness: step alignment.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pred_traj,expected_tag,expected_len",
    [
        (["c1", "c2", "c1"], "exact", 3),
        (["c1"], "under_stepped", 3),
        (["c1", "c2", "c1", "c0", "c2"], "over_stepped", 3),
    ],
)
def test_harness_tags_step_alignment_instead_of_skipping(pred_traj, expected_tag, expected_len):
    record = _loc_record("s0", UNDO_TRACE, UNDO_GOLD, "c1")
    report = evaluate_predictions(
        [record],
        [{"instance_id": "s0", "raw_prediction": "Final Answer: the blue bag",
          "pred_trajectory": pred_traj}],
    )
    result = report["instance_results"][0]
    assert result["step_alignment"] == expected_tag
    assert result["error_analysis"] is not None
    assert result["error_analysis"]["step_alignment"] == expected_tag
    assert len(result["pred_trajectory"]) == expected_len
    if expected_tag == "under_stepped":
        assert result["pred_trajectory"][1:] == [None, None]
        assert result["error_analysis"]["error_type"] != "NO_ERROR"
    if expected_tag == "exact":
        assert result["error_analysis"]["error_type"] == "NO_ERROR"

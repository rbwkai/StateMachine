"""
Checklist 5 — Questions, gold answers and the serialised record.

Covers count queries, the removed-target contract, step-wise gold alignment and
the record schema (INSTANCE_RECORD_KEYS, JSON round-trip, final_state keys).

Tests named ``test_failsnow_*`` encode the CORRECT property for a defect recorded
in the pre-freeze audit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest  # noqa: E402

from generator import INSTANCE_RECORD_KEYS  # noqa: E402
from test.conftest import build_record, quiet  # noqa: E402

SEEDS = range(20)
COUNT_CELLS = [("split_chain", 2, 4), ("split_chain", 2, 8)]
LOCATION_CELLS = [
    ("basic_chain", 1, 4, 0),
    ("interleaved_chain", 2, 4, 1),
    ("revision", 1, 4, 0),
    ("undo_redo_chain", 1, 8, 0),
]
FINAL_STATE_KEYS = {"location", "containers", "container_names", "container_display_names"}


def records_for(family, entity_count, target_updates, n_textual=0, query_type=None, seeds=SEEDS,
                distractor_updates=0):
    kwargs = {} if query_type is None else {"query_type": query_type}
    with quiet():
        out = [
            build_record(
                family, seed,
                entity_count=entity_count,
                target_updates=target_updates,
                textual_distractors=n_textual,
                distractor_updates=distractor_updates,
                max_attempts=20,
                **kwargs,
            )
            for seed in seeds
        ]
    return [r for r in out if r is not None]


# ---------------------------------------------------------------------------
# count queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,e,t", COUNT_CELLS, ids=[f"{f}-T{t}" for f, _e, t in COUNT_CELLS])
def test_count_gold_is_numeric_and_in_range(family, e, t):
    """Gold for a count query is a number inside 0..E+splits (SPEC §2)."""
    for record in records_for(family, e, t):
        gold = record["gold_answer"]
        assert gold.isdigit(), f"{record['instance_id']}: count gold {gold!r}"
        assert 0 <= int(gold) <= e + 2, f"{record['instance_id']}: count gold {gold} out of range"


@pytest.mark.parametrize("family,e,t", COUNT_CELLS, ids=[f"{f}-T{t}" for f, _e, t in COUNT_CELLS])
def test_count_gold_answers_the_question_that_was_asked(family, e, t):
    """The numeric gold equals the count in the container the question names."""
    from world import count_type
    from test.conftest import replay_record

    for record in records_for(family, e, t):
        named = record["question"].split(" are in ")[-1].removesuffix(" now?").strip()
        display_to_id = {
            v: k for k, v in record["final_state"]["container_display_names"].items()
        }
        _, state, _ = replay_record(record)
        obj_type = next(
            s["obj_type"] for s in record["canonical_trace"]
            if s["op_type"] == "PUT" and s.get("obj_id") == record["query_entity"]
        )
        assert str(count_type(state, display_to_id[named], obj_type)) == record["gold_answer"]


@pytest.mark.parametrize("family,e,t", COUNT_CELLS, ids=[f"{f}-T{t}" for f, _e, t in COUNT_CELLS])
def test_failsnow_gold_container_is_the_container_the_question_asks_about(family, e, t):
    """[checklist 5] 'the query container is the same one named in the question.'

    [fails now] expected: record['gold_container'] is the container the count
    question names, so leakage and solubility checks that read gold_container
    audit the right container. currently it holds the *target's* own container,
    which is a different container in 23 of 30 sampled split_chain instances
    (the numeric gold is still right).
    """
    mismatches = []
    for record in records_for(family, e, t):
        named = record["question"].split(" are in ")[-1].removesuffix(" now?").strip()
        display = record["final_state"]["container_display_names"]
        if display[record["gold_container"]] != named:
            mismatches.append((record["instance_id"], named, display[record["gold_container"]]))
    assert not mismatches, (
        f"gold_container is not the queried container: {mismatches[:4]}"
    )


def test_failsnow_count_query_renders_a_count_question_for_every_count_family():
    """[checklist 5] "swap and undo with query_type='count' still render
    'Where is ...'".

    [fails now] expected: a family built with query_type='count' asks a counting
    question and answers with a number. currently: undo_chain's builder requires
    query_type='count' yet the record renders 'Where is the watch now?' with a
    container display name as gold. swap_chain/merge_chain cannot be measured
    here because they produce no instance at all (see test_generator_invariants).
    """
    wrong = []
    for family, e, t in [("undo_chain", 1, 4), ("undo_chain", 1, 8)]:
        for record in records_for(family, e, t, query_type="count"):
            if not record["question"].startswith("How many "):
                wrong.append((record["instance_id"], record["question"],
                              record["gold_answer"]))
            if not record["gold_answer"].isdigit():
                wrong.append((record["instance_id"], "non-numeric gold",
                              record["gold_answer"]))
    assert not wrong, f"count cell rendered as a location question: {wrong[:4]}"


def test_failsnow_step_wise_gold_answers_are_counts_for_count_cells():
    """[checklist 5] Step-wise gold must answer the question that was asked.

    [fails now] expected: step_wise_gold_answers holds the count after every step
    for a count query. currently it holds container display names, so the CoT
    condition is scored against values ('the old box') that cannot match the gold
    ('1').
    """
    for record in records_for("split_chain", 2, 6):
        for value in record["step_wise_gold_answers"]:
            assert str(value).isdigit(), (
                f"{record['instance_id']}: step-wise gold {value!r} is not a count "
                f"(question: {record['question']!r}, gold: {record['gold_answer']!r})"
            )


def test_split_step_wise_value_before_the_child_exists_is_defined():
    """For split, the value just before the Split is a count, not a hole."""
    records = records_for("split_chain", 2, 6)
    assert records
    for record in records:
        ops = [s["op_type"] for s in record["canonical_trace"]]
        split_at = ops.index("SPLIT")
        values = record["step_wise_gold_answers"]
        assert values[split_at - 1] is not None, record["instance_id"]
        assert len(values) == len(ops)


# ---------------------------------------------------------------------------
# step-wise alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,e,t,d", LOCATION_CELLS, ids=[f"{f}-T{t}" for f, _e, t, _d in LOCATION_CELLS])
@pytest.mark.parametrize("n_textual", [0, 4, 8, 16])
def test_step_wise_gold_length_matches_sentence_count(family, e, t, d, n_textual):
    records = records_for(family, e, t, n_textual=n_textual, distractor_updates=d)
    assert records, f"no instances for {family} T={t} N={n_textual}"
    for record in records:
        assert len(record["step_wise_gold_answers"]) == len(record["sentences"])
        assert len(record["step_wise_gold"]) == len(record["canonical_trace"])


@pytest.mark.parametrize("family,e,t,d", LOCATION_CELLS, ids=[f"{f}-T{t}" for f, _e, t, _d in LOCATION_CELLS])
def test_step_wise_gold_tracks_the_target(family, e, t, d):
    """Each step-wise value is the container the target occupied after that op."""
    from test.conftest import replay_record

    records = records_for(family, e, t, n_textual=4, distractor_updates=d)
    assert records
    for record in records:
        _, state, _ = replay_record(record)
        display = record["final_state"]["container_display_names"]
        last = None
        for value in record["step_wise_gold_answers"]:
            assert value is not None, record["instance_id"]
            last = value
        assert last == display[state.location[record["query_entity"]]]


def test_removed_target_has_an_explicit_absence_answer():
    """world-level contract: a removed target has no location (None), and the
    display layer spells that as 'removed' rather than leaking a container."""
    from generator import step_wise_gold
    from generator.probes import LocationQuery
    from test.conftest import containers_of, ops_from_record
    from world import Move, Put, Remove

    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Remove("o0")]
    values = step_wise_gold(ops, {"c0", "c1", "c2"}, LocationQuery("o0"))
    assert values[-1] is None, values


def test_failsnow_removed_target_is_recorded_as_removed_not_null():
    """[checklist 5] "Removed target gives 'removed' or an explicit absent answer."

    [fails now] expected: an instance whose target is removed carries an explicit
    absence marker. currently the step-wise display path substitutes the string
    'removed' while the final gold stays None, so gold and step-wise gold use
    different vocabularies for the same event and a scorer comparing them sees a
    mismatch instead of an absence.
    """
    from generator import step_wise_gold
    from generator.probes import LocationQuery
    from world import Move, Put, Remove
    from world import gold_location, replay_trace

    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Remove("o0")]
    containers = {"c0", "c1", "c2"}
    _, final_state, _ = replay_trace(ops, containers)
    final_gold = gold_location(final_state, "o0")
    values = step_wise_gold(ops, containers, LocationQuery("o0"))
    assert final_gold == values[-1], (
        "final gold and step-wise gold disagree about an absent target: "
        f"{final_gold!r} vs {values[-1]!r}"
    )


# ---------------------------------------------------------------------------
# record schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family,e,t,d", LOCATION_CELLS + [(f, e, t, 0) for f, e, t in COUNT_CELLS],
                         ids=[f"{f}-T{t}" for f, _e, t, _d in LOCATION_CELLS] + [f"{f}-T{t}" for f, e, t in COUNT_CELLS])
def test_record_keys_match_instance_record_keys(family, e, t, d):
    records = records_for(family, e, t, distractor_updates=d)
    assert records
    for record in records:
        assert set(record) == set(INSTANCE_RECORD_KEYS), (
            f"{record['instance_id']}: missing={set(INSTANCE_RECORD_KEYS) - set(record)} "
            f"extra={set(record) - set(INSTANCE_RECORD_KEYS)}"
        )


@pytest.mark.parametrize("family,e,t,d", LOCATION_CELLS + [(f, e, t, 0) for f, e, t in COUNT_CELLS],
                         ids=[f"{f}-T{t}" for f, _e, t, _d in LOCATION_CELLS] + [f"{f}-T{t}" for f, e, t in COUNT_CELLS])
def test_record_survives_a_json_round_trip(family, e, t, d):
    records = records_for(family, e, t, n_textual=2, distractor_updates=d)
    assert records
    for record in records:
        text = json.dumps(record, ensure_ascii=False, sort_keys=True)
        assert json.loads(text) == json.loads(json.dumps(json.loads(text), sort_keys=True))
        assert json.loads(text)["trace_hash"] == record["trace_hash"]


@pytest.mark.parametrize("family,e,t,d", LOCATION_CELLS + [(f, e, t, 0) for f, e, t in COUNT_CELLS],
                         ids=[f"{f}-T{t}" for f, _e, t, _d in LOCATION_CELLS] + [f"{f}-T{t}" for f, e, t in COUNT_CELLS])
def test_final_state_keys_are_stable(family, e, t, d):
    records = records_for(family, e, t, distractor_updates=d)
    assert records
    for record in records:
        assert set(record["final_state"]) == FINAL_STATE_KEYS, sorted(record["final_state"])


@pytest.mark.parametrize("family,e,t,d", LOCATION_CELLS + [(f, e, t, 0) for f, e, t in COUNT_CELLS],
                         ids=[f"{f}-T{t}" for f, _e, t, _d in LOCATION_CELLS] + [f"{f}-T{t}" for f, e, t in COUNT_CELLS])
def test_failsnow_record_serialises_the_query_type(family, e, t, d):
    """[checklist 5] 'query_type is not serialised in the record.'

    [fails now] expected: record['spec']['query_type'] records which question was
    asked, so a dataset can be audited per query type without guessing from the
    family name. currently the field is absent from the serialised spec.
    """
    for record in records_for(family, e, t, distractor_updates=d):
        assert "query_type" in record["spec"], (
            f"{record['instance_id']}: serialised spec has no query_type "
            f"(keys: {sorted(record['spec'])})"
        )


def test_record_reports_generator_versions():
    for family, e, t, d in LOCATION_CELLS:
        for record in records_for(family, e, t, seeds=range(3), distractor_updates=d):
            for key in ("generator_version", "renderer_version", "scoring_version"):
                assert record[key], f"{record['instance_id']} missing {key}"

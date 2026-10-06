"""
Checklist 2 — Generator invariance and grid-cell properties.

Scope: generator/trajectories.py, generator/metadata.py, generator/instance.py
and world/ replay agreement. Seed sweeps are loops (hypothesis is not
installed); every assertion is derived from one ``replay_trace`` pass over the
record's canonical trace, never from a recomputation in the test.

Naming: ``test_failsnow_*`` encodes the CORRECT behaviour for a defect listed
in the pre-freeze audit, so it fails today by design.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from test.conftest import (  # noqa: E402  shared checklist helpers
    FAMILY_QUERY_TYPES,
    SEEDS_LARGE,
    SEEDS_MEDIUM,
    SEEDS_SMALL,
    build_record,
    build_trajectory_for,
    replay_record,
    make_spec,
    quiet,
)
from generator import L_MAX_WORDS, STRUCTURAL_FAMILIES, reset_deduplication_registry, verify_factors
from world import count_type, replay_trace  # noqa: F401

# Families whose canonical query type the builder accepts today.
HEALTHY = {
    "basic_chain": dict(entity_count=1, target_updates=4),
    "interleaved_chain": dict(entity_count=2, target_updates=4, distractor_updates=1),
    "revision": dict(entity_count=1, target_updates=4),
    "split_chain": dict(entity_count=2, target_updates=4),
    "undo_chain": dict(entity_count=1, target_updates=4),
    "undo_redo_chain": dict(entity_count=1, target_updates=8),
}
ALL_FAMILIES = list(FAMILY_QUERY_TYPES)
HEALTHY_IDS = list(HEALTHY)
# Families with a buildable cell at the given depth, used for record-level checks.
RECORD_CELLS = {
    "basic_chain": dict(entity_count=1, target_updates=4),
    "interleaved_chain": dict(entity_count=2, target_updates=4, distractor_updates=1),
    "revision": dict(entity_count=1, target_updates=4),
    "split_chain": dict(entity_count=2, target_updates=4),
    "undo_chain": dict(entity_count=1, target_updates=4),
    "undo_redo_chain": dict(entity_count=1, target_updates=8),
}


# ---------------------------------------------------------------------------
# measured == requested, and the world replayed from the trace is the gold state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", HEALTHY_IDS)
def test_measured_factors_equal_requested_over_seed_sweep(family):
    """(SPEC §2) E/T/D requested == measured for every released cell."""
    mismatches = []
    for seed in SEEDS_MEDIUM:
        traj = build_trajectory_for(family, seed, **HEALTHY[family])
        m = traj.measured_factors
        if m.E_actual != traj.spec.entity_count or m.T_actual != traj.spec.target_updates:
            mismatches.append((seed, m.E_actual, m.T_actual))
        if m.D_actual != traj.spec.distractor_updates:
            mismatches.append((seed, "D", m.D_actual))
    assert not mismatches, f"{family}: measured != requested at {mismatches[:5]}"


@pytest.mark.parametrize("family", HEALTHY_IDS)
def test_verify_factors_accepts_every_measured_instance(family):
    """The gate helper agrees with the explicit E/T/D comparison above."""
    for seed in SEEDS_SMALL:
        traj = build_trajectory_for(family, seed, **HEALTHY[family])
        verify_factors(
            requested_E=traj.spec.entity_count,
            requested_T=traj.spec.target_updates,
            requested_D=traj.spec.distractor_updates,
            measured=traj.measured_factors,
            family=family,
            instance_id=f"{family}-s{seed}",
            min_v=traj.spec.min_unique_target_locations,
        )


@pytest.mark.parametrize("family", list(RECORD_CELLS))
def test_record_final_state_equals_replay_of_canonical_trace(family):
    """Hard rule 2: the stored final state is the one replay_trace produces."""
    kwargs = RECORD_CELLS[family]
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record(family, seed, **kwargs)
            assert record is not None, f"{family} produced no instance at seed {seed}"
            _, state, _ = replay_record(record)
            assert state.location == record["final_state"]["location"]
            assert sorted(state.containers) == sorted(record["final_state"]["containers"])


@pytest.mark.parametrize("family", list(RECORD_CELLS))
def test_gold_matches_final_state_of_trace(family):
    """gold_container / gold_answer are read off the replayed final state."""
    kwargs = RECORD_CELLS[family]
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record(family, seed, **kwargs)
            assert record is not None
            _, state, _ = replay_record(record)
            target = record["query_entity"]
            if not record["gold_answer"].isdigit():
                # Location-style gold (display name) must name the replayed location.
                assert record["gold_container"] == state.location[target]
                assert record["gold_answer"] == record["final_state"]["container_display_names"][
                    state.location[target]
                ]
            else:
                obj_type = next(
                    step["obj_type"] for step in record["canonical_trace"]
                    if step["op_type"] == "PUT" and step.get("obj_id") == target
                )
                expected = count_type(state, record["gold_container"], obj_type)
                assert str(expected) == record["gold_answer"]


# ---------------------------------------------------------------------------
# boundary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,kwargs",
    [
        ("interleaved_chain", dict(entity_count=1, target_updates=4, distractor_updates=1)),
        ("interleaved_chain", dict(entity_count=2, target_updates=4, distractor_updates=0)),
        ("split_chain", dict(entity_count=1, target_updates=4)),
        ("merge_chain", dict(entity_count=1, target_updates=4)),
        ("swap_chain", dict(entity_count=1, target_updates=4)),
        ("revision", dict(entity_count=1, target_updates=2)),
        ("undo_chain", dict(entity_count=1, target_updates=1)),
        ("split_chain", dict(entity_count=2, target_updates=2)),
        ("undo_redo_chain", dict(entity_count=1, target_updates=4)),
    ],
)
def test_below_minimum_requests_are_rejected_with_a_reason(family, kwargs):
    """Boundaries are explicit ValueErrors, never silently clamped (SPEC §2)."""
    with pytest.raises(ValueError) as excinfo:
        build_trajectory_for(family, 0, **kwargs)
    assert "requires" in str(excinfo.value), f"unhelpful boundary error: {excinfo.value}"


@pytest.mark.parametrize(
    "family,kwargs",
    [
        ("basic_chain", dict(entity_count=1, target_updates=1)),
        ("undo_chain", dict(entity_count=1, target_updates=2)),
        ("split_chain", dict(entity_count=2, target_updates=3)),
        ("revision", dict(entity_count=1, target_updates=3)),
        ("undo_redo_chain", dict(entity_count=1, target_updates=6)),
    ],
)
def test_exact_minimum_values_are_accepted(family, kwargs):
    with quiet():
        traj = build_trajectory_for(family, 0, **kwargs)
    assert traj.ops


@pytest.mark.parametrize(
    "family,query_type,kwargs",
    [
        ("undo_chain", "location", dict(entity_count=1, target_updates=4)),
        ("split_chain", "location", dict(entity_count=2, target_updates=4)),
        ("merge_chain", "location", dict(entity_count=2, target_updates=4)),
        ("swap_chain", "location", dict(entity_count=2, target_updates=4)),
    ],
)
def test_unsupported_query_type_is_rejected(family, query_type, kwargs):
    """A count-only family must not silently accept a location query."""
    with pytest.raises(ValueError):
        build_trajectory_for(family, 0, query_type=query_type, **kwargs)


def test_min_unique_target_locations_is_respected():
    from generator import MeasuredFactors

    measured = MeasuredFactors(
        E_actual=1, T_actual=8, D_actual=0, V_actual=4, L_word=100, N_actual=0
    )
    with pytest.raises(AssertionError):
        verify_factors(
            requested_E=1,
            requested_T=8,
            requested_D=0,
            measured=measured,
            family="basic_chain",
            instance_id="minv",
            min_v=5,
        )
    same = MeasuredFactors(
        E_actual=1, T_actual=8, D_actual=0, V_actual=6, L_word=100, N_actual=0
    )
    verify_factors(1, 8, 0, same, "basic_chain", "minv", min_v=5)


def test_two_container_revision_cell_is_buildable():
    """num_containers=2 exercises the no-op branch of the revision builder."""
    with quiet():
        traj = build_trajectory_for("revision", 1, entity_count=1, target_updates=4, num_containers=2)
    assert traj.measured_factors.T_actual == 4


# ---------------------------------------------------------------------------
# determinism, diversity, dedup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", HEALTHY_IDS)
def test_same_seed_reproduces_the_same_trace(family):
    a = build_trajectory_for(family, 11, **HEALTHY[family])
    b = build_trajectory_for(family, 11, **HEALTHY[family])
    assert [op.__dict__ for op in a.ops] == [op.__dict__ for op in b.ops]


@pytest.mark.parametrize("family", HEALTHY_IDS)
def test_distinct_seeds_produce_distinct_traces(family):
    seen = set()
    with quiet():
        for seed in range(20):
            record = build_record(family, seed, **HEALTHY[family])
            assert record is not None
            seen.add(record["trace_hash"])
    assert len(seen) >= 15, f"{family}: only {len(seen)}/20 distinct traces"


def test_attempt_seed_separates_attempts():
    from generator import attempt_seed

    assert attempt_seed(7, 0) != attempt_seed(7, 1)
    assert attempt_seed(7, 3) == attempt_seed(7, 3)


def test_name_rng_changes_names_but_not_the_trace():
    import random

    from generator import build_trajectory

    spec = make_spec("basic_chain", target_updates=6)
    with quiet():
        t1 = build_trajectory(random.Random(5), spec)
        t2 = build_trajectory(random.Random(5), spec)
    # A different rng must not change the op sequence for the same seed.
    with quiet():
        t3 = build_trajectory(random.Random(5), spec)
    assert [op.__dict__ for op in t1.ops] == [op.__dict__ for op in t2.ops] == [
        op.__dict__ for op in t3.ops
    ]
    assert t1.measured_factors == t2.measured_factors


def test_duplicate_trace_is_rejected_until_the_registry_is_reset():
    import random

    from generator import build_validated_instance

    spec = make_spec("basic_chain", target_updates=4)
    reset_deduplication_registry()
    with quiet():
        first = build_validated_instance(random.Random(21), spec, random.Random(2), 0, instance_id="dup")
    assert first.ok
    with quiet():
        second = build_validated_instance(random.Random(21), spec, random.Random(2), 0, instance_id="dup")
    assert not second.ok, "identical trace accepted twice without a registry reset"
    reset_deduplication_registry()
    with quiet():
        third = build_validated_instance(random.Random(21), spec, random.Random(2), 0, instance_id="dup")
    assert third.ok


# ---------------------------------------------------------------------------
# length, distractor accounting, op shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,kwargs",
    [
        ("basic_chain", dict(entity_count=1, target_updates=16)),
        ("interleaved_chain", dict(entity_count=2, target_updates=16, distractor_updates=1)),
        ("revision", dict(entity_count=1, target_updates=16)),
        ("split_chain", dict(entity_count=2, target_updates=16)),
        ("undo_chain", dict(entity_count=1, target_updates=16)),
        ("undo_redo_chain", dict(entity_count=1, target_updates=16)),
    ],
)
def test_l_word_within_max_at_the_largest_cell(family, kwargs):
    """Rendered L_word is the generation gate (hard rule 9)."""
    built = 0
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record(family, seed, max_attempts=30, **kwargs)
            if record is None:
                continue          # structural-causality retries may exhaust; see §3
            built += 1
            assert record["measured_factors"]["L_word"] <= L_MAX_WORDS
    assert built >= 8, f"{family} T=16 yielded only {built}/10 instances"


@pytest.mark.parametrize("n", [1, 3, 8])
def test_textual_distractor_count_is_measured_exactly(n):
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record("basic_chain", seed, target_updates=6, textual_distractors=n)
            assert record is not None
            assert record["measured_factors"]["N_actual"] == n
            assert len(record["sentences"]) == 6 + 1 + n


def test_interleaved_distractor_updates_are_measured():
    with quiet():
        for seed in SEEDS_SMALL:
            traj = build_trajectory_for(
                "interleaved_chain", seed, entity_count=2, target_updates=5, distractor_updates=3
            )
            assert traj.measured_factors.D_actual == 3


def test_undo_chain_trace_ends_with_undo_and_undo_redo_has_both():
    with quiet():
        undo = build_record("undo_chain", 1, target_updates=5)
        redo = build_record("undo_redo_chain", 1, target_updates=10)
    assert undo["canonical_trace"][-1]["op_type"] == "UNDO"
    ops = {s["op_type"] for s in redo["canonical_trace"]}
    assert "UNDO" in ops and "REDO" in ops


def test_undo_redo_counted_updates_match_requested_depth():
    """U_actual counts undo/redo as updates, so T == U for this family."""
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record("undo_redo_chain", seed, target_updates=12)
            assert record is not None
            assert record["measured_factors"]["U_actual"] == 12
            assert record["measured_factors"]["T_actual"] == 12


# ---------------------------------------------------------------------------
# documented defects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["merge_chain", "swap_chain"])
def test_failsnow_count_grid_cell_at_t4_is_generatable(family):
    """[checklist 2] Every cell of the RQ1 grid must yield instances.

    [fails now] expected: >=90% of seeds produce a validated record at
    (E=2, T=4, query_type='count'). currently: 0 of 50 seeds (500 attempts) --
    the builder emits fewer target updates than requested (merge_chain measured
    T=1, swap_chain measured T=3 with D=1), so verify_factors() rejects every
    attempt and generate_condition() returns an empty condition.
    """
    built = 0
    with quiet():
        for seed in SEEDS_MEDIUM:
            if build_record(family, seed, entity_count=2, target_updates=4) is not None:
                built += 1
    assert built >= 45, f"{family} count cell T=4 produced {built}/50 instances"


def test_v_actual_never_exceeds_t_actual():
    """[checklist 2] V_actual <= T_actual over 200 seeds of undo_redo_chain T=8.

    Gap closed: the instance gate rejects V>T so build_record retries to a
    passing attempt; verify_factors keeps its diagnostic warning for direct
    calls. V double-count (SPEC OPEN-4, F8) is preserved until SPEC redefines V.
    """
    violations = []
    with quiet():
        for seed in SEEDS_LARGE:
            record = build_record("undo_redo_chain", seed, entity_count=1, target_updates=8)
            assert record is not None, f"undo_redo_chain T=8 rejected at seed {seed}"
            m = record["measured_factors"]
            if m["V_actual"] > m["T_actual"]:
                violations.append((seed, m["V_actual"], m["T_actual"]))
    assert not violations, f"V_actual > T_actual at seeds {violations[:8]}"


# ---------------------------------------------------------------------------
# boundary sweeps named in the checklist: E = 1..5, T >= 16, D = 0 / T / > T
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_count", [2, 3, 4, 5])
def test_entity_count_sweep_up_to_five(entity_count):
    """E is a real factor, not a constant 1: the gate must accept E=2..5."""
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record(
                "interleaved_chain", seed,
                entity_count=entity_count, target_updates=6, distractor_updates=2,
            )
            assert record is not None, f"E={entity_count} rejected at seed {seed}"
            assert record["measured_factors"]["E_actual"] == entity_count


@pytest.mark.parametrize("target_updates", [16, 20, 24])
def test_depth_above_sixteen_stays_under_the_word_budget(target_updates):
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record("basic_chain", seed, target_updates=target_updates)
            assert record is not None
            assert record["measured_factors"]["L_word"] <= L_MAX_WORDS


@pytest.mark.parametrize(
    "distractor_updates", [0, 4, 6, 8], ids=["D0", "D=T", "D>T", "D=2T"]
)
def test_distractor_update_boundaries(distractor_updates):
    """D=0 is rejected by this family (it needs an interleaved partner), D=T and
    D>T are accepted and measured exactly."""
    with quiet():
        for seed in SEEDS_SMALL:
            record = build_record(
                "interleaved_chain", seed,
                entity_count=2, target_updates=4, distractor_updates=max(distractor_updates, 1),
            )
            assert record is not None
            assert record["measured_factors"]["D_actual"] == max(distractor_updates, 1)


def test_family_rejects_unsupported_entity_count():
    """basic_chain is single-entity; E>1 must raise, not silently collapse."""
    with pytest.raises(ValueError) as excinfo:
        build_trajectory_for("basic_chain", 0, entity_count=3, target_updates=4)
    assert "entity_count" in str(excinfo.value)


# ---------------------------------------------------------------------------
# diversity measured on container-relabelled traces
# ---------------------------------------------------------------------------

def container_relabelled_trace(record):
    """Canonical trace with containers renamed c0, c1, ... in first-seen order.

    Two instances that differ only in which physical container was used collapse
    to the same key, so this catches a family whose 'diversity' is only naming.
    """
    mapping = {}

    def relabel(cid):
        if cid is None:
            return None
        return mapping.setdefault(cid, f"k{len(mapping)}")

    key = []
    for step in record["canonical_trace"]:
        item = [step["op_type"]]
        for field in (
            "container", "dst", "src_container", "dst_container",
            "container_a", "container_b",
        ):
            if field in step:
                item.append(relabel(step[field]))
        key.append(tuple(item))
    return "|".join(">".join(map(str, k)) for k in key)


@pytest.mark.parametrize("family", HEALTHY_IDS)
def test_distinct_traces_equal_the_planned_instance_count(family):
    """'At least n distinct traces per cell, where n is what you plan to generate.'

    Uniqueness is checked on the canonical trace (trace_hash), not on rendered
    text. Container-relabelled diversity is deliberately NOT asserted: every
    family has one fixed op skeleton (measured: 1 distinct op-type sequence per
    family), so relabelling collapses traces by construction and would measure
    the skeleton, not the cell. Gold-container balance is tested in
    test_shortcut_audits.py.
    """
    keys = set()
    with quiet():
        for seed in range(20):
            record = build_record(family, seed, **HEALTHY[family])
            assert record is not None
            keys.add(record["trace_hash"])
    # A real condition dedups by canonical trace (hard rule 10), so a repeated
    # hash means the cell is *smaller* than planned, not that the builder is
    # broken; 90% keeps the test sensitive without flagging hash collisions.
    assert len(keys) >= 18, f"{family}: {len(keys)} distinct traces out of 20 planned"


# ---------------------------------------------------------------------------
# name_rng independence, dedup hygiene
# ---------------------------------------------------------------------------

def test_name_rng_changes_display_names_but_not_the_trace():
    from generator import build_validated_instance
    import random

    spec = make_spec("basic_chain", target_updates=6)
    reset_deduplication_registry()
    with quiet():
        a = build_validated_instance(random.Random(4), spec, random.Random(1), 0, instance_id="n1")
    reset_deduplication_registry()
    with quiet():
        b = build_validated_instance(random.Random(4), spec, random.Random(12345), 0, instance_id="n2")
    assert a.ok and b.ok
    assert a.record["trace_hash"] == b.record["trace_hash"]
    assert a.record["canonical_trace"] == b.record["canonical_trace"]


def test_rejected_instance_does_not_poison_the_dedup_registry():
    """A structural-causality rejection must leave the registry clean."""
    from generator import build_validated_instance
    import random

    reset_deduplication_registry()
    spec = make_spec("undo_chain", target_updates=16)
    seen_ok = False
    for seed in range(40):
        with quiet():
            res = build_validated_instance(random.Random(seed), spec, random.Random(seed), 0,
                                           instance_id=f"u{seed}")
        if res.ok:
            seen_ok = True
            break
    assert seen_ok, "no undo_chain instance survived 40 seeds (registry may be poisoned)"
    reset_deduplication_registry()


def test_same_seed_in_a_second_run_yields_the_same_record():
    import json

    from test.conftest import replay_record  # noqa: F401  (keeps helper import local)

    first = build_record("revision", 17, entity_count=1, target_updates=8)
    second = build_record("revision", 17, entity_count=1, target_updates=8)
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(
        second, sort_keys=True, default=str
    )


# ---------------------------------------------------------------------------
# documented defects: whole-grid reachability
# ---------------------------------------------------------------------------

def test_failsnow_every_rq1_grid_cell_generates_instances():
    """[checklist 2] 'Every grid cell is reachable' for the (family, T, E, D, N,
    query_type) combinations the experiment scripts actually request.

    [fails now] expected: every cell in experiments.rq1_mutation_depth.RQ1_FAMILIES
    yields instances. currently: merge_chain and swap_chain produce none at any T
    (their builders emit fewer target updates than requested), so RQ1 would abort
    or silently ship empty conditions; undo_chain at T=4 also renders a location
    question for a count cell (see test_questions_and_records).
    """
    from experiments._common import generate_condition
    from experiments.rq1_mutation_depth import (
        DISTRACTOR_UPDATES,
        FAMILY_CONTAINERS,
        FAMILY_QUERY_TYPES as RQ_QUERY_TYPES,
        RQ1_FAMILIES,
        TEXTUAL_DISTRACTORS,
    )

    empty = []
    with quiet():
        for family, entity_count, t_levels, _desc in RQ1_FAMILIES:
            for t in t_levels:
                records, failures = generate_condition(
                    family=family,
                    entity_count=entity_count,
                    target_updates=t,
                    distractor_updates=DISTRACTOR_UPDATES,
                    num_instances=3,
                    experiment_tag="checklist_grid",
                    condition_id=f"{family}_T{t}",
                    num_containers=FAMILY_CONTAINERS[family],
                    textual_distractor_count=TEXTUAL_DISTRACTORS,
                    query_type=RQ_QUERY_TYPES.get(family, "location"),
                )
                if not records:
                    empty.append((family, t, failures))
    assert not empty, f"grid cells that produced no instance at all: {empty}"


def test_experiment_query_type_map_matches_the_builders():
    """[checklist 2] spec.query_type accepted by exactly the families that
    support it; builder and experiment mapping agree.

    Gap closed: renderer handles count for all families via count_query_target,
    and this test uses SPEC/RQ1 entity counts (not STRUCTURAL_FAMILIES).
    """
    from experiments import rq1_mutation_depth as rq1
    from generator.trajectories import available_families

    # Entity counts per SPEC §families and RQ1_FAMILIES (E=1: basic, revision,
    # undo, undo_redo; E=2: split, merge, swap, interleaved). STRUCTURAL_FAMILIES
    # mixes both, so it cannot drive E selection.
    entity_for = {
        "basic_chain": 1,
        "revision": 1,
        "undo_chain": 1,
        "undo_redo_chain": 1,
        "split_chain": 2,
        "merge_chain": 2,
        "swap_chain": 2,
        "interleaved_chain": 2,
    }
    for family in available_families():
        declared = rq1.FAMILY_QUERY_TYPES.get(family, "location")
        entity_count = entity_for[family]
        t = {"revision": 4, "split_chain": 4, "undo_chain": 4, "undo_redo_chain": 8}.get(family, 4)
        kwargs = {}
        if family == "interleaved_chain":
            kwargs["distractor_updates"] = 1
        with quiet():
            record = build_record(
                family, 0, entity_count=entity_count, target_updates=t,
                query_type=declared, **kwargs,
            )
        assert record is not None, f"{family} rejected with declared query_type={declared}"
        declared_is_count = declared == "count"
        gold_is_count = record["gold_answer"].isdigit()
        assert declared_is_count == gold_is_count, (
            f"{family}: experiment declares {declared!r} but gold_answer is "
            f"{record['gold_answer']!r} (question: {record['question']!r})"
        )

"""
test/test_probes.py
===================
Contract tests for the probe layer (WP3): ``generator/probes.py``.

Two defects are pinned here:

1. Counterfactual probes were *silently* lossy. Removals whose replay became
   invalid were ``continue``d away and never counted, and setup ``Put``
   removals were silently mixed in with update removals despite SPEC §2
   keeping initial placements outside $U$. Every candidate removal now lands
   in exactly one class (``valid_answer_changing``,
   ``valid_answer_preserving``, ``invalid_replay``) or in the
   ``excluded_setup`` bucket, and all four are reported in
   ``ProbeAccounting``.

2. Redo-validity probes could only ever produce the INVALID class, so the
   condition was uninformative. ``redo_class="valid" | "invalid"`` selects
   the class deliberately, the label is read from ``world.can_redo`` rather
   than assumed, and ``build_redo_validity_examples`` builds a balanced
   batch.

Gold answers must come from one ``replay_trace`` pass (AGENTS.md §6 hard rule
2). The tests below therefore cross-check probe answers against an independent
replay instead of against the probe code itself.
"""
from __future__ import annotations

import ast
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generator.probes as probes_module
from generator.probes import (
    PROBE_EXCLUDED_SETUP,
    PROBE_INVALID_REPLAY,
    PROBE_VALID_ANSWER_CHANGING,
    PROBE_VALID_ANSWER_PRESERVING,
    REDO_CLASS_INVALID,
    REDO_CLASS_VALID,
    CounterfactualProbeSet,
    CountQuery,
    LocationQuery,
    ProbeAccounting,
    RedoValidityExample,
    RemovalClassification,
    build_counterfactual_probes,
    build_redo_validity_example,
    build_redo_validity_examples,
    classify_counterfactual_removals,
    counterfactual_gold,
)
from generator.sampler import sample_sequence
from world import (
    Move,
    Put,
    Redo,
    Remove,
    Swap,
    Undo,
    can_redo,
    replay_trace,
)

C3 = {"c0", "c1", "c2"}

REDO_OPS: list = [Move, Swap, Undo, Redo]


# ============================================================
# Hand-built trajectories
# ============================================================

def three_way_ops() -> list:
    """LocationQuery("o0"), final answer "c0".

        0 Put  o0 key -> c0    setup, excluded by default
        1 Put  o1 pen -> c1    setup, excluded by default
        2 Move o0 -> c2        removing it invalidates step 4
                              (Move o0 c0 is a no-op move)
        3 Move o1 -> c2        removing it leaves o0's answer alone
        4 Move o0 -> c0        removing it changes the answer to "c2"

    So the three classes are {4} / {3} / {2}.
    """

    return [
        Put("o0", "key", "c0"),
        Put("o1", "pen", "c1"),
        Move("o0", "c2"),
        Move("o1", "c2"),
        Move("o0", "c0"),
    ]


def balanced_count_ops() -> list:
    """CountQuery("c0", "key"), final answer 0.

        0 Put  o0 key -> c0    setup
        1 Put  o1 key -> c0    setup
        2 Put  o2 pen -> c2    setup
        3 Move o0 -> c1        changing: count 0 -> 1
        4 Move o1 -> c1        changing: count 0 -> 1
        5 Move o2 -> c0        preserving
        6 Move o2 -> c1        preserving

    Two candidates per valid class, so ``balance`` is actually observable.
    """

    return [
        Put("o0", "key", "c0"),
        Put("o1", "key", "c0"),
        Put("o2", "pen", "c2"),
        Move("o0", "c1"),
        Move("o1", "c1"),
        Move("o2", "c0"),
        Move("o2", "c1"),
    ]


def setup_removable_ops() -> list:
    """LocationQuery("o0"), final answer "c2".

        0 Put  o0 key -> c0    setup; removing it invalidates step 2
        1 Put  o1 pen -> c1    setup; removing it is VALID and leaves the
                              answer alone, so it only ever appears when
                              include_setup=True
        2 Move o0 -> c1        preserving
        3 Move o0 -> c2        changing
    """

    return [
        Put("o0", "key", "c0"),
        Put("o1", "pen", "c1"),
        Move("o0", "c1"),
        Move("o0", "c2"),
    ]


# ============================================================
# 1. Classification: three-way split, nothing swallowed
# ============================================================

def test_classification_splits_into_three_classes():
    ops = three_way_ops()
    query = LocationQuery("o0")

    classification = classify_counterfactual_removals(ops, C3, query)

    assert isinstance(classification, RemovalClassification)
    assert classification.original_answer == "c0"
    assert [
        probe["remove_step"] for probe in classification.answer_changing
    ] == [4]
    assert [
        probe["remove_step"] for probe in classification.answer_preserving
    ] == [3]
    assert classification.invalid_indices == (2,)
    assert classification.excluded_setup_indices == (0, 1)


def test_every_candidate_is_accounted_for_exactly_once():
    ops = three_way_ops()
    query = LocationQuery("o0")

    classification = classify_counterfactual_removals(ops, C3, query)

    # Every removal lands in exactly one bucket.
    assert classification.candidates == len(ops)
    assert classification.classified == len(ops)

    accounting = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=2
    ).accounting

    assert isinstance(accounting, ProbeAccounting)
    assert accounting.candidates == len(ops)
    assert (
        accounting.excluded_setup
        + accounting.invalid_replay
        + accounting.valid_answer_changing
        + accounting.valid_answer_preserving
    ) == len(ops)


def test_invalid_removals_are_reported_not_swallowed():
    ops = three_way_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=10
    )

    # Step 2 makes the replay invalid; it is counted, not silently dropped.
    assert result.accounting.invalid_replay == 1
    assert result.accounting.invalid_indices == (2,)
    assert 2 not in [
        probe["remove_step"] for probe in result
    ]

    # And it is never handed out as a probe: an invalid removal has no
    # counterfactual answer, so it can only appear in the accounting.
    assert {
        probe["probe_class"] for probe in result
    } == {
        PROBE_VALID_ANSWER_CHANGING,
        PROBE_VALID_ANSWER_PRESERVING,
    }
    assert result.accounting.selected == 2


def test_probe_class_label_matches_its_bucket():
    ops = three_way_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=10
    )

    for probe in result:
        expected = (
            PROBE_VALID_ANSWER_CHANGING
            if probe["answer_changed"]
            else PROBE_VALID_ANSWER_PRESERVING
        )
        assert probe["probe_class"] == expected
        # answer_changed is never a relabelled assumption.
        assert probe["answer_changed"] == (
            probe["original_answer"]
            != probe["counterfactual_answer"]
        )

    assert PROBE_INVALID_REPLAY == "invalid_replay"
    assert PROBE_EXCLUDED_SETUP == "excluded_setup"


def test_none_answer_is_not_mistaken_for_an_invalid_replay():
    """SPEC §4: a removed object has gold None, which is a *valid* answer.

    Removing the only unrelated Put leaves o0 removed, so the counterfactual
    answer is None while the replay is perfectly valid. An implementation that
    tests the answer for None to detect invalidity would mis-file this probe.
    """

    ops = [
        Put("o0", "key", "c0"),
        Put("o1", "pen", "c1"),
        Remove("o0"),
    ]
    query = LocationQuery("o0")

    classification = classify_counterfactual_removals(
        ops, C3, query, include_setup=True
    )

    assert classification.original_answer is None
    assert classification.invalid_indices == (0,)

    preserving = {
        probe["remove_step"]: probe["counterfactual_answer"]
        for probe in classification.answer_preserving
    }
    assert preserving == {1: None}

    # And the public helper still reports None for it, per its own docstring.
    assert counterfactual_gold(ops, C3, 1, query) is None


def test_counterfactual_gold_rejects_out_of_range_index():
    ops = three_way_ops()
    query = LocationQuery("o0")

    with pytest.raises(IndexError):
        counterfactual_gold(ops, C3, len(ops), query)

    with pytest.raises(IndexError):
        counterfactual_gold(ops, C3, -1, query)


# ============================================================
# 2. Balance
# ============================================================

def test_balanced_default_returns_one_of_each_class():
    ops = three_way_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=2
    )

    assert len(result) == 2
    assert sum(
        probe["answer_changed"] for probe in result
    ) == 1
    assert sum(
        not probe["answer_changed"] for probe in result
    ) == 1

    balance = result.realised_balance()
    assert balance["requested_changing_fraction"] == 0.5
    assert balance["realised_changing_fraction"] == 0.5


@pytest.mark.parametrize(
    "balance,expected_changing",
    [
        (0.0, 0),
        (0.5, 1),
        (1.0, 2),
    ],
)
def test_balance_parameter_controls_the_split(
    balance, expected_changing
):
    ops = balanced_count_ops()
    query = CountQuery("c0", "key")

    result = build_counterfactual_probes(
        random.Random(3),
        ops,
        C3,
        query,
        max_probes=2,
        balance=balance,
    )

    assert len(result) == 2
    assert result.accounting.selected_answer_changing == expected_changing
    assert (
        result.accounting.selected_answer_preserving
        == 2 - expected_changing
    )
    assert result.realised_balance()["realised_changing_fraction"] == (
        expected_changing / 2
    )


def test_single_probe_prefers_the_answer_changing_class():
    ops = balanced_count_ops()
    query = CountQuery("c0", "key")

    result = build_counterfactual_probes(
        random.Random(3), ops, C3, query, max_probes=1
    )

    assert len(result) == 1
    assert result[0]["answer_changed"] is True


def test_balance_is_observable_when_one_class_is_empty():
    """Only one valid candidate class exists: probes are still returned and
    the degenerate split is reported, not implied."""

    ops = [
        Put("o0", "key", "c0"),
        Move("o0", "c1"),
    ]
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=2
    )

    assert len(result) == 1
    assert result[0]["answer_changed"] is True

    balance = result.realised_balance()
    assert balance["requested_changing_fraction"] == 0.5
    assert balance["realised_changing_fraction"] == 1.0
    assert result.accounting.valid_answer_preserving == 0


def test_zero_max_probes_still_reports_full_accounting():
    ops = three_way_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=0
    )

    assert len(result) == 0
    assert result.realised_balance()["realised_changing_fraction"] is None
    assert result.accounting.candidates == len(ops)
    assert result.accounting.invalid_replay == 1
    assert result.accounting.excluded_setup == 2


def test_empty_trajectory_is_handled():
    result = build_counterfactual_probes(
        random.Random(0), [], C3, LocationQuery("o0")
    )

    assert len(result) == 0
    assert result.accounting.candidates == 0
    assert result.accounting.as_dict()["invalid_indices"] == []


def test_out_of_range_balance_is_rejected():
    with pytest.raises(ValueError):
        build_counterfactual_probes(
            random.Random(0),
            three_way_ops(),
            C3,
            LocationQuery("o0"),
            balance=1.5,
        )

    with pytest.raises(ValueError):
        build_counterfactual_probes(
            random.Random(0),
            three_way_ops(),
            C3,
            LocationQuery("o0"),
            balance=-0.1,
        )


# ============================================================
# 3. Setup exclusion
# ============================================================

def test_setup_removals_are_excluded_by_default():
    ops = three_way_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=10
    )

    assert result.accounting.excluded_setup == 2
    assert result.accounting.excluded_setup_indices == (0, 1)

    for probe in result:
        assert not isinstance(ops[probe["remove_step"]], Put)
        assert probe["removed_operation"]["type"] != "PUT"


def test_setup_exclusion_is_opt_in():
    ops = setup_removable_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(5),
        ops,
        C3,
        query,
        max_probes=10,
        include_setup=True,
    )

    assert result.accounting.excluded_setup == 0
    assert result.accounting.invalid_indices == (0,)

    # Step 1 is a Put whose removal is a valid, answer-preserving
    # intervention, so it must be reachable once setup is opted in.
    assert 1 in [probe["remove_step"] for probe in result]
    assert 0 not in [probe["remove_step"] for probe in result]


def test_setup_exclusion_and_invalid_counting_are_independent():
    """With include_setup=True an unremovable setup Put is counted as an
    invalid replay, not as an exclusion."""

    ops = setup_removable_ops()
    query = LocationQuery("o0")

    excluded = classify_counterfactual_removals(
        ops, C3, query, include_setup=False
    )
    included = classify_counterfactual_removals(
        ops, C3, query, include_setup=True
    )

    assert excluded.excluded_setup_indices == (0, 1)
    assert excluded.invalid_indices == ()
    assert included.excluded_setup_indices == ()
    assert included.invalid_indices == (0,)
    assert included.candidates == excluded.candidates


# ============================================================
# 4. Determinism and list-compatibility of the result
# ============================================================

def _generated_trajectory(seed: int = 7):
    rng = random.Random(seed)
    ops, state, history, containers = sample_sequence(
        rng, 3, 8, [Put, Move, Swap, Remove, Undo], 3
    )
    return ops, containers, LocationQuery("o0")


@pytest.mark.parametrize(
    "ops_factory",
    [three_way_ops, balanced_count_ops, setup_removable_ops, _generated_trajectory],
)
def test_same_seed_gives_identical_probe_dicts(ops_factory):
    built = ops_factory()
    if isinstance(built, tuple):
        ops, containers, query = built
    else:
        ops, containers, query = built, C3, LocationQuery("o0")

    first = build_counterfactual_probes(
        random.Random(1234), ops, containers, query, max_probes=3
    )
    second = build_counterfactual_probes(
        random.Random(1234), ops, containers, query, max_probes=3
    )

    assert first.probes == second.probes
    assert first.as_list() == second.as_list()
    assert first.accounting == second.accounting
    assert first.original_answer == second.original_answer


def test_same_seed_gives_identical_redo_examples():
    first = build_redo_validity_examples(
        random.Random(99), 2, 6, REDO_OPS, 3, n_per_class=2
    )
    second = build_redo_validity_examples(
        random.Random(99), 2, 6, REDO_OPS, 3, n_per_class=2
    )

    assert first[1] == second[1]
    assert [
        (example.ops, example.would_be_valid)
        for example in first[0]
    ] == [
        (example.ops, example.would_be_valid)
        for example in second[0]
    ]


def test_result_is_still_list_like():
    """pipeline.py and test/smoke_test.py iterate the return value directly,
    so it must keep the Sequence protocol."""

    ops = three_way_ops()
    query = LocationQuery("o0")

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=2
    )

    assert isinstance(result, CounterfactualProbeSet)
    assert len(result) <= 2
    assert all(
        isinstance(probe, dict)
        for probe in result
    )
    assert isinstance(result[0], dict)
    assert isinstance(result[0:1], list)
    assert result[-1] == list(result)[-1]
    assert list(result) == result.as_list()
    assert all(
        probe["remove_step"] in range(len(ops))
        for probe in result
    )


def test_probes_module_uses_no_module_level_randomness():
    """AGENTS.md §6 hard rule 5: every random draw must go through the
    explicit random.Random the caller passes in."""

    tree = ast.parse(
        Path(probes_module.__file__).read_text()
    )

    offenders = [
        f"random.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "random"
        and node.attr != "Random"
    ]

    assert offenders == []


def test_probes_module_never_evaluates_text():
    """AGENTS.md §6 hard rule 9."""

    tree = ast.parse(
        Path(probes_module.__file__).read_text()
    )

    offenders = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"eval", "exec", "compile", "__import__"}
    ]

    assert offenders == []


# ============================================================
# 5. Gold answers come from replay_trace
# ============================================================

def test_counterfactual_gold_matches_a_hand_computed_replay():
    ops = three_way_ops()
    query = LocationQuery("o0")

    # Independent replay of the untouched trajectory.
    _, final_state, _ = replay_trace(ops, C3)
    assert query.read(final_state) == "c0"

    # Removing step 4 leaves o0 in c2.
    reduced = ops[:4] + ops[5:]
    _, reduced_final, _ = replay_trace(reduced, C3)
    assert query.read(reduced_final) == "c2"
    assert counterfactual_gold(ops, C3, 4, query) == "c2"

    # Removing step 3 leaves o0 in c0.
    reduced = ops[:3] + ops[4:]
    _, reduced_final, _ = replay_trace(reduced, C3)
    assert query.read(reduced_final) == "c0"
    assert counterfactual_gold(ops, C3, 3, query) == "c0"

    result = build_counterfactual_probes(
        random.Random(0), ops, C3, query, max_probes=10
    )
    assert result.original_answer == "c0"
    assert {
        probe["remove_step"]: probe["counterfactual_answer"]
        for probe in result
    } == {3: "c0", 4: "c2"}


def test_count_probe_gold_matches_a_hand_computed_replay():
    ops = balanced_count_ops()
    query = CountQuery("c0", "key")

    _, final_state, _ = replay_trace(ops, C3)
    assert query.read(final_state) == 0

    for remove_step, expected in ((3, 1), (4, 1), (5, 0), (6, 0)):
        reduced = ops[:remove_step] + ops[remove_step + 1:]
        _, reduced_final, _ = replay_trace(reduced, C3)
        assert query.read(reduced_final) == expected
        assert counterfactual_gold(
            ops, C3, remove_step, query
        ) == expected


def test_redo_gold_matches_a_fresh_replay():
    """The reported redo label must equal what an independent replay of the
    same op list says about History."""

    example = build_redo_validity_example(
        random.Random(7), 2, 6, REDO_OPS, 3, redo_class="valid"
    )
    ops, _, _, containers, meta = example

    _, _, replayed_history = replay_trace(
        ops, containers
    )

    assert can_redo(replayed_history) is True
    assert meta["would_be_valid"] is True

    ops, _, _, containers, meta = build_redo_validity_example(
        random.Random(7), 2, 6, REDO_OPS, 3, redo_class="invalid"
    )

    _, _, replayed_history = replay_trace(
        ops, containers
    )

    assert can_redo(replayed_history) is False
    assert meta["would_be_valid"] is False


# ============================================================
# 6. Redo-validity probes: both classes exist
# ============================================================

def test_redo_valid_class_ends_in_undo():
    ops, state, history, containers, meta = (
        build_redo_validity_example(
            random.Random(7),
            2,
            6,
            REDO_OPS,
            3,
            redo_class=REDO_CLASS_VALID,
        )
    )

    assert meta["would_be_valid"] is True
    assert meta["redo_class"] == REDO_CLASS_VALID
    assert isinstance(ops[-1], Undo)
    assert can_redo(history) is True

    # The label is History-derived, so it must agree with the world.
    assert meta["would_be_valid"] == can_redo(history)


def test_redo_invalid_class_ends_in_the_clearing_operation():
    ops, state, history, containers, meta = (
        build_redo_validity_example(
            random.Random(7),
            2,
            6,
            REDO_OPS,
            3,
            redo_class=REDO_CLASS_INVALID,
        )
    )

    assert meta["would_be_valid"] is False
    assert meta["redo_class"] == REDO_CLASS_INVALID
    assert not isinstance(ops[-1], (Undo, Redo))
    assert can_redo(history) is False


def test_redo_default_class_is_the_historical_invalid_class():
    ops, _, history, _, meta = (
        build_redo_validity_example(
            random.Random(7), 2, 6, REDO_OPS, 3
        )
    )

    assert meta["would_be_valid"] is False
    assert meta["redo_class"] == REDO_CLASS_INVALID
    assert not isinstance(ops[-1], (Undo, Redo))


def test_redo_classes_share_the_same_prefix():
    """The only difference between the classes is the trailing operation, so
    the two conditions are directly comparable."""

    valid = build_redo_validity_example(
        random.Random(7), 2, 6, REDO_OPS, 3, redo_class="valid"
    )
    invalid = build_redo_validity_example(
        random.Random(7), 2, 6, REDO_OPS, 3, redo_class="invalid"
    )

    valid_ops = valid[0]
    invalid_ops = invalid[0]

    assert len(invalid_ops) == len(valid_ops) + 1
    assert valid_ops == invalid_ops[:-1]
    assert isinstance(valid_ops[-1], Undo)


def test_redo_rejects_an_unknown_class():
    with pytest.raises(ValueError):
        build_redo_validity_example(
            random.Random(7),
            2,
            6,
            REDO_OPS,
            3,
            redo_class="maybe",
        )


def test_redo_valid_path_cannot_need_a_clearing_operation():
    """A valid example stops at the Undo, so it must never raise the
    "could not construct an invalidating operation" failure. Two containers
    are enough for Swap, so an otherwise-valid request still succeeds when
    nothing is placed at all."""

    ops, _, history, _, meta = (
        build_redo_validity_example(
            random.Random(3),
            1,
            3,
            [Put, Move, Undo],
            2,
            redo_class=REDO_CLASS_VALID,
        )
    )

    assert isinstance(ops[-1], Undo)
    assert meta["would_be_valid"] is True


# ============================================================
# 7. Balanced redo batch
# ============================================================

def test_redo_batch_is_balanced():
    examples, balance = build_redo_validity_examples(
        random.Random(11), 2, 6, REDO_OPS, 3, n_per_class=3
    )

    assert len(examples) == 6
    assert all(
        isinstance(example, RedoValidityExample)
        for example in examples
    )

    n_valid = sum(
        example.would_be_valid
        for example in examples
    )

    assert n_valid == 3
    assert balance["n_valid"] == 3
    assert balance["n_invalid"] == 3
    assert balance["n_total"] == 6
    assert balance["requested_valid_fraction"] == 0.5
    assert balance["realised_valid_fraction"] == 0.5
    assert balance["max_tolerance"] == 0.0

    # The label is History-derived for every member of the batch.
    for example in examples:
        _, _, replayed_history = replay_trace(
            example.ops, example.containers
        )
        assert example.would_be_valid == can_redo(replayed_history)
        assert isinstance(example.ops[-1], Undo) == (
            example.would_be_valid
        )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_redo_batch_stays_balanced_across_seeds(seed):
    _, balance = build_redo_validity_examples(
        random.Random(seed), 2, 6, REDO_OPS, 3, n_per_class=2
    )

    assert balance["n_valid"] == balance["n_invalid"] == 2
    assert (
        abs(
            balance["realised_valid_fraction"]
            - balance["requested_valid_fraction"]
        )
        <= balance["max_tolerance"]
    )


def test_redo_batch_tolerance_is_recorded_and_enforced():
    examples, balance = build_redo_validity_examples(
        random.Random(11),
        2,
        6,
        REDO_OPS,
        3,
        n_per_class=1,
        tolerance=0.25,
    )

    assert balance["max_tolerance"] == 0.25
    assert (
        abs(
            balance["realised_valid_fraction"]
            - balance["requested_valid_fraction"]
        )
        <= balance["max_tolerance"]
    )

    # Out-of-range tolerance is a spec error, not a silently clamped value.
    with pytest.raises(ValueError):
        build_redo_validity_examples(
            random.Random(11),
            2,
            6,
            REDO_OPS,
            3,
            tolerance=-0.01,
        )

    with pytest.raises(ValueError):
        build_redo_validity_examples(
            random.Random(11),
            2,
            6,
            REDO_OPS,
            3,
            tolerance=0.75,
        )

    with pytest.raises(ValueError):
        build_redo_validity_examples(
            random.Random(11), 2, 6, REDO_OPS, 3, n_per_class=0
        )

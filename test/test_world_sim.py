"""
Checklist 1 — World simulator unit tests.

Scope: world/operations.py (validity rules + undo/redo bookkeeping) and
world/replay.py (purity, determinism, failure reporting). Everything here is
seed-free and offline: the simulator takes no RNG.

Convention used across the checklist files: a test named ``test_failsnow_*``
states the CORRECT behaviour for a defect recorded in the pre-freeze audit and
therefore fails today. Its docstring names the checklist item, so a failure is
self-documenting rather than mysterious.
"""
from __future__ import annotations

import copy

import pytest

from world import (
    History,
    InvalidOperation,
    Merge,
    Move,
    Put,
    Redo,
    Remove,
    Split,
    Swap,
    Undo,
    WorldState,
    apply_op,
    contents,
    replay_trace,
)

C3 = {"c0", "c1", "c2"}


def fresh():
    return WorldState(object_type={}, location={}, containers=set(C3)), History()


# ---------------------------------------------------------------------------
# Put / Move / Remove
# ---------------------------------------------------------------------------

def test_put_duplicate_object_id_is_invalid():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Put("o0", "pen", "c1"), state, hist)


def test_put_into_unknown_container_is_invalid():
    state, hist = fresh()
    with pytest.raises(InvalidOperation):
        apply_op(Put("o0", "key", "nope"), state, hist)


def test_put_reusing_id_after_remove_is_invalid_because_type_is_kept():
    """Remove keeps object_type (world/operations.py), so a re-Put of the same
    id must stay invalid until the Remove itself is undone."""
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Remove("o0"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Put("o0", "key", "c1"), state, hist)


def test_move_to_current_container_is_invalid():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Move("o0", "c0"), state, hist)


def test_move_of_nonexistent_object_is_invalid():
    state, hist = fresh()
    with pytest.raises(InvalidOperation):
        apply_op(Move("ghost", "c0"), state, hist)


def test_move_of_removed_object_is_invalid():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Remove("o0"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Move("o0", "c1"), state, hist)


def test_remove_then_undo_restores_old_container():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Remove("o0"), state, hist)
    assert "o0" not in state.location
    state = apply_op(Undo(), state, hist)
    assert state.location["o0"] == "c0"


# ---------------------------------------------------------------------------
# Undo / Redo
# ---------------------------------------------------------------------------

def test_undo_with_empty_history_raises():
    state, hist = fresh()
    with pytest.raises(InvalidOperation):
        apply_op(Undo(), state, hist)


def test_redo_with_empty_redo_stack_raises():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Undo(), state, hist)
    hist.redo_stack.clear()
    with pytest.raises(InvalidOperation):
        apply_op(Redo(), state, hist)


def test_undo_right_after_setup_put_erases_object_type_and_allows_reput():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Undo(), state, hist)
    assert "o0" not in state.object_type
    assert "o0" not in state.location
    state = apply_op(Put("o0", "pen", "c1"), state, hist)   # must not raise
    assert state.object_type["o0"] == "pen"


def test_undo_undo_redo_round_trip():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Move("o0", "c1"), state, hist)
    state = apply_op(Move("o0", "c2"), state, hist)
    state = apply_op(Undo(), state, hist)
    assert state.location["o0"] == "c1"
    state = apply_op(Undo(), state, hist)
    assert state.location["o0"] == "c0"
    state = apply_op(Redo(), state, hist)
    assert state.location["o0"] == "c1"   # re-applies the most recent Undo


def test_redo_after_new_op_clears_redo_stack():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Move("o0", "c1"), state, hist)
    state = apply_op(Undo(), state, hist)
    assert hist.redo_stack
    state = apply_op(Move("o0", "c2"), state, hist)
    assert hist.redo_stack == []
    with pytest.raises(InvalidOperation):
        apply_op(Redo(), state, hist)


@pytest.mark.parametrize(
    "ops",
    [
        [Put("o0", "key", "c0"), Split("o0", "o0b"), Move("o0b", "c1"), Undo()],
        [Put("o0", "key", "c0"), Put("o1", "pen", "c1"), Merge("c1", "c0"), Undo()],
        [Put("o0", "key", "c0"), Put("o1", "pen", "c1"), Swap("c0", "c1"), Undo()],
    ],
    ids=["split", "merge", "swap"],
)
def test_undo_of_structural_op_restores_exact_snapshot(ops):
    state, hist = fresh()
    states = [state]
    for op in ops[:-1]:
        state = apply_op(op, state, hist)
        states.append(state)
    snapshot = copy.deepcopy(states[-2].location)   # state before ops[-1]
    restored = apply_op(ops[-1], state, hist)
    assert restored.location == states[-2].location == snapshot


def test_history_depth_after_fifty_ops():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    for i in range(49):
        state = apply_op(Move("o0", "c1" if i % 2 == 0 else "c0"), state, hist)
    assert len(hist.undo_stack) == 50


# ---------------------------------------------------------------------------
# Swap / Merge / Split
# ---------------------------------------------------------------------------

def test_swap_of_two_empty_containers_is_valid_no_op_that_pushes_history():
    state, hist = fresh()
    depth = len(hist.undo_stack)
    state = apply_op(Swap("c0", "c1"), state, hist)
    assert state.location == {}
    assert len(hist.undo_stack) == depth + 1


def test_swap_with_itself_is_invalid():
    state, hist = fresh()
    with pytest.raises(InvalidOperation):
        apply_op(Swap("c0", "c0"), state, hist)


def test_merge_from_empty_source_is_invalid():
    state, hist = fresh()
    with pytest.raises(InvalidOperation):
        apply_op(Merge("c0", "c1"), state, hist)


def test_merge_into_itself_is_invalid():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Merge("c0", "c0"), state, hist)


def test_split_with_existing_new_id_is_invalid():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Put("o1", "key", "c1"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Split("o0", "o1"), state, hist)


def test_split_with_missing_source_is_invalid():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    with pytest.raises(InvalidOperation):
        apply_op(Split("ghost", "o0b"), state, hist)


def test_swap_exchanges_contents_without_clobbering():
    state, hist = fresh()
    state = apply_op(Put("o0", "key", "c0"), state, hist)
    state = apply_op(Put("o1", "pen", "c1"), state, hist)
    state = apply_op(Swap("c0", "c1"), state, hist)
    assert sorted(contents(state, "c0")) == ["o1"]
    assert sorted(contents(state, "c1")) == ["o0"]


# ---------------------------------------------------------------------------
# replay_trace
# ---------------------------------------------------------------------------

def test_replay_trace_does_not_mutate_its_inputs():
    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Undo()]
    before = copy.deepcopy(ops)
    containers = set(C3)
    replay_trace(ops, containers)
    assert ops == before
    assert containers == C3


def test_replay_trace_is_deterministic():
    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Put("o1", "pen", "c1"), Merge("c1", "c2")]
    _, first, _ = replay_trace(ops, C3)
    _, second, _ = replay_trace(ops, C3)
    assert first.location == second.location
    assert first.object_type == second.object_type


def test_replay_trace_is_independent_of_container_set_order():
    ops = [Put("o0", "key", "c0"), Merge("c0", "c1"), Swap("c1", "c2")]
    _, a, _ = replay_trace(ops, {"c0", "c1", "c2"})
    _, b, _ = replay_trace(ops, ["c2", "c0", "c1"])
    assert a.location == b.location


def test_invalid_op_mid_trace_raises_invalid_operation():
    ops = [Put("o0", "key", "c0"), Remove("o0"), Move("o0", "c1")]
    with pytest.raises(InvalidOperation):
        replay_trace(ops, C3)


def test_failsnow_invalid_operation_names_the_offending_index():
    """[checklist 1] 'An invalid op mid-trace raises InvalidOperation and names
    its index.'

    Currently apply_op() raises a bare repr of the rejected operation and
    nothing records which position of the trace was rejected. Counterfactual
    replays (generator.structural, SPEC §3) rely on this exception to decide
    that a structural operation is *necessary*, so an index would make a
    rejected ablation auditable instead of re-derivable only by replaying
    prefixes by hand.
    """
    import re

    ops = [Put("o0", "key", "c0"), Put("o1", "pen", "c1"), Move("ghost", "c2")]
    with pytest.raises(InvalidOperation) as excinfo:
        replay_trace(ops, C3)
    message = str(excinfo.value)
    assert re.search(r"(index|step|position|op)\s*[=:#]?\s*2\b", message, re.IGNORECASE), (
        f"expected the failing trace index (2) named in the message, got: {message}"
    )

from __future__ import annotations

from typing import List, Sequence, Tuple

from world import (
    Merge, Move, Operation, Put, Redo, Remove, Split, Swap, Undo,
    WorldState, replay_trace,
)
from .names import NameRegistry


def _indefinite_article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


# ---------------------------------------------------------------------------
# Operation rendering
# ---------------------------------------------------------------------------

def render_put(
    op: Put,
    before: WorldState,
    names: NameRegistry,
) -> str:
    return (
        f"{_indefinite_article(op.obj_type)} {op.obj_type} "
        f"was placed in {names.container(op.container)}."
    )


def render_move(
    op: Move,
    before: WorldState,
    names: NameRegistry,
) -> str:
    src = before.location[op.obj_id]
    phrase = names.obj(op.obj_id, before)

    return (
        f"{phrase.capitalize()} was moved from "
        f"{names.container(src)} to {names.container(op.dst)}."
    )


def render_remove(
    op: Remove,
    before: WorldState,
    names: NameRegistry,
) -> str:
    src = before.location[op.obj_id]
    phrase = names.obj(op.obj_id, before)

    return (
        f"{phrase.capitalize()} was taken out of "
        f"{names.container(src)}."
    )


def render_undo(
    op: Undo,
    before: WorldState,
    names: NameRegistry,
) -> str:
    return "That last action was undone."


def render_redo(
    op: Redo,
    before: WorldState,
    names: NameRegistry,
) -> str:
    return "The undone action was redone."


def render_split(
    op: Split,
    before: WorldState,
    names: NameRegistry,
) -> str:
    container = before.location[op.source_obj_id]
    phrase = names.obj(op.source_obj_id, before)

    return (
        f"{phrase.capitalize()} in {names.container(container)} "
        f"split into two identical copies."
    )


def render_merge(
    op: Merge,
    before: WorldState,
    names: NameRegistry,
) -> str:
    return (
        f"Everything in {names.container(op.src_container)} "
        f"was moved into {names.container(op.dst_container)}."
    )


def render_swap(
    op: Swap,
    before: WorldState,
    names: NameRegistry,
) -> str:
    return (
        f"The contents of {names.container(op.container_a)} and "
        f"{names.container(op.container_b)} were swapped."
    )


RENDER_DISPATCH = {
    Put: render_put,
    Move: render_move,
    Remove: render_remove,
    Undo: render_undo,
    Redo: render_redo,
    Split: render_split,
    Merge: render_merge,
    Swap: render_swap,
}


def render_narrative(
    ops: Sequence[Operation],
    containers,
    names: NameRegistry,
) -> Tuple[List[str], WorldState]:
    """Render the canonical operation trace.

    The same replay_trace() used for gold-answer generation is used here,
    ensuring that the natural-language narrative and simulator state cannot
    silently diverge.
    """
    trace, final_state, _ = replay_trace(ops, containers)

    sentences = [
        RENDER_DISPATCH[type(op)](op, before, names)
        for op, before, _after in trace
    ]

    return sentences, final_state


# ---------------------------------------------------------------------------
# Question rendering
# ---------------------------------------------------------------------------

def question_location(
    obj_id: str,
    state: WorldState,
    names: NameRegistry,
) -> str:
    """Generate a location question for an object."""
    return f"Where is {names.obj(obj_id, state)} now?"


def question_count(
    container_id: str,
    obj_type: str,
    names: NameRegistry,
) -> str:
    """Generate a count question for a container/type pair."""
    return (
        f"How many {obj_type}s are in "
        f"{names.container(container_id)} now?"
    )


def question_redo_validity() -> str:
    """Generate the redo-validity probe question."""
    return (
        "If someone tried to redo the last undone action right now, "
        "would that succeed?"
    )


def question_counterfactual(
    removed_sentence: str,
    obj_id: str,
    state: WorldState,
) -> str:
    """Generate a counterfactual location question."""
    obj_type = state.object_type[obj_id]

    return (
        f'Suppose this had not happened: "{removed_sentence}" '
        f"Where would the {obj_type} be now?"
    )
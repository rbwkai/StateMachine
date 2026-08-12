from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from world import (
    Merge, Move, Operation, Put, Redo, Remove, Split, Swap, Undo,
    WorldState, replay_trace,
)
from .names import NameRegistry, OBJECT_TYPES

# ---------------------------------------------------------------------------
# One render function per operation type. Each takes (op, state_before,
# names) and returns a single natural-language sentence. state_before is
# needed by Move/Remove/Split to describe *where something was*, and comes
# from the shared replay_trace pass (see render_narrative below) so
# narration can never disagree with the gold-answer computation.
# ---------------------------------------------------------------------------

def _indefinite_article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


def render_put(op: Put, before: WorldState, names: NameRegistry) -> str:
    article = _indefinite_article(op.obj_type)
    return f"{article} {op.obj_type} was placed in {names.container(op.container)}."


def render_move(op: Move, before: WorldState, names: NameRegistry) -> str:
    src = before.location[op.obj_id]
    phrase = names.obj(op.obj_id, before)
    return f"{phrase.capitalize()} was moved from {names.container(src)} to {names.container(op.dst)}."


def render_remove(op: Remove, before: WorldState, names: NameRegistry) -> str:
    src = before.location[op.obj_id]
    phrase = names.obj(op.obj_id, before)
    return f"{phrase.capitalize()} was taken out of {names.container(src)}."


def render_undo(op: Undo, before: WorldState, names: NameRegistry) -> str:
    return "That last action was undone."


def render_redo(op: Redo, before: WorldState, names: NameRegistry) -> str:
    return "The undone action was redone."


def render_split(op: Split, before: WorldState, names: NameRegistry) -> str:
    container = before.location[op.source_obj_id]
    phrase = names.obj(op.source_obj_id, before)
    return f"{phrase.capitalize()} in {names.container(container)} split into two identical copies."


def render_merge(op: Merge, before: WorldState, names: NameRegistry) -> str:
    return (
        f"Everything in {names.container(op.src_container)} "
        f"was moved into {names.container(op.dst_container)}."
    )


def render_swap(op: Swap, before: WorldState, names: NameRegistry) -> str:
    return (
        f"The contents of {names.container(op.container_a)} "
        f"and {names.container(op.container_b)} were swapped."
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
    """Renders one sentence per operation. Returns (sentences, final_state).
    Uses the same replay_trace() as gold-answer computation, so the text
    the model reads and the gold answer it's scored against are always
    derived from one consistent pass over the same operation list.
    """
    trace, final_state, _ = replay_trace(ops, containers)
    sentences = [
        RENDER_DISPATCH[type(op)](op, before, names)
        for op, before, after in trace
    ]
    return sentences, final_state


# ---------------------------------------------------------------------------
# Distractors: text only, never touch WorldState.
# ---------------------------------------------------------------------------

DISTRACTOR_FLAVOR = [
    "has a faint smell of cedar",
    "creaks when opened",
    "was recently dusted",
    "sits near a window",
    "was a gift from a relative",
    "has a small scratch on one side",
    "is slightly heavier than it looks",
    "was bought many years ago",
]


def make_distractor_sentences(
    rng: random.Random,
    n: int,
    names: NameRegistry,
    used_object_types: Sequence[str],
) -> List[str]:
    sentences = []
    all_containers = list(names.container_names.values())
    unrelated_types = [t for t in OBJECT_TYPES if t not in used_object_types] or OBJECT_TYPES
    for _ in range(n):
        if rng.random() < 0.5 and all_containers:
            container_name = rng.choice(all_containers)
            flavor = rng.choice(DISTRACTOR_FLAVOR)
            sentences.append(f"{container_name.capitalize()} {flavor}.")
        else:
            t = rng.choice(unrelated_types)
            sentences.append(f"Someone mentioned that {t}s have become harder to find lately.")
    return sentences


def splice_distractors(
    rng: random.Random,
    op_sentences: Sequence[str],
    distractor_sentences: Sequence[str],
) -> List[str]:
    """Interleaves distractor sentences at random positions among the
    operation sentences. Distractors never get an entry in ops_applied,
    so gold-answer computation is completely unaffected by N."""
    combined = list(op_sentences)
    for sentence in distractor_sentences:
        pos = rng.randint(0, len(combined))
        combined.insert(pos, sentence)
    return combined


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def question_location(obj_id: str, state: WorldState, names: NameRegistry) -> str:
    obj_type = state.object_type[obj_id]
    return f"Where is the {obj_type} now?"


def question_count(container_id: str, obj_type: str, names: NameRegistry) -> str:
    return f"How many {obj_type}s are in {names.container(container_id)} now?"


def question_redo_validity() -> str:
    return "If someone tried to redo the last undone action right now, would that succeed?"


def question_counterfactual(removed_sentence: str, obj_id: str, state: WorldState) -> str:
    obj_type = state.object_type[obj_id]
    return (
        f'Suppose this had not happened: "{removed_sentence}" '
        f"Where would the {obj_type} be now?"
    )
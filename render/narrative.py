from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from world import (
    Merge, Move, Operation, Put, Redo, Remove, Split, Swap, Undo,
    WorldState, replay_trace,
)
from .names import UNDO_SENTENCE, NameRegistry, pluralize_object


def _indefinite_article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


# ---------------------------------------------------------------------------
# Operation rendering
# ---------------------------------------------------------------------------

def render_put(
    op: Put,
    before: WorldState,
    names: NameRegistry,
    shared_rank: int = 0,
) -> str:
    """Render a Put.

    ``shared_rank`` is the 1-based creation rank of this object among the
    objects of its type, and 0 when the type is not shared. For a shared type
    the bare "A key was placed ..." leaves the reader unable to resolve the
    later references to "the original" / "the duplicate" / "the 3rd key", so the
    sentence names the object with the same phrase obj() will use from then on.
    The "<word> was placed in <container>." shape is preserved because
    eval.robustness.PARAPHRASE_RULES matches on it.
    """
    if not shared_rank:
        return (
            f"{_indefinite_article(op.obj_type)} {op.obj_type} "
            f"was placed in {names.container(op.container)}."
        )
    # "duplicate" on its own does not say the object is the second one, so the
    # subject carries the ordinal as well.
    subject = (
        f"second {op.obj_type}" if shared_rank == 2 else op.obj_type
    )
    return (
        f"{_indefinite_article(op.obj_type)} {subject} "
        f"was placed in {names.container(op.container)} "
        f"as {names.obj_at(op.obj_type, shared_rank)}."
    )


def render_move(
    op: Move,
    before: WorldState,
    names: NameRegistry,
    include_source: bool = True,
) -> str:
    phrase = names.obj(op.obj_id, before)
    if not include_source:
        return f"{phrase.capitalize()} was moved to {names.container(op.dst)}."
    src = before.location[op.obj_id]

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
    return UNDO_SENTENCE


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
    obj_type = before.object_type[op.source_obj_id]
    phrase = names.obj(op.source_obj_id, before)

    # Split leaves the source in place and adds an identical object of the same
    # type in the same container, so the copy is narrated as a placement. That
    # keeps one sentence template (and one paraphrase rule) for every op that
    # puts an object somewhere.
    return (
        f"{_indefinite_article(obj_type)} {obj_type} was placed in "
        f"{names.container(container)} as an identical copy of {phrase}."
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


def _creation_ranks(trace) -> Tuple[Dict[str, int], Set[str]]:
    """1-based creation rank per object, and the types that are ever shared.

    Put and Split are the only operations that introduce an object, so the
    trace order is the creation order the reader sees. Kept separate from
    NameRegistry.obj() so a Put can name its object before the state contains
    it. An Undo can retire an object that a later Put reintroduces, so an id
    already in the per-type list is re-inserted rather than counted twice.

    Returns (ranks, shared_types); a type is shared once a second object of it
    has ever existed, which is the point at which the narrative starts saying
    "the original" / "the duplicate" / "the 3rd key".
    """
    ranks: Dict[str, int] = {}
    per_type: Dict[str, List[str]] = {}
    shared: Set[str] = set()
    for op, before, _after in trace:
        if isinstance(op, Put):
            obj_id, obj_type = op.obj_id, op.obj_type
        elif isinstance(op, Split):
            obj_id = op.new_obj_id
            obj_type = before.object_type[op.source_obj_id]
        else:
            continue
        live = [oid for oid in per_type.get(obj_type, []) if oid != obj_id]
        live.append(obj_id)
        per_type[obj_type] = live
        ranks[obj_id] = len(live)
        if len(live) > 1:
            shared.add(obj_type)
    return ranks, shared


def render_narrative(
    ops: Sequence[Operation],
    containers,
    names: NameRegistry,
    include_move_sources: bool = False,
) -> Tuple[List[str], WorldState]:
    """Render the canonical operation trace.

    The same replay_trace() used for gold-answer generation is used here,
    ensuring that the natural-language narrative and simulator state cannot
    silently diverge.
    """
    trace, final_state, _ = replay_trace(ops, containers)

    # Creation rank per object, in the order the reader meets them. Put renders
    # the rank so "the original key" / "the duplicate key" resolve to a specific
    # object on first mention instead of appearing unexplained later.
    creation_rank, shared_types = _creation_ranks(trace)

    sentences = []
    for op, before, _after in trace:
        if isinstance(op, Move):
            sentences.append(
                render_move(
                    op,
                    before,
                    names,
                    include_source=include_move_sources,
                )
            )
        elif isinstance(op, Put) and op.obj_type in shared_types:
            sentences.append(
                render_put(
                    op,
                    before,
                    names,
                    shared_rank=creation_rank[op.obj_id],
                )
            )
        else:
            sentences.append(RENDER_DISPATCH[type(op)](op, before, names))

    return sentences, final_state


# ---------------------------------------------------------------------------
# Question rendering
# ---------------------------------------------------------------------------

def question_location(obj_id: str, state: WorldState, names: NameRegistry) -> str:
    return f"Where is {names.obj(obj_id, state)} now?"


def question_count(container_id: str, obj_type: str, names: NameRegistry) -> str:
    return (
        f"How many {pluralize_object(obj_type)} are in "
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
    obj_type = state.object_type[obj_id]
    return (
        f'Suppose this had not happened: "{removed_sentence}" '
        f"Where would the {obj_type} be now?"
    )

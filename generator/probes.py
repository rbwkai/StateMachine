from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from world import (
    GenerationError, History, InvalidOperation, Merge, Move, Operation, Put,
    Redo, Remove, Split, Swap, Undo, WorldState, apply_op, gold_count,
    gold_location, replay_trace,
)
from .sampler import _construct_move, _construct_swap, sample_sequence


@dataclass
class LocationQuery:
    obj_id: str

    def read(self, state: WorldState):
        return gold_location(state, self.obj_id)


@dataclass
class CountQuery:
    container: str
    obj_type: str

    def read(self, state: WorldState):
        return gold_count(state, self.container, self.obj_type)


Query = Union[LocationQuery, CountQuery]


def select_query(rng: random.Random, ops_applied: Sequence[Operation], final_state: WorldState) -> Query:
    """Chooses location vs. count query based on whether the trajectory
    contains a Split/Merge -- these make single-object location questions
    ambiguous (which copy do you mean?), so count queries are used instead
    whenever they occur."""
    has_split_or_merge = any(isinstance(op, (Split, Merge)) for op in ops_applied)
    if has_split_or_merge:
        present_types = sorted({
            final_state.object_type[oid] for oid in final_state.location
        })
        if present_types:
            obj_type = rng.choice(present_types)
            container = rng.choice(sorted(final_state.containers))
            return CountQuery(container, obj_type)
        # fallback: nothing is present anywhere (rare, e.g. everything removed)
    present_objs = sorted(final_state.location)
    if not present_objs:
        # last resort: query an object that existed even if it was removed
        all_objs = sorted(final_state.object_type)
        if not all_objs:
            raise GenerationError(
                "no entity was ever created in this example (PUT likely "
                "excluded from operations_enabled) -- nothing to query"
            )
        return LocationQuery(rng.choice(all_objs))

    # Prefer an object that actually appears in multiple operations, so the
    # narrative and step-wise trajectory are informative rather than mostly
    # nulls (e.g. an object PUT in the very last step and never touched
    # again). Ties broken randomly for variety across examples.
    touch_counts: Dict[str, int] = {oid: 0 for oid in present_objs}
    for op in ops_applied:
        oid = getattr(op, "obj_id", None) or getattr(op, "source_obj_id", None)
        if oid in touch_counts:
            touch_counts[oid] += 1
    max_touches = max(touch_counts.values())
    best_objs = [oid for oid, c in touch_counts.items() if c == max_touches]
    return LocationQuery(rng.choice(best_objs))


def step_wise_gold(
    ops_applied: Sequence[Operation],
    containers: Set[str],
    query: Query,
) -> List:
    trace, _, _ = replay_trace(ops_applied, containers)
    return [query.read(after) for _, _before, after in trace]


def counterfactual_gold(
    ops_applied: Sequence[Operation],
    containers: Set[str],
    remove_index: int,
    query: Query,
) -> Optional[object]:
    """Replays the sequence with one step removed. Returns None (probe
    dropped) if removing that step makes a later step invalid, rather
    than emitting a broken example."""
    reduced = list(ops_applied[:remove_index]) + list(ops_applied[remove_index + 1:])
    try:
        _, final_state, _ = replay_trace(reduced, containers)
    except InvalidOperation:
        return None
    return query.read(final_state)


def build_counterfactual_probes(
    rng: random.Random,
    ops_applied: Sequence[Operation],
    containers: Set[str],
    query: Query,
    max_probes: int = 2,
) -> List[Dict]:
    """Samples a few step indices to counterfactually remove and reports
    their gold answers, skipping any that turn out to invalidate a later
    step."""
    candidate_indices = list(range(len(ops_applied)))
    rng.shuffle(candidate_indices)
    probes = []
    for idx in candidate_indices:
        if len(probes) >= max_probes:
            break
        answer = counterfactual_gold(ops_applied, containers, idx, query)
        if answer is None:
            continue
        probes.append({"remove_step": idx, "gold_answer": answer})
    return probes


def build_redo_validity_example(
    rng: random.Random,
    entity_count: int,
    update_count: int,
    operations_enabled: Sequence[type],
    num_containers: Optional[int] = None,
) -> Tuple[List[Operation], WorldState, History, Set[str], Dict]:
    """Forces the subsequence [..., <op>, Undo, <new op>] so that a final
    Redo attempt (not applied -- it's the thing being asked about) is
    deliberately invalid. This subsequence essentially never occurs by
    chance under normal random sampling, so it needs its own generation
    mode rather than being left to show up organically.
    """
    base_update_count = max(update_count - 3, 2)
    ops, state, history, containers = sample_sequence(
        rng, entity_count, base_update_count, operations_enabled, num_containers
    )

    setup_op = _construct_move(rng, state) or _construct_swap(rng, state)
    if setup_op is None:
        raise RuntimeError("could not construct a setup op for redo-validity probe")
    state = apply_op(setup_op, state, history)
    ops.append(setup_op)

    undo_op = Undo()
    state = apply_op(undo_op, state, history)
    ops.append(undo_op)

    new_action = _construct_move(rng, state) or _construct_swap(rng, state)
    if new_action is None:
        raise RuntimeError("could not construct an invalidating op for redo-validity probe")
    state = apply_op(new_action, state, history)
    ops.append(new_action)

    from world import can_redo
    would_be_valid = can_redo(history)  # expected False
    return ops, state, history, containers, {"would_be_valid": would_be_valid}
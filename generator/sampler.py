from __future__ import annotations

import random
from typing import List, Optional, Sequence, Set, Tuple

from render.names import OBJECT_TYPES
from world import (
    GenerationError, History, InvalidOperation, Merge, Move, Operation,
    Put, Redo, Remove, Split, Swap, Undo, WorldState, apply_op,
    can_redo, can_undo, contents,
)

DEFAULT_WEIGHTS = {
    Put: 3.0,
    Move: 4.0,
    Remove: 1.0,
    Undo: 1.0,
    Redo: 0.5,
    Split: 1.0,
    Merge: 1.0,
    Swap: 1.0,
}


# ---------------------------------------------------------------------------
# Per-type constructors. Each samples a *concrete, currently-valid*
# instance of its operation from the current state, rather than sampling
# blind and hoping is_valid() passes -- this keeps rejection rates low
# even with 8 operation types competing for the same world.
# Returns None if no valid instance can currently be constructed.
# ---------------------------------------------------------------------------

def _construct_put(rng: random.Random, state: WorldState, next_obj_idx: int) -> Optional[Put]:
    obj_id = f"o{next_obj_idx}"
    obj_type = rng.choice(OBJECT_TYPES)
    container = rng.choice(sorted(state.containers))
    return Put(obj_id, obj_type, container)


def _construct_move(rng: random.Random, state: WorldState) -> Optional[Move]:
    if not state.location:
        return None
    obj_id = rng.choice(sorted(state.location))
    candidates = sorted(state.containers - {state.location[obj_id]})
    if not candidates:
        return None
    return Move(obj_id, rng.choice(candidates))


def _construct_remove(rng: random.Random, state: WorldState) -> Optional[Remove]:
    if not state.location:
        return None
    return Remove(rng.choice(sorted(state.location)))


def _construct_split(rng: random.Random, state: WorldState, next_obj_idx: int) -> Optional[Split]:
    if not state.location:
        return None
    source = rng.choice(sorted(state.location))
    return Split(source, f"o{next_obj_idx}")


def _construct_merge(rng: random.Random, state: WorldState) -> Optional[Merge]:
    nonempty = [c for c in state.containers if contents(state, c)]
    if not nonempty:
        return None
    src = rng.choice(nonempty)
    candidates = sorted(state.containers - {src})
    if not candidates:
        return None
    return Merge(src, rng.choice(candidates))


def _construct_swap(rng: random.Random, state: WorldState) -> Optional[Swap]:
    if len(state.containers) < 2:
        return None
    a, b = rng.sample(sorted(state.containers), 2)
    return Swap(a, b)


_CONSTRUCTORS = {
    Put: lambda rng, state, next_idx: _construct_put(rng, state, next_idx),
    Move: lambda rng, state, next_idx: _construct_move(rng, state),
    Remove: lambda rng, state, next_idx: _construct_remove(rng, state),
    Split: lambda rng, state, next_idx: _construct_split(rng, state, next_idx),
    Merge: lambda rng, state, next_idx: _construct_merge(rng, state),
    Swap: lambda rng, state, next_idx: _construct_swap(rng, state),
}


def _feasible_types(
    state: WorldState,
    history: History,
    objects_created: int,
    entity_cap: int,
    operations_enabled: Sequence[type],
) -> List[type]:
    feasible = []
    for op_type in operations_enabled:
        if op_type is Put or op_type is Split:
            if objects_created >= entity_cap:
                continue
            if op_type is Split and not state.location:
                continue
        if op_type is Move or op_type is Remove:
            if not state.location:
                continue
        if op_type is Undo and not can_undo(history):
            continue
        if op_type is Redo and not can_redo(history):
            continue
        if op_type is Merge and not any(contents(state, c) for c in state.containers):
            continue
        if op_type is Swap and len(state.containers) < 2:
            continue
        feasible.append(op_type)
    return feasible


def _propose_operation(
    rng: random.Random,
    state: WorldState,
    history: History,
    objects_created: int,
    entity_cap: int,
    operations_enabled: Sequence[type],
) -> Optional[Operation]:
    feasible = _feasible_types(state, history, objects_created, entity_cap, operations_enabled)
    if not feasible:
        return None
    weights = [DEFAULT_WEIGHTS.get(t, 1.0) for t in feasible]
    op_type = rng.choices(feasible, weights=weights, k=1)[0]
    if op_type is Undo:
        return Undo()
    if op_type is Redo:
        return Redo()
    constructor = _CONSTRUCTORS[op_type]
    return constructor(rng, state, objects_created)


def sample_sequence(
    rng: random.Random,
    entity_count: int,
    update_count: int,
    operations_enabled: Sequence[type],
    num_containers: Optional[int] = None,
    max_attempts_factor: int = 25,
) -> Tuple[List[Operation], WorldState, History, Set[str]]:
    """Samples a sequence of `update_count` valid operations under the
    given (E, U) budget and enabled operation vocabulary.

    Returns (ops_applied, final_state, final_history, containers).
    Raises GenerationError if the budget can't be met (e.g. U requested
    higher than what's reachable given entity_count and container count).
    """
    num_containers = num_containers or max(2, entity_count // 2 + 1)
    containers = {f"c{i}" for i in range(num_containers)}
    state = WorldState(object_type={}, location={}, containers=containers)
    history = History()
    ops_applied: List[Operation] = []
    objects_created = 0

    # Guarantee at least one entity exists, mirroring the proposal's own
    # "initial valid state, then updates" structure. Without this, pure
    # random sampling can (rarely, but observed under stress-testing)
    # produce an all-Swap/Undo sequence on a world that never has any
    # object in it, which leaves every possible query with nothing to ask
    # about.
    if Put in operations_enabled and update_count >= 1:
        first_put = _construct_put(rng, state, objects_created)
        state = apply_op(first_put, state, history)
        ops_applied.append(first_put)
        objects_created += 1

    budget = update_count * max_attempts_factor
    while len(ops_applied) < update_count and budget > 0:
        budget -= 1
        op = _propose_operation(
            rng, state, history, objects_created, entity_count, operations_enabled
        )
        if op is None:
            continue
        try:
            state = apply_op(op, state, history)
        except InvalidOperation:
            continue
        ops_applied.append(op)
        if isinstance(op, (Put, Split)):
            objects_created += 1

    if len(ops_applied) < update_count:
        raise GenerationError(
            f"could not reach U={update_count} within budget "
            f"(reached {len(ops_applied)}; try more containers or a higher entity_count)"
        )

    return ops_applied, state, history, containers

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Set, Tuple

from render.names import OBJECT_TYPES
from world import (
    GenerationError, History, Merge, Move, Operation, Put, Redo, Remove,
    Split, Swap, Undo, WorldState, apply_op, can_redo, can_undo, contents,
)

# Weights are used only for diversity inside an explicitly selected
# operation family. They are NOT the experimental difficulty controller.
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


def _construct_put(rng, state, next_obj_idx):
    return Put(
        f"o{next_obj_idx}",
        rng.choice(OBJECT_TYPES),
        rng.choice(sorted(state.containers)),
    )


def _construct_move(rng, state):
    if not state.location:
        return None
    oid = rng.choice(sorted(state.location))
    candidates = sorted(state.containers - {state.location[oid]})
    return Move(oid, rng.choice(candidates)) if candidates else None


def _construct_remove(rng, state):
    if not state.location:
        return None
    return Remove(rng.choice(sorted(state.location)))


def _construct_split(rng, state, next_obj_idx):
    if not state.location:
        return None
    oid = rng.choice(sorted(state.location))
    return Split(oid, f"o{next_obj_idx}")


def _construct_merge(rng, state):
    nonempty = [c for c in state.containers if contents(state, c)]
    if not nonempty:
        return None
    src = rng.choice(sorted(nonempty))
    candidates = sorted(state.containers - {src})
    return Merge(src, rng.choice(candidates)) if candidates else None


def _construct_swap(rng, state):
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
        if op_type in (Put, Split) and objects_created >= entity_cap:
            continue
        if op_type in (Move, Remove, Split) and not state.location:
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
    rng,
    state,
    history,
    objects_created,
    entity_cap,
    operations_enabled,
):
    feasible = _feasible_types(
        state, history, objects_created, entity_cap, operations_enabled
    )
    if not feasible:
        return None

    weights = [DEFAULT_WEIGHTS.get(t, 1.0) for t in feasible]
    op_type = rng.choices(feasible, weights=weights, k=1)[0]

    if op_type is Undo:
        return Undo()
    if op_type is Redo:
        return Redo()
    return _CONSTRUCTORS[op_type](rng, state, objects_created)


def sample_sequence(
    rng: random.Random,
    entity_count: int,
    update_count: int,
    operations_enabled: Sequence[type],
    num_containers: Optional[int] = None,
    max_attempts_factor: int = 50,
) -> Tuple[List[Operation], WorldState, History, Set[str]]:
    """Generate one valid canonical trajectory.

    This function is intentionally agnostic about query difficulty. The
    caller should repeatedly sample trajectories and use QuerySpec /
    analyze_trajectory to accept only instances satisfying the intended
    experimental condition.
    """
    if update_count < 1:
        raise GenerationError("update_count must be >= 1")

    num_containers = num_containers or max(2, entity_count // 2 + 1)
    containers = {f"c{i}" for i in range(num_containers)}
    state = WorldState({}, {}, containers, 0)
    history = History()
    ops: List[Operation] = []
    objects_created = 0

    if Put in operations_enabled:
        first = _construct_put(rng, state, objects_created)
        state = apply_op(first, state, history)
        ops.append(first)
        objects_created += 1

    budget = update_count * max_attempts_factor
    while len(ops) < update_count and budget > 0:
        budget -= 1
        op = _propose_operation(
            rng, state, history, objects_created, entity_count, operations_enabled
        )
        if op is None:
            continue
        try:
            state = apply_op(op, state, history)
        except Exception:
            continue
        ops.append(op)
        if isinstance(op, (Put, Split)):
            objects_created += 1

    if len(ops) < update_count:
        raise GenerationError(
            f"could not reach D={update_count}; reached {len(ops)}"
        )

    return ops, state, history, containers

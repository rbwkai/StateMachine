
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
from .analysis import QuerySpec, analyze_trajectory


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


@dataclass
class RedoValidityQuery:
    def read(self, state: WorldState):
        # This query is handled through the history object in the pipeline.
        raise NotImplementedError


Query = Union[LocationQuery, CountQuery]


def candidate_queries(
    rng: random.Random,
    ops_applied: Sequence[Operation],
    final_state: WorldState,
    query_type: str = "location",
) -> List[Query]:
    """Return candidate queries; validation decides which one is acceptable."""
    if query_type == "location":
        candidates = sorted(final_state.object_type)
        rng.shuffle(candidates)
        return [LocationQuery(oid) for oid in candidates]

    if query_type == "count":
        types = sorted(set(final_state.object_type.values()))
        containers = sorted(final_state.containers)
        pairs = [(c, t) for c in containers for t in types]
        rng.shuffle(pairs)
        return [CountQuery(c, t) for c, t in pairs]

    raise GenerationError(f"unknown query_type={query_type!r}")


def select_query(
    rng: random.Random,
    ops_applied: Sequence[Operation],
    final_state: WorldState,
    containers: Set[str],
    query_spec: QuerySpec,
) -> Tuple[Query, object]:
    """Select a query by satisfying a formal QuerySpec, not by heuristic
    'interestingness'.

    Every candidate is analyzed against the same canonical trajectory.
    """
    candidates = candidate_queries(
        rng, ops_applied, final_state, query_spec.query_type
    )

    for query in candidates:
        analysis = analyze_trajectory(ops_applied, containers, query)
        if query_spec.matches(analysis):
            return query, analysis

    raise GenerationError(
        f"no query satisfied QuerySpec={query_spec!r} for sampled trajectory"
    )


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
):
    """Construct a history-specific condition:

        ... A, Undo, B

    Therefore the redo stack must be cleared and redo must be invalid.
    """
    if update_count < 3:
        raise GenerationError("redo validity requires update_count >= 3")

    base = max(update_count - 3, 1)
    ops, state, history, containers = sample_sequence(
        rng, entity_count, base, operations_enabled, num_containers
    )

    setup = _construct_move(rng, state) or _construct_swap(rng, state)
    if setup is None:
        raise GenerationError("could not construct setup action for redo probe")

    state = apply_op(setup, state, history)
    ops.append(setup)

    undo = Undo()
    state = apply_op(undo, state, history)
    ops.append(undo)

    new_action = _construct_move(rng, state) or _construct_swap(rng, state)
    if new_action is None:
        raise GenerationError("could not construct post-undo action")

    state = apply_op(new_action, state, history)
    ops.append(new_action)

    from world import can_redo
    return ops, state, history, containers, {"would_be_valid": can_redo(history)}

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from world import (
    GenerationError,
    History,
    InvalidOperation,
    Merge,
    Move,
    Operation,
    Put,
    Redo,
    Remove,
    Split,
    Swap,
    Undo,
    WorldState,
    apply_op,
    gold_count,
    gold_location,
    replay_trace,
)

from .sampler import (
    _construct_move,
    _construct_swap,
    sample_sequence,
)

from analysis import QuerySpec, analyze_trajectory


# ============================================================
# Query definitions
# ============================================================

@dataclass
class LocationQuery:
    obj_id: str

    def read(self, state: WorldState):
        return gold_location(
            state,
            self.obj_id,
        )


@dataclass
class CountQuery:
    container: str
    obj_type: str

    def read(self, state: WorldState):
        return gold_count(
            state,
            self.container,
            self.obj_type,
        )


@dataclass
class RedoValidityQuery:
    """
    Marker query for redo-validity probes.

    Redo validity itself is evaluated using the History object,
    so this query is not used by normal trajectory generation.
    """

    def read(self, state: WorldState):
        raise NotImplementedError(
            "RedoValidityQuery is evaluated through History"
        )


Query = Union[
    LocationQuery,
    CountQuery,
]


# ============================================================
# Candidate query generation
# ============================================================

def candidate_queries(
    rng: random.Random,
    ops_applied: Sequence[Operation],
    final_state: WorldState,
    query_type: str = "location",
) -> List[Query]:
    """
    Generate candidate queries for a sampled final state.

    QuerySpec validation is performed separately by select_query().
    """

    if query_type == "location":

        candidates = sorted(
            final_state.object_type
        )

        rng.shuffle(candidates)

        return [
            LocationQuery(obj_id)
            for obj_id in candidates
        ]

    if query_type == "count":

        types = sorted(
            set(
                final_state.object_type.values()
            )
        )

        containers = sorted(
            final_state.containers
        )

        pairs = [
            (container, obj_type)
            for container in containers
            for obj_type in types
        ]

        rng.shuffle(pairs)

        return [
            CountQuery(
                container,
                obj_type,
            )
            for container, obj_type in pairs
        ]

    raise GenerationError(
        f"unknown query_type={query_type!r}"
    )


# ============================================================
# Query selection
# ============================================================

def select_query(
    rng: random.Random,
    ops_applied: Sequence[Operation],
    final_state: WorldState,
    containers: Set[str],
    query_spec: QuerySpec,
) -> Tuple[Query, object]:
    """
    Select a query satisfying the supplied QuerySpec.

    Every candidate is evaluated against the canonical symbolic
    trajectory. This makes difficulty a property of the generated
    trajectory rather than an arbitrary query heuristic.
    """

    candidates = candidate_queries(
        rng,
        ops_applied,
        final_state,
        query_spec.query_type,
    )

    for query in candidates:

        analysis = analyze_trajectory(
            ops_applied,
            containers,
            query,
        )

        if query_spec.matches(analysis):
            return query, analysis

    raise GenerationError(
        "no query satisfied "
        f"QuerySpec={query_spec!r} "
        "for sampled trajectory"
    )


# ============================================================
# Step-wise gold
# ============================================================

def step_wise_gold(
    ops_applied: Sequence[Operation],
    containers: Set[str],
    query: Query,
) -> List:
    """
    Return the answer to the query after every operation.

    This function remains available for compatibility. The richer
    trajectory representation is constructed in trajectory.py.
    """

    trace, _, _ = replay_trace(
        ops_applied,
        containers,
    )

    return [
        query.read(after)
        for _, _before, after in trace
    ]


# ============================================================
# Counterfactual probes
# ============================================================

def counterfactual_gold(
    ops_applied: Sequence[Operation],
    containers: Set[str],
    remove_index: int,
    query: Query,
) -> Optional[object]:
    """
    Replay the sequence with one operation removed.

    Returns None if removing the operation makes a later operation
    invalid.
    """

    reduced = (
        list(ops_applied[:remove_index])
        + list(ops_applied[remove_index + 1:])
    )

    try:

        _, final_state, _ = replay_trace(
            reduced,
            containers,
        )

    except InvalidOperation:
        return None

    return query.read(
        final_state
    )


def build_counterfactual_probes(
    rng: random.Random,
    ops_applied: Sequence[Operation],
    containers: Set[str],
    query: Query,
    max_probes: int = 2,
) -> List[Dict]:
    """
    Sample operation removals and compute the resulting gold answer.

    Invalid removals are discarded rather than producing broken
    benchmark instances.
    """

    candidate_indices = list(
        range(len(ops_applied))
    )

    rng.shuffle(candidate_indices)

    probes: List[Dict] = []

    for idx in candidate_indices:

        if len(probes) >= max_probes:
            break

        answer = counterfactual_gold(
            ops_applied,
            containers,
            idx,
            query,
        )

        if answer is None:
            continue

        probes.append(
            {
                "remove_step": idx,
                "gold_answer": answer,
            }
        )

    return probes


# ============================================================
# Redo-validity probe
# ============================================================

def build_redo_validity_example(
    rng: random.Random,
    entity_count: int,
    update_count: int,
    operations_enabled: Sequence[type],
    num_containers: Optional[int] = None,
) -> Tuple[
    List[Operation],
    WorldState,
    History,
    Set[str],
    Dict,
]:
    """
    Construct a trajectory ending in:

        ... -> operation -> Undo -> new operation

    The new operation invalidates the redo history, allowing the
    benchmark to ask whether the undone operation could be redone.

    The Redo itself is NOT applied. Its validity is the query.
    """

    base_update_count = max(
        update_count - 3,
        2,
    )

    (
        ops,
        state,
        history,
        containers,
    ) = sample_sequence(
        rng,
        entity_count,
        base_update_count,
        operations_enabled,
        num_containers,
    )

    # --------------------------------------------------------
    # Establish an operation that can subsequently be undone.
    # --------------------------------------------------------

    setup_op = (
        _construct_move(rng, state)
        or _construct_swap(rng, state)
    )

    if setup_op is None:
        raise RuntimeError(
            "could not construct a setup operation "
            "for redo-validity probe"
        )

    state = apply_op(
        setup_op,
        state,
        history,
    )

    ops.append(
        setup_op
    )

    # --------------------------------------------------------
    # Undo the setup operation.
    # --------------------------------------------------------

    undo_op = Undo()

    state = apply_op(
        undo_op,
        state,
        history,
    )

    ops.append(
        undo_op
    )

    # --------------------------------------------------------
    # Perform a new operation.
    #
    # This invalidates the redo history.
    # --------------------------------------------------------

    new_action = (
        _construct_move(rng, state)
        or _construct_swap(rng, state)
    )

    if new_action is None:
        raise RuntimeError(
            "could not construct an invalidating operation "
            "for redo-validity probe"
        )

    state = apply_op(
        new_action,
        state,
        history,
    )

    ops.append(
        new_action
    )

    # --------------------------------------------------------
    # Test whether redo would now succeed.
    # --------------------------------------------------------

    from world import can_redo

    would_be_valid = can_redo(
        history
    )

    return (
        ops,
        state,
        history,
        containers,
        {
            "would_be_valid": would_be_valid,
        },
    )
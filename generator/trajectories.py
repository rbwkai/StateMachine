from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

from world import (
    History,
    Move,
    Operation,
    Put,
    WorldState,
    apply_op,
    replay_trace,
)
from generator.trajectory_specs import TrajectorySpec
from .trajectory_validation import validate_trajectory


# ============================================================
# Constructed trajectory
# ============================================================

@dataclass
class ConstructedTrajectory:
    """
    Symbolic trajectory produced by a trajectory constructor.

    ops:
        Canonical sequence of world operations.

    containers:
        Valid containers in the simulated world.

    final_state:
        State produced while constructing the trajectory.

    history:
        Undo/redo history associated with construction.

    target_obj:
        Object whose state is queried by the benchmark.

    spec:
        Structural specification used to construct the trajectory.
    """

    ops: List[Operation]
    containers: Set[str]
    final_state: WorldState
    history: History
    target_obj: str
    spec: TrajectorySpec


# ============================================================
# Helpers
# ============================================================

def _initial_world(
    num_containers: int,
) -> Tuple[WorldState, History, Set[str]]:
    """
    Create an empty canonical world.
    """

    if num_containers < 2:
        raise ValueError(
            "trajectory requires at least 2 containers"
        )

    containers = {
        f"c{i}"
        for i in range(num_containers)
    }

    state = WorldState(
        object_type={},
        location={},
        containers=containers,
        step_index=0,
    )

    history = History()

    return state, history, containers


def _apply(
    state: WorldState,
    history: History,
    ops: List[Operation],
    op: Operation,
) -> WorldState:
    """
    Apply an operation through the canonical world simulator.

    Every operation emitted by a trajectory constructor goes
    through apply_op().
    """

    state = apply_op(
        op,
        state,
        history,
    )

    ops.append(op)

    return state


def _put(
    state: WorldState,
    history: History,
    ops: List[Operation],
    obj_id: str,
    obj_type: str,
    container: str,
) -> WorldState:
    """
    Create one entity during setup.
    """

    return _apply(
        state,
        history,
        ops,
        Put(
            obj_id=obj_id,
            obj_type=obj_type,
            container=container,
        ),
    )


def _move(
    state: WorldState,
    history: History,
    ops: List[Operation],
    obj_id: str,
    destination: str,
) -> WorldState:
    """
    Move one existing entity.
    """

    return _apply(
        state,
        history,
        ops,
        Move(
            obj_id=obj_id,
            dst=destination,
        ),
    )


# ============================================================
# Basic chain
# ============================================================

def build_basic_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a pure temporal-maintenance trajectory.

    Structure:

        Put target -> c0       setup

        Move target -> c1      target update
        Move target -> c2      target update
        Move target -> c0      target update
        ...

    Properties:

        entity_count      = 1
        distractors       = 0
        all updates       = target updates
    """

    if spec.entity_count != 1:
        raise ValueError(
            "basic_chain requires entity_count=1"
        )

    if spec.target_updates < 1:
        raise ValueError(
            "basic_chain requires at least "
            "1 target update"
        )

    if spec.distractor_updates != 0:
        raise ValueError(
            "basic_chain does not support "
            "distractor updates"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    target = spec.target_obj or "o0"

    container_list = sorted(containers)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        "key",
        container_list[0],
    )

    # --------------------------------------------------------
    # Target updates
    # --------------------------------------------------------

    current = container_list[0]

    for i in range(spec.target_updates):

        candidates = [
            c
            for c in container_list
            if c != current
        ]

        destination = candidates[
            i % len(candidates)
        ]

        state = _move(
            state,
            history,
            ops,
            target,
            destination,
        )

        current = destination

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )


# ============================================================
# Interleaved chain
# ============================================================

def build_interleaved_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a target chain with explicitly controlled
    distractor operations.

    Example:

        setup:
            Put target
            Put distractor 1
            Put distractor 2

        updates:
            Move target
            Move distractor
            Move target
            Move distractor
            ...

    Properties:

        target_updates
            controls relevant target operations.

        distractor_updates
            controls irrelevant operations.

        entity_count
            controls the number of objects available as
            distractors.
    """

    if spec.entity_count < 2:
        raise ValueError(
            "interleaved_chain requires at least "
            "2 entities"
        )

    if spec.target_updates < 1:
        raise ValueError(
            "interleaved_chain requires at least "
            "1 target update"
        )

    if spec.distractor_updates < 1:
        raise ValueError(
            "interleaved_chain requires at least "
            "1 distractor update"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = spec.target_obj or "o0"

    state = _put(
        state,
        history,
        ops,
        target,
        "key",
        container_list[0],
    )

    # --------------------------------------------------------
    # Distractors
    # --------------------------------------------------------

    distractor_types = [
        "apple",
        "phone",
        "map",
        "coin",
        "book",
        "pen",
    ]

    distractor_ids: List[str] = []

    for i in range(spec.entity_count - 1):

        obj_id = f"o{i + 1}"

        obj_type = distractor_types[
            i % len(distractor_types)
        ]

        container = container_list[
            (i + 1) % len(container_list)
        ]

        state = _put(
            state,
            history,
            ops,
            obj_id,
            obj_type,
            container,
        )

        distractor_ids.append(obj_id)

    # --------------------------------------------------------
    # Controlled interleaving
    # --------------------------------------------------------

    target_current = container_list[0]

    target_done = 0
    distractor_done = 0

    while (
        target_done < spec.target_updates
        or distractor_done < spec.distractor_updates
    ):

        # Target operation.
        if target_done < spec.target_updates:

            candidates = [
                c
                for c in container_list
                if c != target_current
            ]

            destination = candidates[
                target_done % len(candidates)
            ]

            state = _move(
                state,
                history,
                ops,
                target,
                destination,
            )

            target_current = destination
            target_done += 1

        # Distractor operation.
        if distractor_done < spec.distractor_updates:

            distractor_id = distractor_ids[
                distractor_done
                % len(distractor_ids)
            ]

            current = state.location[
                distractor_id
            ]

            candidates = [
                c
                for c in container_list
                if c != current
            ]

            destination = candidates[
                distractor_done
                % len(candidates)
            ]

            state = _move(
                state,
                history,
                ops,
                distractor_id,
                destination,
            )

            distractor_done += 1

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )


# ============================================================
# Revision
# ============================================================

def build_revision(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory where the target repeatedly changes
    location and later revisits previously occupied locations.

    Example:

        c0 -> c1 -> c2 -> c1 -> c0 -> c2 -> c0

    The repeated locations create explicit state revision.
    """

    if spec.entity_count != 1:
        raise ValueError(
            "revision requires entity_count=1"
        )

    if spec.target_updates < 3:
        raise ValueError(
            "revision requires at least "
            "3 target updates"
        )

    if spec.distractor_updates != 0:
        raise ValueError(
            "revision does not currently support "
            "distractor updates"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    target = spec.target_obj or "o0"

    container_list = sorted(containers)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        "key",
        container_list[0],
    )

    # --------------------------------------------------------
    # Explicit revision pattern
    # --------------------------------------------------------

    revision_pattern = [
        container_list[1],
        container_list[2 % len(container_list)],
        container_list[1],
        container_list[0],
        container_list[2 % len(container_list)],
        container_list[0],
    ]

    current = container_list[0]

    for i in range(spec.target_updates):

        destination = revision_pattern[
            i % len(revision_pattern)
        ]

        # Defensive no-op prevention.
        if destination == current:

            candidates = [
                c
                for c in container_list
                if c != current
            ]

            destination = candidates[
                i % len(candidates)
            ]

        state = _move(
            state,
            history,
            ops,
            target,
            destination,
        )

        current = destination

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )


# ============================================================
# Constructor registry
# ============================================================

_CONSTRUCTORS: Dict[
    str,
    Callable[
        [random.Random, TrajectorySpec],
        ConstructedTrajectory,
    ],
] = {
    "basic_chain": build_basic_chain,
    "interleaved_chain": build_interleaved_chain,
    "revision": build_revision,
}


# ============================================================
# Public builder
# ============================================================

def build_trajectory(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Build and validate a trajectory.

    Validation happens centrally here so every trajectory family
    passes through exactly the same validation gate.

    Validation stages:

        1. Constructor
        2. Structural validation
        3. Canonical replay
        4. Final-state consistency
    """

    try:
        constructor = _CONSTRUCTORS[
            spec.family
        ]
    except KeyError as exc:
        raise ValueError(
            "unknown trajectory family: "
            f"{spec.family!r}; "
            f"available={sorted(_CONSTRUCTORS)}"
        ) from exc

    # --------------------------------------------------------
    # Construct
    # --------------------------------------------------------

    result = constructor(
        rng,
        spec,
    )

    # --------------------------------------------------------
    # Canonical replay validation
    # --------------------------------------------------------

    _, replay_final, _ = replay_trace(
        result.ops,
        result.containers,
    )

    # --------------------------------------------------------
    # Location consistency
    # --------------------------------------------------------

    if (
        replay_final.location
        != result.final_state.location
    ):
        raise AssertionError(
            "constructed trajectory disagrees "
            "with canonical replay locations"
        )

    # --------------------------------------------------------
    # Object-type consistency
    # --------------------------------------------------------

    if (
        replay_final.object_type
        != result.final_state.object_type
    ):
        raise AssertionError(
            "constructed trajectory disagrees "
            "with canonical replay object types"
        )

    # --------------------------------------------------------
    # Container consistency
    # --------------------------------------------------------

    if (
        replay_final.containers
        != result.final_state.containers
    ):
        raise AssertionError(
            "constructed trajectory disagrees "
            "with canonical replay containers"
        )

    return result


# ============================================================
# Family discovery
# ============================================================

def available_families() -> List[str]:
    """
    Return all currently implemented trajectory families.
    """

    return sorted(_CONSTRUCTORS)
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from world import (
    History,
    Merge,
    Move,
    Operation,
    Put,
    Redo,
    Split,
    Swap,
    Undo,
    WorldState,
    apply_op,
    replay_trace,
)
from generator.trajectory_specs import TrajectorySpec
from .trajectory_validation import validate_trajectory
from .metadata import MeasuredFactors, measure_factors


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

    measured_factors:
        Independently measured (E, T, D, V, L) values computed from
        the canonical replay.  Populated by build_trajectory(); None
        if the trajectory was built directly via a constructor function
        (bypassing build_trajectory).
    """

    ops: List[Operation]
    containers: Set[str]
    final_state: WorldState
    history: History
    target_obj: str
    spec: TrajectorySpec
    measured_factors: Optional[MeasuredFactors] = field(default=None, compare=False)


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


def _split(
    state: WorldState,
    history: History,
    ops: List[Operation],
    source_obj_id: str,
    new_obj_id: str,
) -> WorldState:
    """
    Split one existing entity into two.

    The new entity starts at the same container as the source.
    Both entities continue to exist independently afterwards.
    """

    return _apply(
        state,
        history,
        ops,
        Split(
            source_obj_id=source_obj_id,
            new_obj_id=new_obj_id,
        ),
    )


def _merge(
    state: WorldState,
    history: History,
    ops: List[Operation],
    src_container: str,
    dst_container: str,
) -> WorldState:
    """
    Move all objects from src_container into dst_container.

    Objects continue to exist; only their location changes.
    The over-persistence failure mode: a model that remembers
    the old src location after the merge has happened.
    """

    return _apply(
        state,
        history,
        ops,
        Merge(
            src_container=src_container,
            dst_container=dst_container,
        ),
    )


def _swap(
    state: WorldState,
    history: History,
    ops: List[Operation],
    container_a: str,
    container_b: str,
) -> WorldState:
    """
    Atomically swap all objects between two containers.

    The no-temp-variable failure mode: a model that puts both
    entities in the same container (applies only one direction
    of the swap).
    """

    return _apply(
        state,
        history,
        ops,
        Swap(
            container_a=container_a,
            container_b=container_b,
        ),
    )


def _undo(
    state: WorldState,
    history: History,
    ops: List[Operation],
) -> WorldState:
    """
    Roll back the most recent undoable operation.
    """

    return _apply(state, history, ops, Undo())


def _redo(
    state: WorldState,
    history: History,
    ops: List[Operation],
) -> WorldState:
    """
    Re-apply the most recently undone operation.
    """

    return _apply(state, history, ops, Redo())


OBJECT_TYPES: List[str] = [
    "key", "apple", "phone", "map", "coin", "book", "pen",
    "card", "ring", "token", "gem", "watch", "letter",
]


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

    target_type = rng.choice(OBJECT_TYPES)
    start_container = rng.choice(container_list)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        start_container,
    )

    # --------------------------------------------------------
    # Target updates
    # --------------------------------------------------------

    current = start_container

    for _ in range(spec.target_updates):

        candidates = [
            c
            for c in container_list
            if c != current
        ]

        destination = rng.choice(candidates)

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
    target = spec.target_obj or "o0"

    types_pool = rng.sample(
        OBJECT_TYPES, k=min(len(OBJECT_TYPES), spec.entity_count)
    )
    target_type = types_pool[0]
    target_start = rng.choice(container_list)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        target_start,
    )

    # --------------------------------------------------------
    # Distractors
    # --------------------------------------------------------

    distractor_ids: List[str] = []

    for i in range(spec.entity_count - 1):

        obj_id = f"o{i + 1}"

        obj_type = types_pool[(i + 1) % len(types_pool)]
        container = rng.choice(container_list)

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

    # --------------------------------------------------------
    # Controlled interleaving - generate target trajectory FIRST
    # --------------------------------------------------------

    # Derive a deterministic target seed from FIXED parameters only
    # (family, target_updates, entity_count, num_containers) so that
    # the target trajectory is identical across matched conditions
    # with different distractor_updates or textual_distractor_count.
    target_seed = int(
        hashlib.sha256(
            f"{spec.family}|T{spec.target_updates}|E{spec.entity_count}|C{spec.num_containers}|target"
            .encode()
        ).hexdigest()[:8],
        16,
    )
    target_rng = random.Random(target_seed)

    # Generate target trajectory independently using target_rng
    target_moves = []  # list of (destination_container)
    target_current = target_start
    for _ in range(spec.target_updates):
        candidates = [c for c in container_list if c != target_current]
        destination = target_rng.choice(candidates)
        target_current = destination
        target_moves.append(destination)

    # Now interleave: replay target moves and interleave distractors
    target_current = target_start
    target_done = 0
    distractor_done = 0

    # Re-initialize state and history for full replay
    state, history, containers = _initial_world(spec.num_containers)
    ops = []

    # Setup: Put target
    state = _put(state, history, ops, target, target_type, target_start)

    # Setup: Put distractors
    distractor_ids: List[str] = []

    for i in range(spec.entity_count - 1):
        obj_id = f"o{i + 1}"
        obj_type = types_pool[(i + 1) % len(types_pool)]
        container = rng.choice(container_list)
        state = _put(state, history, ops, obj_id, obj_type, container)
        distractor_ids.append(obj_id)

    target_current = target_start
    target_done = 0
    distractor_done = 0
    target_move_idx = 0

    while (
        target_done < spec.target_updates
        or distractor_done < spec.distractor_updates
    ):

        # Target operation.
        if target_done < spec.target_updates:
            # Replay the pre-determined target move
            destination = target_moves[target_move_idx]
            state = _move(
                state,
                history,
                ops,
                target,
                destination,
            )
            target_current = destination
            target_done += 1
            target_move_idx += 1

        # Distractor operation.
        if distractor_done < spec.distractor_updates:
            distractor_id = rng.choice(distractor_ids)
            current = state.location[distractor_id]
            candidates = [c for c in container_list if c != current]
            destination = rng.choice(candidates)
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

    perm = rng.sample(container_list, len(container_list))
    start_container = perm[0]
    target_type = rng.choice(OBJECT_TYPES)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        start_container,
    )

    # --------------------------------------------------------
    # Explicit revision pattern over permuted containers
    # --------------------------------------------------------

    revision_pattern = [
        perm[1],
        perm[2 % len(perm)],
        perm[1],
        perm[0],
        perm[2 % len(perm)],
        perm[0],
    ]

    current = start_container

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

            destination = rng.choice(candidates)

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
# split_chain
# ============================================================

def build_split_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that introduces identity multiplication
    via a Split operation, with a count query that makes Split
    causally necessary.

    Pattern:
    Put target -> c0
    Move target -> c1 (pre-split)
    Split target -> child (at c1)
    Merge c1 -> c2 (moves both target and child to c2)
    Count query: How many [type] in c2? Answer: 2 (with Split) vs 1 (without)

    This ensures Split is causally necessary:
    - With Split: both target and child exist, Merge moves both, count = 2
    - Without Split: only target exists, Merge moves only target, count = 1
    Trace remains playable in both cases.
    """

    if spec.entity_count != 2:
        raise ValueError(
            "split_chain requires entity_count=2 "
            "(target + one child from the split)"
        )

    if spec.target_updates < 3:
        raise ValueError(
            "split_chain requires at least "
            "3 target_updates (pre-split move + split + merge)"
        )

    if spec.query_type != "count":
        raise ValueError(
            "split_chain requires query_type='count'"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"
    child = "o1"

    start_c = rng.choice(container_list)
    target_type = rng.choice(OBJECT_TYPES)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        start_c,
    )

    # --------------------------------------------------------
    # Pre-split: move target to c1
    # --------------------------------------------------------

    pre_split_candidates = [
        c for c in container_list if c != start_c
    ]
    c1 = rng.choice(pre_split_candidates)

    state = _move(
        state,
        history,
        ops,
        target,
        c1,
    )

    target_current = c1
    target_updates_done = 1

    # --------------------------------------------------------
    # Split: child spawns at the same container as target (c1)
    # --------------------------------------------------------

    state = _split(
        state,
        history,
        ops,
        target,
        child,
    )

    child_current = target_current
    target_updates_done += 1

    # --------------------------------------------------------
    # Merge: move both target and child from c1 to c2
    # --------------------------------------------------------

    merge_candidates = [c for c in container_list if c != target_current]
    c2 = rng.choice(merge_candidates)

    state = _merge(
        state,
        history,
        ops,
        target_current,
        c2,
    )

    target_current = c2
    child_current = c2
    target_updates_done += 1

    # --------------------------------------------------------
    # Additional target moves if needed (after merge)
    # --------------------------------------------------------

    while target_updates_done < spec.target_updates:
        candidates = [
            c for c in container_list if c != target_current
        ]

        dst = rng.choice(candidates)

        state = _move(
            state,
            history,
            ops,
            target,
            dst,
        )

        target_current = dst
        target_updates_done += 1

    # --------------------------------------------------------
    # Distractor: move child
    # --------------------------------------------------------

    for _ in range(spec.distractor_updates):

        candidates = [
            c for c in container_list if c != child_current
        ]

        dst = rng.choice(candidates)

        state = _move(
            state,
            history,
            ops,
            child,
            dst,
        )

        child_current = dst

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )


# ============================================================
# merge_chain
# ============================================================

# ============================================================
# merge_chain
# ============================================================

def build_merge_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that introduces identity consolidation
    via a container-level Merge operation, with a count query that makes Merge
    causally necessary.

    Pattern:
    Put target + companions -> c0
    Merge c0 -> c1 (moves all entities to c1)
    Count query: How many [type] in c1? Answer: E (with Merge) vs 0 (without)

    This ensures Merge is causally necessary:
    - With Merge: all entities moved to c1, count = E
    - Without Merge: entities stay in c0, count in c1 = 0
    Trace remains playable in both cases.
    """

    if spec.entity_count < 2:
        raise ValueError(
            "merge_chain requires entity_count >= 2 "
            "(target + at least one companion)"
        )

    if spec.target_updates < 1:
        raise ValueError(
            "merge_chain requires at least "
            "1 target_updates (the merge)"
        )

    if spec.query_type != "count":
        raise ValueError(
            "merge_chain requires query_type='count'"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"

    types_pool = rng.sample(
        OBJECT_TYPES, k=min(len(OBJECT_TYPES), spec.entity_count)
    )
    start_c = rng.choice(container_list)

    # --------------------------------------------------------
    # Setup: place target and companions in initial container
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        types_pool[0],
        start_c,
    )

    companion_ids: List[str] = []

    for i in range(spec.entity_count - 1):
        companion_id = f"o{i + 1}"
        obj_type = types_pool[(i + 1) % len(types_pool)]

        state = _put(
            state,
            history,
            ops,
            companion_id,
            obj_type,
            start_c,
        )

        companion_ids.append(companion_id)

    # --------------------------------------------------------
    # Merge: relocate everything from start_c to merge_dst
    # --------------------------------------------------------

    merge_candidates = [c for c in container_list if c != start_c]
    merge_dst = rng.choice(merge_candidates)

    state = _merge(
        state,
        history,
        ops,
        start_c,
        merge_dst,
    )

    target_current = merge_dst
    target_updates_done = 1

    # --------------------------------------------------------
    # No trailing Moves after Merge - Merge is the FINAL target-affecting operation
    # If target_updates > 1, the extra target_updates are absorbed by the Merge
    # since Merge moves all entities at once (counts as 1 target update).
    # This ensures the count query depends on the Merge.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Distractor: move companions
    # --------------------------------------------------------

    for _ in range(spec.distractor_updates):
        companion_id = rng.choice(companion_ids)

        current = state.location[companion_id]

        candidates = [
            c for c in container_list if c != current
        ]

        dst = rng.choice(candidates)

        state = _move(
            state,
            history,
            ops,
            companion_id,
            dst,
        )

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )





# ============================================================
# swap_chain
# ============================================================

def build_swap_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that introduces simultaneous bilateral
    updates via Swap operations, with a count query that makes Swap
    causally necessary.

    Pattern:
    Put target (type A) -> c0
    Put other (type B) -> c1
    [Optional: Move target to c2, Move other to c3]
    Swap cX <-> cY (FINAL target-affecting operation, containers chosen randomly)
    Count query: How many A in cY?
      - With Swap: target (type A) moved to cY, count = 1
      - Without Swap: target stayed in cX, count in cY = 0

    This ensures Swap is causally necessary and is the final target-affecting operation.
    Uses 3+ containers to break parity.
    """

    if spec.entity_count < 2:
        raise ValueError(
            "swap_chain requires entity_count >= 2"
        )

    if spec.num_containers < 3:
        raise ValueError(
            "swap_chain requires at least 3 containers"
        )

    if spec.target_updates < 1:
        raise ValueError(
            "swap_chain requires at least 1 target_update"
        )

    if spec.query_type != "count":
        raise ValueError(
            "swap_chain requires query_type='count'"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"
    other = "o1"

    types_pool = rng.sample(
        OBJECT_TYPES, k=min(len(OBJECT_TYPES), spec.entity_count)
    )
    target_type = types_pool[0]
    other_type = types_pool[1 % len(types_pool)]

    # --------------------------------------------------------
    # Setup: target at c0, other at c1
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        "c0",
    )

    state = _put(
        state,
        history,
        ops,
        other,
        other_type,
        "c1",
    )

    companion_ids = [other]

    for i in range(2, spec.entity_count):
        companion_id = f"o{i}"
        c = rng.choice(container_list)
        state = _put(
            state,
            history,
            ops,
            companion_id,
            types_pool[i % len(types_pool)],
            c,
        )
        companion_ids.append(companion_id)

    target_current = "c0"
    other_current = "c1"

    updates_done = 0

    # --------------------------------------------------------
    # Optional pre-Swap Moves to break parity
    # --------------------------------------------------------

    # The final Swap counts as 1 target update, so we need (target_updates - 1) pre-swap target Moves
    # Pre-swap Moves on "other" are distractors (D), not target updates (T)
    max_pre_swap_target_moves = spec.target_updates - 1
    pre_swap_target_moves = max_pre_swap_target_moves
    
    for _ in range(pre_swap_target_moves):
        # Move target to a different container (always affects target = T update)
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        updates_done += 1

        # Optional: Move other as distractor (D update, not counted toward T)
        # This adds noise without consuming target update budget
        if rng.random() < 0.5:
            candidates = [c for c in container_list if c != other_current]
            if candidates:
                dst = rng.choice(candidates)
                state = _move(state, history, ops, other, dst)
                other_current = dst
                # Note: this is a distractor update, NOT counted in updates_done

    # --------------------------------------------------------
    # Final Swap between two randomly chosen containers (always executed, affects target = T update)
    # --------------------------------------------------------

    # Choose two distinct containers for the final swap, ensuring target's container is one of them
    swap_candidates = list(container_list)
    if len(swap_candidates) >= 2:
        # Ensure target's current container is one of the swapped containers
        if target_current in swap_candidates:
            other_containers = [c for c in swap_candidates if c != target_current]
            if other_containers:
                c_a = target_current
                c_b = rng.choice(other_containers)
            else:
                c_a, c_b = rng.sample(swap_candidates, 2)
        else:
            c_a, c_b = rng.sample(swap_candidates, 2)
        state = _swap(state, history, ops, c_a, c_b)
        # Update positions if target/other involved
        if target_current == c_a:
            target_current = c_b
        elif target_current == c_b:
            target_current = c_a
        if other_current == c_a:
            other_current = c_b
        elif other_current == c_b:
            other_current = c_a
        updates_done += 1  # Final Swap counts as 1 target update

    # --------------------------------------------------------
    # Distractor: move companions
    # --------------------------------------------------------

    for _ in range(spec.distractor_updates):

        companion_id = rng.choice(companion_ids)

        current = state.location[companion_id]

        candidates = [
            c for c in container_list if c != current
        ]

        dst = rng.choice(candidates)

        state = _move(
            state,
            history,
            ops,
            companion_id,
            dst,
        )

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )


# ============================================================
# undo_chain
# ============================================================

def build_undo_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that tests rollback / contradiction
    handling via Undo operations, with a count query that makes
    Undo causally necessary.

    Pattern:
    [Pre-moves] -> Move to c1 -> [More moves] -> Undo -> [Post-Undo moves]
    Count query: How many [type] in c0 (original container)?
      - With Undo: target back in c0, count = 1
      - Without Undo: target in c1, count in c0 = 0

    This ensures Undo is causally necessary.
    Undo position varies randomly to break the "penultimate Move" shortcut.
    """

    if spec.entity_count != 1:
        raise ValueError(
            "undo_chain requires entity_count=1"
        )

    if spec.target_updates < 2:
        raise ValueError(
            "undo_chain requires at least "
            "2 target_updates (move + undo)"
        )

    if spec.query_type != "count":
        raise ValueError(
            "undo_chain requires query_type='count'"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"
    target_start = rng.choice(container_list)
    target_type = rng.choice(OBJECT_TYPES)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        target_start,
    )

    target_current = target_start
    updates_done = 0

    # Pre-Undo moves (at least 1, at most target_updates - 1)
    pre_undo_moves = rng.randint(1, spec.target_updates - 1)
    for _ in range(pre_undo_moves):
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        updates_done += 1

    # Undo operation
    state = _undo(state, history, ops)
    # Read actual target location from state after Undo (not assumed target_start)
    target_current = state.location.get(target, target_start)
    updates_done += 1

    # Post-Undo moves (if any target_updates remain)
    while updates_done < spec.target_updates:
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        updates_done += 1

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )


# ============================================================
# undo_redo_chain
# ============================================================

def build_undo_redo_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that tests 3-way edit-history
    awareness via Undo/Redo.

    Pattern (for T >= 6):
    [Extra moves] -> MoveA -> MoveB -> MoveC -> Undo -> Undo -> Redo (FINAL)
    
    Where MoveA/MoveB/MoveC are the last 3 moves before the Undo/Redo sequence.
    Extra moves (if any) come before MoveA and don't affect the causal structure.
    
    This ensures the Undo/Redo pair has a causal effect on the final answer:
    - MoveA: ... -> cA
    - MoveB: cA -> cB
    - MoveC: cB -> cC
    - Undo1: undoes MoveC, target back to cB
    - Undo2: undoes MoveB, target back to cA
    - Redo: redoes MoveB (the last undone), target to cB (FINAL)
    
    Final answer = cB (destination of MoveB).
    Without the last Undo/Redo: final = cC (destination of MoveC).
    Removing last Undo changes answer (cB vs cC).
    Removing Redo changes answer (cA vs cB).
    """

    if spec.entity_count != 1:
        raise ValueError(
            "undo_redo_chain requires entity_count=1"
        )

    if spec.target_updates < 6:
        raise ValueError(
            "undo_redo_chain requires at least "
            "6 target_updates (move1 + move2 + move3 + undo + undo + redo)"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"
    target_start = rng.choice(container_list)
    target_type = rng.choice(OBJECT_TYPES)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        target_type,
        target_start,
    )

    target_current = target_start
    updates_done = 0

    # Extra moves before the causal 3-move sequence
    extra_moves = spec.target_updates - 6
    for _ in range(extra_moves):
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        updates_done += 1

    # MoveA
    candidates = [c for c in container_list if c != target_current]
    cA = rng.choice(candidates)
    state = _move(state, history, ops, target, cA)
    target_current = cA
    updates_done += 1

    # MoveB
    candidates = [c for c in container_list if c != target_current]
    cB = rng.choice(candidates)
    state = _move(state, history, ops, target, cB)
    target_current = cB
    updates_done += 1

    # MoveC
    candidates = [c for c in container_list if c != target_current]
    cC = rng.choice(candidates)
    state = _move(state, history, ops, target, cC)
    target_current = cC
    updates_done += 1

    # Undo1 (undoes MoveC, back to cB)
    state = _undo(state, history, ops)
    target_current = cB
    updates_done += 1

    # Undo2 (undoes MoveB, back to cA)
    state = _undo(state, history, ops)
    target_current = cA
    updates_done += 1

    # Redo (redoes MoveB, back to cB) - THIS IS THE FINAL OPERATION
    state = _redo(state, history, ops)
    target_current = cB  # Redo redoes MoveB
    updates_done += 1

    return ConstructedTrajectory(
        ops=ops,
        containers=containers,
        final_state=state,
        history=history,
        target_obj=target,
        spec=spec,
    )



_CONSTRUCTORS: Dict[
    str,
    Callable[
        [random.Random, TrajectorySpec],
        ConstructedTrajectory,
    ],
] = {
    # --------------------------------------------------------
    # Original RQ1-4 families
    # --------------------------------------------------------
    "basic_chain": build_basic_chain,
    "interleaved_chain": build_interleaved_chain,
    "revision": build_revision,
    # --------------------------------------------------------
    # RQ5 structural families  (PENDING_CALIBRATION)
    # --------------------------------------------------------
    "split_chain": build_split_chain,
    "merge_chain": build_merge_chain,
    "swap_chain": build_swap_chain,
    "undo_chain": build_undo_chain,
    "undo_redo_chain": build_undo_redo_chain,
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

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    validate_trajectory(
        result.ops,
        result.target_obj,
        result.spec,
    )

    # --------------------------------------------------------
    # Measured factor computation
    #
    # Independently measure (E, T, D, V) from the canonical
    # replay — do not trust the requested spec values.
    # --------------------------------------------------------

    measured = measure_factors(
        result.ops,
        result.containers,
        result.target_obj,
    )

    result.measured_factors = measured

    return result


# ============================================================
# Family discovery
# ============================================================

def available_families() -> List[str]:
    """
    Return all currently implemented trajectory families.
    """

    return sorted(_CONSTRUCTORS)
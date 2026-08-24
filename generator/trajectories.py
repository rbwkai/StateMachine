from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

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
# split_chain
# ============================================================

def build_split_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that introduces identity multiplication
    via a Split operation.

    Structure::

        Setup:
            Put target -> c0

        Pre-split:
            Move target -> c1        (establishes independent identity)

        Split:
            Split target -> child    (child appears at c1 alongside target)

        Post-split (target_updates - 2 remaining target moves):
            Move target -> c2
            Move target -> c0
            ...

        Distractor moves (distractor_updates child moves):
            Move child -> c2
            Move child -> c1
            ...

    target_obj
        The original entity (before the split).  After the split
        both target and child are trackable, but queries probe
        the original.

    Distinct failure mode
        The model merges child back into target (reports target's
        location for child or vice versa), or loses track of one
        of the two post-split entities entirely.

    Constraints
        entity_count = 2 (target + child)
        target_updates >= 2 (at least pre-split move + post-split move)
        structural_ops must contain "split"
    """

    if spec.entity_count != 2:
        raise ValueError(
            "split_chain requires entity_count=2 "
            "(target + one child from the split)"
        )

    if spec.target_updates < 2:
        raise ValueError(
            "split_chain requires at least "
            "2 target_updates (pre-split move + "
            "at least one post-split move)"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"
    child = "o1"

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
    # Pre-split: move target to establish independent identity
    # --------------------------------------------------------

    pre_split_candidates = [
        c for c in container_list if c != container_list[0]
    ]
    pre_split_dst = pre_split_candidates[0]

    state = _move(
        state,
        history,
        ops,
        target,
        pre_split_dst,
    )

    target_current = pre_split_dst
    target_updates_done = 1

    # --------------------------------------------------------
    # Split: child spawns at the same container as target
    # The Split operation counts as 1 target update on target.
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
    # Post-split: move target independently
    # --------------------------------------------------------

    while target_updates_done < spec.target_updates:

        candidates = [
            c for c in container_list if c != target_current
        ]

        dst = candidates[
            target_updates_done % len(candidates)
        ]

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

    for i in range(spec.distractor_updates):

        candidates = [
            c for c in container_list if c != child_current
        ]

        dst = candidates[i % len(candidates)]

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

def build_merge_chain(
    rng: random.Random,
    spec: TrajectorySpec,
) -> ConstructedTrajectory:
    """
    Construct a trajectory that introduces identity consolidation
    via a container-level Merge operation.

    The Merge op moves all objects from a source container into a
    destination container.  Objects continue to exist; only their
    location changes.

    Structure::

        Setup:
            Put target -> c0
            Put companion -> c1

        Pre-merge moves (optional, controlled by target_updates):
            Move target -> c0
            Move companion -> c0   (bring companion into same container)

        Merge:
            Merge c0 -> c2         (target and companion both relocate)

        Post-merge moves:
            Move target -> c3
            ...

    target_obj
        The primary entity whose location is queried.

    Distinct failure mode
        Over-persistence: the model reports target at c0 (its
        pre-merge container) instead of c2 (the merge destination),
        because it failed to apply the container-level relocation.

    Constraints
        entity_count >= 2 (target + at least one companion to merge with)
        target_updates >= 2 (at least one move to set up the merge src +
                             the merge op itself counts as one target update)
        structural_ops must contain "merge"
    """

    if spec.entity_count < 2:
        raise ValueError(
            "merge_chain requires entity_count >= 2 "
            "(target + at least one companion)"
        )

    if spec.target_updates < 2:
        raise ValueError(
            "merge_chain requires at least "
            "2 target_updates (setup move + merge)"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"

    distractor_types = [
        "apple", "phone", "map", "coin", "book", "pen",
    ]

    # --------------------------------------------------------
    # Setup: place target and companions in initial container
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        "key",
        container_list[0],
    )

    companion_ids: List[str] = []

    for i in range(spec.entity_count - 1):
        companion_id = f"o{i + 1}"
        obj_type = distractor_types[i % len(distractor_types)]

        state = _put(
            state,
            history,
            ops,
            companion_id,
            obj_type,
            container_list[0],
        )

        companion_ids.append(companion_id)

    target_current = container_list[0]

    # --------------------------------------------------------
    # Merge: relocate everything from target_current to c1
    # This causally moves target (and all companions) to merge_dst,
    # counting as exactly 1 target update.
    # --------------------------------------------------------

    merge_dst = container_list[
        (container_list.index(target_current) + 1)
        % len(container_list)
    ]

    state = _merge(
        state,
        history,
        ops,
        target_current,
        merge_dst,
    )

    target_current = merge_dst
    target_updates_done = 1

    # --------------------------------------------------------
    # Post-merge: additional target moves
    # --------------------------------------------------------

    while target_updates_done < spec.target_updates:

        candidates = [
            c for c in container_list if c != target_current
        ]

        dst = candidates[
            target_updates_done % len(candidates)
        ]

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
    # Distractor: move companions (strictly distractor_updates)
    # --------------------------------------------------------

    for i in range(spec.distractor_updates):

        companion_id = companion_ids[
            i % len(companion_ids)
        ]

        current = state.location[companion_id]

        candidates = [
            c for c in container_list if c != current
        ]

        dst = candidates[i % len(candidates)]

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
    updates via Swap operations.

    Structure::

        Setup:
            Put target -> c0
            Put other   -> c1

        Swap c0 <-> c1              (target now at c1, other at c0)

        Move target -> c2           (target moves independently)

        Swap c0 <-> c2              (target now at c0, other at c2)
        ...

    target_obj
        The entity initially at c0.

    Distinct failure mode
        The no-temp-variable bug: the model applies the swap
        as two sequential moves, placing both entities in the
        same container (whichever was written second).

    Constraints
        entity_count >= 2 (need two entities to observe a swap)
        num_containers >= 2 (need two containers to swap)
        target_updates >= 1 (at least one swap counts as a target update)
        structural_ops must contain "swap"
    """

    if spec.entity_count < 2:
        raise ValueError(
            "swap_chain requires entity_count >= 2"
        )

    if spec.num_containers < 2:
        raise ValueError(
            "swap_chain requires at least 2 containers"
        )

    if spec.target_updates < 1:
        raise ValueError(
            "swap_chain requires at least 1 target_update"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"
    other = "o1"

    distractor_types = [
        "apple", "phone", "map", "coin", "book", "pen",
    ]

    # --------------------------------------------------------
    # Setup: target at c0, other at c1
    # --------------------------------------------------------

    state = _put(
        state,
        history,
        ops,
        target,
        "key",
        container_list[0],
    )

    state = _put(
        state,
        history,
        ops,
        other,
        distractor_types[0],
        container_list[1],
    )

    # Extra companions if entity_count > 2
    companion_ids = [other]

    for i in range(2, spec.entity_count):
        companion_id = f"o{i}"
        state = _put(
            state,
            history,
            ops,
            companion_id,
            distractor_types[i % len(distractor_types)],
            container_list[i % len(container_list)],
        )
        companion_ids.append(companion_id)

    target_current = container_list[0]
    other_current = container_list[1]

    updates_done = 0

    # --------------------------------------------------------
    # Alternating: Swap then optional Move
    # --------------------------------------------------------

    while updates_done < spec.target_updates:

        # Swap target's container with other's container.
        state = _swap(
            state,
            history,
            ops,
            target_current,
            other_current,
        )

        # After swap: target is now where other was.
        target_current, other_current = (
            other_current,
            target_current,
        )

        updates_done += 1

        if updates_done >= spec.target_updates:
            break

        # Move target to a third container (if available).
        candidates = [
            c
            for c in container_list
            if c != target_current and c != other_current
        ]

        if candidates:
            dst = candidates[
                updates_done % len(candidates)
            ]

            state = _move(
                state,
                history,
                ops,
                target,
                dst,
            )

            target_current = dst
            updates_done += 1

    # --------------------------------------------------------
    # Distractor: move companions
    # --------------------------------------------------------

    for i in range(spec.distractor_updates):

        companion_id = companion_ids[
            i % len(companion_ids)
        ]

        current = state.location[companion_id]

        candidates = [
            c for c in container_list if c != current
        ]

        dst = candidates[i % len(candidates)]

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
    handling via Undo operations.

    target_updates counts ALL non-Put ops that affect the target's
    effective state trajectory: moves AND undos.

    Structure (target_updates=4 example)::

        Setup:
            Put target -> c0

        Move target -> c1            (+1)
        Undo                         (+1, target back at c0)
        Move target -> c2            (+1)
        Move target -> c3            (+1)

    target_obj
        The single tracked entity.

    Distinct failure mode
        The model treats the undone action as if it happened:
        it reports c1 (the undone destination) instead of c0
        (the correctly rolled-back location).

    Constraints
        entity_count = 1
        target_updates >= 2 (at least one move + one undo)
        structural_ops must contain "undo"
    """

    if spec.entity_count != 1:
        raise ValueError(
            "undo_chain requires entity_count=1"
        )

    if spec.target_updates < 2:
        raise ValueError(
            "undo_chain requires at least "
            "2 target_updates (one move + one undo)"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"

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

    target_current = container_list[0]
    updates_done = 0

    # --------------------------------------------------------
    # Pattern: Move, Undo, Move, Move, ...
    #
    # One Undo is placed after the first Move to create the
    # contradiction.  Subsequent ops are plain Moves.
    # --------------------------------------------------------

    # Step 1: move to c1 (will be undone)
    candidates = [c for c in container_list if c != target_current]
    dst = candidates[0]

    state = _move(state, history, ops, target, dst)
    pre_undo_current = target_current   # where target will return to
    updates_done += 1

    if updates_done >= spec.target_updates:
        # Degenerate edge case: only 1 update requested; skip undo.
        # (Validator will catch target_updates < 2 before we get here.)
        return ConstructedTrajectory(
            ops=ops,
            containers=containers,
            final_state=state,
            history=history,
            target_obj=target,
            spec=spec,
        )

    # Step 2: Undo (target rolls back to pre_undo_current)
    state = _undo(state, history, ops)
    target_current = pre_undo_current
    updates_done += 1

    # Steps 3+: plain moves
    move_idx = 0

    while updates_done < spec.target_updates:

        candidates = [
            c for c in container_list if c != target_current
        ]

        dst = candidates[move_idx % len(candidates)]

        state = _move(state, history, ops, target, dst)

        target_current = dst
        updates_done += 1
        move_idx += 1

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
    awareness: never-happened / happened-then-undone /
    happened-undone-then-redone.

    target_updates counts ALL non-Put ops that affect the target's
    effective state trajectory: moves, undos, AND redos.

    Structure (target_updates=5 example)::

        Setup:
            Put target -> c0

        Move target -> c1            (+1)   happened
        Undo                         (+1)   undone (target at c0)
        Redo                         (+1)   redone (target at c1)
        Move target -> c2            (+1)   normal move
        Move target -> c3            (+1)   normal move

    target_obj
        The single tracked entity.

    Distinct failure mode
        Conflating "undone" (target at c0) with "redone" (target
        at c1): the model reports the correct final location but
        cannot correctly answer step-wise queries about the
        undo/redo cycle.

    Constraints
        entity_count = 1
        target_updates >= 3 (move + undo + redo minimum)
        structural_ops must contain "undo" and "redo"
    """

    if spec.entity_count != 1:
        raise ValueError(
            "undo_redo_chain requires entity_count=1"
        )

    if spec.target_updates < 3:
        raise ValueError(
            "undo_redo_chain requires at least "
            "3 target_updates (move + undo + redo)"
        )

    state, history, containers = _initial_world(
        spec.num_containers
    )

    ops: List[Operation] = []

    container_list = sorted(containers)

    target = spec.target_obj or "o0"

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

    target_current = container_list[0]
    updates_done = 0

    # --------------------------------------------------------
    # Core undo/redo cycle: Move -> Undo -> Redo
    # --------------------------------------------------------

    # Step 1: Move to c1
    candidates = [c for c in container_list if c != target_current]
    c1 = candidates[0]

    state = _move(state, history, ops, target, c1)
    updates_done += 1

    # Step 2: Undo (target back at c0)
    state = _undo(state, history, ops)
    target_current = container_list[0]
    updates_done += 1

    # Step 3: Redo (target back at c1)
    state = _redo(state, history, ops)
    target_current = c1
    updates_done += 1

    # Steps 4+: plain moves
    move_idx = 0

    while updates_done < spec.target_updates:

        candidates = [
            c for c in container_list if c != target_current
        ]

        dst = candidates[move_idx % len(candidates)]

        state = _move(state, history, ops, target, dst)

        target_current = dst
        updates_done += 1
        move_idx += 1

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

    return result


# ============================================================
# Family discovery
# ============================================================

def available_families() -> List[str]:
    """
    Return all currently implemented trajectory families.
    """

    return sorted(_CONSTRUCTORS)
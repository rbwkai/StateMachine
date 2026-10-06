from __future__ import annotations

import random
from collections import Counter
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

    # Derive a deterministic target seed mixing instance RNG so paths vary
    # across seeds but remain invariant to distractor/text changes when the
    # same RNG seed is used with different spec params (excluding those params).
    target_seed = rng.getrandbits(32)
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

def _revision_walk(rng: random.Random, containers: list, start: str, T: int, tries: int = 200) -> list:
    """Random walk with at least one revisit, final != start, final not most-mentioned.

    Falls back to revisit-only when containers==2 makes strict impossible
    (alternation forces final==start on even T).
    """
    def genuine(path: list) -> bool:
        return any(
            path[i] == path[j]
            for i in range(len(path))
            for j in range(i + 2, len(path))
        )

    fallback = None
    for _ in range(tries):
        cur, path = start, []
        for _ in range(T):
            cur = rng.choice([c for c in containers if c != cur])
            path.append(cur)
        if not genuine(path):
            continue
        if fallback is None:
            fallback = list(path)
        visited = [start] + path
        counts = Counter(visited)
        # most-mentioned constraint only when satisfiable (T>=4 with 3+
        # containers; at T=3 any genuine [a,b,a] ends on its own max).
        need_balance = not (T < 4 or len(containers) < 3)
        if path[-1] != start and (
            not need_balance or counts[path[-1]] < max(counts.values())
        ):
            return path
    if fallback is not None:
        return fallback
    raise ValueError("no valid revision walk")


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

    if spec.revision_count > spec.target_updates:
        raise ValueError(
            f"revision requires revision_count <= target_updates (revision_count={spec.revision_count} > target_updates={spec.target_updates})"
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

    # Random walk with rejection: at least one revisit, final != start,
    # final not most-mentioned. Breaks stateless/last-move shortcuts.
    path = _revision_walk(rng, container_list, start_container, spec.target_updates)

    for destination in path:
        state = _move(
            state,
            history,
            ops,
            target,
            destination,
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
    # Pre-split moves: random count so post-merge count varies 50/50.
    # T = pre + Split + Merge + post; pre uniform in [0, T-2] gives
    # co-located (post even/return) vs apart variety at fixed T.
    # --------------------------------------------------------

    pre_count = rng.randrange(0, spec.target_updates - 1)
    target_current = start_c
    target_updates_done = 0
    for _ in range(pre_count):
        candidates = [c for c in container_list if c != target_current]
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        target_updates_done += 1
    c1 = target_current

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
    # Additional target moves: 50/50 co-located vs apart at fixed T.
    # post==0 forces co-located, post==1 forces apart; otherwise sample
    # flag first then rejection-sample a walk ending as flagged.
    # --------------------------------------------------------

    post_needed = spec.target_updates - target_updates_done
    if post_needed > 0:
        if post_needed == 1:
            want_together = False
        elif target_current == child_current:
            # post==0 handled (loop skipped); here post>=2 from same spot:
            # either ending reachable; sample 50/50.
            want_together = rng.choice([True, False])
        else:
            want_together = rng.choice([True, False])
        chosen = None
        for _ in range(50):
            cur, path = target_current, []
            for _ in range(post_needed):
                cur = rng.choice([c for c in container_list if c != cur])
                path.append(cur)
            if (path[-1] == child_current) == want_together:
                chosen = path
                break
        if chosen is None:
            # Fallback: pure random walk (keeps T exact deterministically).
            cur, chosen = target_current, []
            for _ in range(post_needed):
                cur = rng.choice([c for c in container_list if c != cur])
                chosen.append(cur)
        for dst in chosen:
            state = _move(state, history, ops, target, dst)
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
    # Pre-merge target Moves
    # --------------------------------------------------------

    # The extra target updates go BEFORE the Merge, not after. A target Move
    # after the Merge would carry the target out of the Merge destination, and
    # the count question asks about that destination -- so the gold count would
    # be 0 whether or not the Merge ran, and the necessity check could not tell
    # the two traces apart. Pre-merge Moves leave the target in the destination
    # once the Merge fires, which is what makes Merge causally necessary.
    target_current = start_c
    updates_done = 0
    while updates_done < spec.target_updates - 1:
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)

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
    # Merge: relocate everything from target's container to merge_dst
    # --------------------------------------------------------

    merge_candidates = [c for c in container_list if c != target_current]
    merge_dst = rng.choice(merge_candidates)

    state = _merge(
        state,
        history,
        ops,
        target_current,
        merge_dst,
    )

    target_current = merge_dst
    updates_done += 1

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

    # Interleaved `other` Moves are D operations, so they are drawn from the
    # distractor budget rather than from a coin flip: a coin flip made D_actual a
    # function of T, so a D=0 request produced D>0 and `verify_factors` rejected
    # the instance (SPEC §6 rule 3). Spreading the budget over the pre-swap Moves
    # keeps the interleaved character while holding D_actual == requested.
    distractor_budget = spec.distractor_updates
    distractor_stride = (
        max(1, pre_swap_target_moves // distractor_budget)
        if distractor_budget
        else 0
    )
    distractors_done = 0

    for i in range(pre_swap_target_moves):
        # Move target to a different container (always affects target = T update)
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        updates_done += 1

        # Move "other" as a distractor once per stride, while budget remains.
        if (
            distractor_stride
            and distractors_done < distractor_budget
            and (i + 1) % distractor_stride == 0
        ):
            candidates = [c for c in container_list if c != other_current]
            if candidates:
                dst = rng.choice(candidates)
                state = _move(state, history, ops, other, dst)
                other_current = dst
                distractors_done += 1

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

    # Only the unspent remainder of the distractor budget lands here; the
    # interleaved pre-swap Moves above already consumed part of it, and charging
    # the full budget twice made D_actual exceed the request.
    for _ in range(distractor_budget - distractors_done):

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

    # Uniform k in {1,2,3} with k<=T-1 and k<moves (strict: k==moves
    # always returns to start and trips the final==start gate, collapsing
    # variation to k=1). T=4 admits k=1 only: penultimate ceiling documented.
    allowed = [
        x for x in (1, 2, 3)
        if x <= spec.target_updates - 1 and x < spec.target_updates - x
    ]
    if not allowed:
        allowed = [1]
    k = rng.choice(allowed)
    for _ in range(spec.target_updates - k):
        candidates = [c for c in container_list if c != target_current]
        if not candidates:
            break
        dst = rng.choice(candidates)
        state = _move(state, history, ops, target, dst)
        target_current = dst
        updates_done += 1

    for _ in range(k):
        state = _undo(state, history, ops)
        target_current = state.location.get(target, target_start)
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

    # 2U2R breaks every-required-op necessity (extra Redo goes invalid when
    # an Undo is removed), so keep 2U1R only; penultimate reported as ceiling.
    variant = "2U1R"
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
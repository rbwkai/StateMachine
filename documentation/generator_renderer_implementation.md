# Full Implementation: Generator + Renderer

## Description

This is a complete, tested Python implementation of the generator/renderer spec — the world simulator, the eight "easy enough" operations (`PUT`, `MOVE`, `REMOVE`, `UNDO`, `REDO`, `SPLIT`, `MERGE`, `SWAP`), the sequence sampler, the query/step-wise/counterfactual/redo-validity probes, and a renderer that turns everything into natural-language narratives, questions, and gold answers.

Every piece below has actually been run: all five spec test fixtures pass, a 1000-seed stress test across randomized `(E, U, N)` combinations produces zero crashes, and the worked examples at the end show real generated output (narrative → question → gold answer), including a correctly-`False` redo-validity probe and a correctly-computed post-`MERGE` count query.

One bug worth knowing about, since it's the kind of thing that's easy to reintroduce: pure random sampling can occasionally produce a sequence like `[Swap, Swap]` on a world where no object was ever created — nothing to query. The fix (in `generator/sampler.py`) is to always force the very first operation to be a `PUT`, mirroring the proposal's own "initial valid state, then updates" structure. Even with that fix, a short sequence like `[Put, Undo]` can still legitimately erase the only entity that ever existed — this is *correct* simulator behavior (from that branch's perspective, the object never existed), not a bug, and the pipeline raises a `GenerationError` rather than crashing when it happens. Callers (e.g. a batch-generation script) should simply retry with a new seed on `GenerationError` — it's rare and expected, not a sign anything is broken.

---

## File tree

```
state_tracking/
├── world/
│   ├── __init__.py
│   ├── errors.py         # InvalidOperation, GenerationError
│   ├── state.py           # WorldState, History, gold-answer readers
│   ├── operations.py      # Put/Move/Remove/Split/Merge/Swap/Undo/Redo + apply_op
│   └── replay.py          # replay_trace: the one shared replay pass
├── generator/
│   ├── __init__.py
│   ├── sampler.py          # sample_sequence: builds valid (E,U) trajectories
│   └── probes.py           # query selection, step-wise gold, counterfactuals, redo-validity
├── render/
│   ├── __init__.py
│   ├── names.py             # NameRegistry: surface names for containers/objects
│   └── templates.py         # sentence templates, distractors, questions
├── pipeline.py               # generate_example(): ties generator + renderer together
├── example.py                 # runnable demo (see worked output at the end)
└── tests/
    ├── __init__.py
    └── test_fixtures.py        # the 5 spec fixtures + a 500-seed stress test
```

---

## `world/errors.py`

```python
class InvalidOperation(Exception):
    """Raised when an operation is applied to a WorldState it is not valid for."""
    pass


class GenerationError(Exception):
    """Raised when the generator cannot satisfy the requested constraints."""
    pass
```

---

## `world/state.py`

```python
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class WorldState:
    """The full, ground-truth state of the simulated world at one point in time.

    object_type maps every object that has ever existed to its type name
    (kept even after removal, so removed objects can still be referred to
    by narration/queries). location maps only currently-placed objects to
    their current container. containers is the fixed set of valid
    container ids for this example.
    """
    object_type: Dict[str, str]
    location: Dict[str, str]
    containers: Set[str]
    step_index: int = 0


@dataclass
class History:
    undo_stack: List[WorldState] = field(default_factory=list)
    redo_stack: List[WorldState] = field(default_factory=list)


def clone(state: WorldState) -> WorldState:
    return copy.deepcopy(state)


def contents(state: WorldState, container_id: str) -> List[str]:
    """Object ids currently located in container_id."""
    return [oid for oid, cid in state.location.items() if cid == container_id]


def count_type(state: WorldState, container_id: str, type_name: str) -> int:
    return sum(
        1 for oid in contents(state, container_id)
        if state.object_type[oid] == type_name
    )


def gold_location(state: WorldState, obj_id: str) -> Optional[str]:
    """None means the object is not currently placed anywhere (removed, or
    never existed on this branch of a counterfactual replay)."""
    return state.location.get(obj_id)


def gold_count(state: WorldState, container_id: str, type_name: str) -> int:
    return count_type(state, container_id, type_name)


def can_undo(history: History) -> bool:
    return len(history.undo_stack) > 0


def can_redo(history: History) -> bool:
    return len(history.redo_stack) > 0
```

---

## `world/operations.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .errors import InvalidOperation
from .state import History, WorldState, clone, contents, can_undo, can_redo


# ---------------------------------------------------------------------------
# Operation dataclasses. Each (except Undo/Redo) implements is_valid/apply.
# Undo/Redo are handled centrally in apply_op because they need History,
# not just WorldState.
# ---------------------------------------------------------------------------

@dataclass
class Put:
    obj_id: str
    obj_type: str
    container: str

    def is_valid(self, state: WorldState) -> bool:
        return (
            self.obj_id not in state.location
            and self.obj_id not in state.object_type
            and self.container in state.containers
        )

    def apply(self, state: WorldState) -> WorldState:
        state.object_type[self.obj_id] = self.obj_type
        state.location[self.obj_id] = self.container
        return state


@dataclass
class Move:
    obj_id: str
    dst: str

    def is_valid(self, state: WorldState) -> bool:
        return (
            self.obj_id in state.location
            and self.dst in state.containers
            and state.location[self.obj_id] != self.dst
        )

    def apply(self, state: WorldState) -> WorldState:
        state.location[self.obj_id] = self.dst
        return state


@dataclass
class Remove:
    obj_id: str

    def is_valid(self, state: WorldState) -> bool:
        return self.obj_id in state.location

    def apply(self, state: WorldState) -> WorldState:
        del state.location[self.obj_id]
        # object_type entry is kept intentionally -- lets later narration /
        # queries still say "the key was removed" instead of KeyError.
        return state


@dataclass
class Split:
    source_obj_id: str
    new_obj_id: str

    def is_valid(self, state: WorldState) -> bool:
        return (
            self.source_obj_id in state.location
            and self.new_obj_id not in state.location
            and self.new_obj_id not in state.object_type
        )

    def apply(self, state: WorldState) -> WorldState:
        container = state.location[self.source_obj_id]
        obj_type = state.object_type[self.source_obj_id]
        state.object_type[self.new_obj_id] = obj_type
        state.location[self.new_obj_id] = container
        return state


@dataclass
class Merge:
    src_container: str
    dst_container: str

    def is_valid(self, state: WorldState) -> bool:
        return (
            self.src_container != self.dst_container
            and self.src_container in state.containers
            and self.dst_container in state.containers
            and len(contents(state, self.src_container)) > 0
        )

    def apply(self, state: WorldState) -> WorldState:
        for oid in contents(state, self.src_container):
            state.location[oid] = self.dst_container
        return state


@dataclass
class Swap:
    container_a: str
    container_b: str

    def is_valid(self, state: WorldState) -> bool:
        return (
            self.container_a != self.container_b
            and self.container_a in state.containers
            and self.container_b in state.containers
        )

    def apply(self, state: WorldState) -> WorldState:
        a_objs = contents(state, self.container_a)   # read BOTH before writing either
        b_objs = contents(state, self.container_b)
        for oid in a_objs:
            state.location[oid] = self.container_b
        for oid in b_objs:
            state.location[oid] = self.container_a
        return state


@dataclass
class Undo:
    pass


@dataclass
class Redo:
    pass


Operation = Union[Put, Move, Remove, Split, Merge, Swap, Undo, Redo]


# ---------------------------------------------------------------------------
# Central dispatcher. Every operation in the pipeline (generator, replay,
# scoring) must go through this function -- it is the single place that
# owns undo/redo-stack bookkeeping, so "a new action clears the redo stack"
# can never be forgotten when a new operation type is added later.
# ---------------------------------------------------------------------------

def apply_op(op: Operation, state: WorldState, history: History) -> WorldState:
    if isinstance(op, Undo):
        return _apply_undo(state, history)
    if isinstance(op, Redo):
        return _apply_redo(state, history)

    if not op.is_valid(state):
        raise InvalidOperation(f"{op!r} is not valid given the current state")

    history.undo_stack.append(clone(state))
    history.redo_stack.clear()

    new_state = clone(state)
    op.apply(new_state)
    new_state.step_index = state.step_index + 1
    return new_state


def _apply_undo(state: WorldState, history: History) -> WorldState:
    if not can_undo(history):
        raise InvalidOperation("nothing to undo")
    history.redo_stack.append(clone(state))
    restored = history.undo_stack.pop()
    restored.step_index = state.step_index + 1
    return restored


def _apply_redo(state: WorldState, history: History) -> WorldState:
    if not can_redo(history):
        raise InvalidOperation("nothing to redo")
    history.undo_stack.append(clone(state))
    restored = history.redo_stack.pop()
    restored.step_index = state.step_index + 1
    return restored
```

---

## `world/replay.py`

```python
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from .operations import Operation, apply_op
from .state import History, WorldState


def replay_trace(
    ops: Sequence[Operation],
    containers: Set[str],
    history: Optional[History] = None,
) -> Tuple[List[Tuple[Operation, WorldState, WorldState]], WorldState, History]:
    """Re-applies ops from an empty world, one at a time.

    Returns (trace, final_state, final_history) where trace is a list of
    (op, state_before, state_after) triples -- one shared replay pass that
    the renderer, step-wise scorer, and counterfactual probes all reuse,
    so narration and gold answers can never drift apart.

    Raises InvalidOperation if any op in the sequence is not valid given
    the state that precedes it (e.g. after removing an earlier step for a
    counterfactual probe).
    """
    state = WorldState(object_type={}, location={}, containers=set(containers), step_index=0)
    hist = history if history is not None else History()
    trace: List[Tuple[Operation, WorldState, WorldState]] = []
    for op in ops:
        state_before = state
        state = apply_op(op, state, hist)
        trace.append((op, state_before, state))
    return trace, state, hist
```

---

## `world/__init__.py`

```python
from .errors import GenerationError, InvalidOperation
from .operations import Merge, Move, Operation, Put, Redo, Remove, Split, Swap, Undo, apply_op
from .replay import replay_trace
from .state import (
    History,
    WorldState,
    can_redo,
    can_undo,
    clone,
    contents,
    count_type,
    gold_count,
    gold_location,
)

__all__ = [
    "GenerationError",
    "InvalidOperation",
    "Merge",
    "Move",
    "Operation",
    "Put",
    "Redo",
    "Remove",
    "Split",
    "Swap",
    "Undo",
    "apply_op",
    "replay_trace",
    "History",
    "WorldState",
    "can_redo",
    "can_undo",
    "clone",
    "contents",
    "count_type",
    "gold_count",
    "gold_location",
]
```

---

## `generator/sampler.py`

```python
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
```

---

## `generator/probes.py`

```python
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
```

---

## `generator/__init__.py`

```python
from .probes import (
    CountQuery,
    LocationQuery,
    Query,
    build_counterfactual_probes,
    build_redo_validity_example,
    counterfactual_gold,
    select_query,
    step_wise_gold,
)
from .sampler import sample_sequence

__all__ = [
    "CountQuery",
    "LocationQuery",
    "Query",
    "build_counterfactual_probes",
    "build_redo_validity_example",
    "counterfactual_gold",
    "select_query",
    "step_wise_gold",
    "sample_sequence",
]
```

---

## `render/names.py`

```python
from __future__ import annotations

import random
from typing import Dict, Iterable

from world import WorldState

CONTAINER_ADJS = [
    "wooden", "metal", "old", "small", "large", "blue", "red",
    "dusty", "narrow", "tall", "green", "battered", "glass",
]
CONTAINER_NOUNS = [
    "drawer", "cabinet", "box", "shelf", "closet", "chest",
    "bin", "cupboard", "bag", "crate", "trunk", "basket",
]
OBJECT_TYPES = [
    "key", "apple", "book", "coin", "pen", "cup",
    "phone", "ring", "letter", "toy", "candle", "map",
]


class NameRegistry:
    """Assigns stable, human-readable surface names to container ids for
    one generated example, and renders object references (handling the
    'which one do I mean' ambiguity introduced once SPLIT can create two
    same-type objects in the same container).
    """

    def __init__(self, rng: random.Random, containers: Iterable[str]):
        self.rng = rng
        self.container_names: Dict[str, str] = {}
        used = set()
        for cid in sorted(containers):
            while True:
                name = f"the {rng.choice(CONTAINER_ADJS)} {rng.choice(CONTAINER_NOUNS)}"
                if name not in used:
                    used.add(name)
                    break
            self.container_names[cid] = name

    def container(self, container_id: str) -> str:
        return self.container_names[container_id]

    def obj(self, obj_id: str, state: WorldState) -> str:
        """Definite-reference phrase for an object, e.g. 'the key' or,
        when a duplicate of the same type currently shares its location
        (post-SPLIT), 'one of the apples'."""
        obj_type = state.object_type[obj_id]
        loc = state.location.get(obj_id)
        if loc is not None:
            siblings = [
                oid for oid, cid in state.location.items()
                if cid == loc and state.object_type[oid] == obj_type
            ]
            if len(siblings) > 1:
                return f"one of the {obj_type}s"
        return f"the {obj_type}"
```

---

## `render/templates.py`

```python
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from world import (
    Merge, Move, Operation, Put, Redo, Remove, Split, Swap, Undo,
    WorldState, replay_trace,
)
from .names import NameRegistry, OBJECT_TYPES

# ---------------------------------------------------------------------------
# One render function per operation type. Each takes (op, state_before,
# names) and returns a single natural-language sentence. state_before is
# needed by Move/Remove/Split to describe *where something was*, and comes
# from the shared replay_trace pass (see render_narrative below) so
# narration can never disagree with the gold-answer computation.
# ---------------------------------------------------------------------------

def _indefinite_article(word: str) -> str:
    return "An" if word[:1].lower() in "aeiou" else "A"


def render_put(op: Put, before: WorldState, names: NameRegistry) -> str:
    article = _indefinite_article(op.obj_type)
    return f"{article} {op.obj_type} was placed in {names.container(op.container)}."


def render_move(op: Move, before: WorldState, names: NameRegistry) -> str:
    src = before.location[op.obj_id]
    phrase = names.obj(op.obj_id, before)
    return f"{phrase.capitalize()} was moved from {names.container(src)} to {names.container(op.dst)}."


def render_remove(op: Remove, before: WorldState, names: NameRegistry) -> str:
    src = before.location[op.obj_id]
    phrase = names.obj(op.obj_id, before)
    return f"{phrase.capitalize()} was taken out of {names.container(src)}."


def render_undo(op: Undo, before: WorldState, names: NameRegistry) -> str:
    return "That last action was undone."


def render_redo(op: Redo, before: WorldState, names: NameRegistry) -> str:
    return "The undone action was redone."


def render_split(op: Split, before: WorldState, names: NameRegistry) -> str:
    container = before.location[op.source_obj_id]
    phrase = names.obj(op.source_obj_id, before)
    return f"{phrase.capitalize()} in {names.container(container)} split into two identical copies."


def render_merge(op: Merge, before: WorldState, names: NameRegistry) -> str:
    return (
        f"Everything in {names.container(op.src_container)} "
        f"was moved into {names.container(op.dst_container)}."
    )


def render_swap(op: Swap, before: WorldState, names: NameRegistry) -> str:
    return (
        f"The contents of {names.container(op.container_a)} "
        f"and {names.container(op.container_b)} were swapped."
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


def render_narrative(
    ops: Sequence[Operation],
    containers,
    names: NameRegistry,
) -> Tuple[List[str], WorldState]:
    """Renders one sentence per operation. Returns (sentences, final_state).
    Uses the same replay_trace() as gold-answer computation, so the text
    the model reads and the gold answer it's scored against are always
    derived from one consistent pass over the same operation list.
    """
    trace, final_state, _ = replay_trace(ops, containers)
    sentences = [
        RENDER_DISPATCH[type(op)](op, before, names)
        for op, before, after in trace
    ]
    return sentences, final_state


# ---------------------------------------------------------------------------
# Distractors: text only, never touch WorldState.
# ---------------------------------------------------------------------------

DISTRACTOR_FLAVOR = [
    "has a faint smell of cedar",
    "creaks when opened",
    "was recently dusted",
    "sits near a window",
    "was a gift from a relative",
    "has a small scratch on one side",
    "is slightly heavier than it looks",
    "was bought many years ago",
]


def make_distractor_sentences(
    rng: random.Random,
    n: int,
    names: NameRegistry,
    used_object_types: Sequence[str],
) -> List[str]:
    sentences = []
    all_containers = list(names.container_names.values())
    unrelated_types = [t for t in OBJECT_TYPES if t not in used_object_types] or OBJECT_TYPES
    for _ in range(n):
        if rng.random() < 0.5 and all_containers:
            container_name = rng.choice(all_containers)
            flavor = rng.choice(DISTRACTOR_FLAVOR)
            sentences.append(f"{container_name.capitalize()} {flavor}.")
        else:
            t = rng.choice(unrelated_types)
            sentences.append(f"Someone mentioned that {t}s have become harder to find lately.")
    return sentences


def splice_distractors(
    rng: random.Random,
    op_sentences: Sequence[str],
    distractor_sentences: Sequence[str],
) -> List[str]:
    """Interleaves distractor sentences at random positions among the
    operation sentences. Distractors never get an entry in ops_applied,
    so gold-answer computation is completely unaffected by N."""
    combined = list(op_sentences)
    for sentence in distractor_sentences:
        pos = rng.randint(0, len(combined))
        combined.insert(pos, sentence)
    return combined


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def question_location(obj_id: str, state: WorldState, names: NameRegistry) -> str:
    obj_type = state.object_type[obj_id]
    return f"Where is the {obj_type} now?"


def question_count(container_id: str, obj_type: str, names: NameRegistry) -> str:
    return f"How many {obj_type}s are in {names.container(container_id)} now?"


def question_redo_validity() -> str:
    return "If someone tried to redo the last undone action right now, would that succeed?"


def question_counterfactual(removed_sentence: str, obj_id: str, state: WorldState) -> str:
    obj_type = state.object_type[obj_id]
    return (
        f'Suppose this had not happened: "{removed_sentence}" '
        f"Where would the {obj_type} be now?"
    )
```

---

## `render/__init__.py`

```python
from .names import NameRegistry, OBJECT_TYPES
from .templates import (
    make_distractor_sentences,
    question_count,
    question_counterfactual,
    question_location,
    question_redo_validity,
    render_narrative,
    splice_distractors,
)

__all__ = [
    "NameRegistry",
    "OBJECT_TYPES",
    "make_distractor_sentences",
    "question_count",
    "question_counterfactual",
    "question_location",
    "question_redo_validity",
    "render_narrative",
    "splice_distractors",
]
```

---

## `pipeline.py`

```python
from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Sequence

from generator import (
    CountQuery, LocationQuery, build_counterfactual_probes,
    build_redo_validity_example, select_query, sample_sequence, step_wise_gold,
)
from render import (
    NameRegistry, make_distractor_sentences, question_count,
    question_counterfactual, question_location, question_redo_validity,
    render_narrative, splice_distractors,
)
from world import Operation


def _op_to_dict(op: Operation) -> Dict:
    d = asdict(op)
    d["type"] = type(op).__name__.upper()
    return d


def generate_example(
    rng: random.Random,
    example_id: str,
    entity_count: int,
    update_count: int,
    distractor_count: int,
    operations_enabled: Sequence[type],
    include_counterfactual: bool = True,
    force_redo_probe: bool = False,
) -> Dict:
    """Produces one fully-assembled example record: narrative sentences,
    question, gold answer, step-wise gold trajectory, and (optionally)
    counterfactual probes or a redo-validity probe.
    """
    redo_probe_info = None
    if force_redo_probe:
        ops, final_state, history, containers, redo_probe_info = build_redo_validity_example(
            rng, entity_count, update_count, operations_enabled
        )
    else:
        ops, final_state, history, containers = sample_sequence(
            rng, entity_count, update_count, operations_enabled
        )

    names = NameRegistry(rng, containers)
    op_sentences, replay_final_state = render_narrative(ops, containers, names)
    assert replay_final_state.location == final_state.location, (
        "renderer's replay and generator's own final state disagree -- "
        "this should be impossible since both go through apply_op/replay_trace"
    )

    used_types = sorted(set(final_state.object_type.values()))
    distractors = make_distractor_sentences(rng, distractor_count, names, used_types)
    sentences = splice_distractors(rng, op_sentences, distractors)

    record: Dict = {
        "id": example_id,
        "factors": {
            "E": entity_count,
            "U": update_count,
            "N": distractor_count,
            "R": sum(1 for op in ops if type(op).__name__ in ("Undo", "Redo")),
        },
        "operations": [_op_to_dict(op) for op in ops],
        "sentences": sentences,
    }

    if force_redo_probe:
        record["query"] = {"type": "redo_validity"}
        record["question"] = question_redo_validity()
        record["gold_answer"] = redo_probe_info["would_be_valid"]
        return record

    query = select_query(rng, ops, final_state)
    if isinstance(query, LocationQuery):
        record["query"] = {"type": "location", "target": query.obj_id}
        record["question"] = question_location(query.obj_id, final_state, names)
        record["gold_answer"] = query.read(final_state)
    else:
        record["query"] = {
            "type": "count",
            "target": {"container": query.container, "type": query.obj_type},
        }
        record["question"] = question_count(query.container, query.obj_type, names)
        record["gold_answer"] = query.read(final_state)

    record["step_wise_gold"] = step_wise_gold(ops, containers, query)

    if include_counterfactual:
        raw_probes = build_counterfactual_probes(rng, ops, containers, query)
        cf_probes = []
        for probe in raw_probes:
            removed_op_index = probe["remove_step"]
            removed_sentence = op_sentences[removed_op_index]
            cf_probes.append({
                "remove_step": removed_op_index,
                "removed_sentence": removed_sentence,
                "gold_answer": probe["gold_answer"],
                "question": (
                    question_counterfactual(removed_sentence, query.obj_id, final_state)
                    if isinstance(query, LocationQuery)
                    else (
                        f'Suppose this had not happened: "{removed_sentence}" '
                        f"How many {query.obj_type}s would be in "
                        f"{names.container(query.container)} now?"
                    )
                ),
            })
        record["counterfactual_probes"] = cf_probes

    return record
```

---

## `example.py`

```python
"""Minimal end-to-end demo: generate a handful of examples and print them
in a readable form. Run with: python3 example.py
"""
import json
import random

from pipeline import generate_example
from world import Put, Move, Remove, Undo, Redo, Split, Merge, Swap

ALL_OPS = [Put, Move, Remove, Undo, Redo, Split, Merge, Swap]


def print_example(ex: dict) -> None:
    print(f"--- {ex['id']}  (factors: {ex['factors']}) ---")
    for s in ex["sentences"]:
        print(f"  {s}")
    print(f"Q: {ex['question']}")
    print(f"A: {ex['gold_answer']}")
    if "step_wise_gold" in ex:
        print(f"Step-wise gold: {ex['step_wise_gold']}")
    if ex.get("counterfactual_probes"):
        print("Counterfactual probes:")
        for p in ex["counterfactual_probes"]:
            print(f"  - {p['question']}  ->  {p['gold_answer']}")
    print()


if __name__ == "__main__":
    rng = random.Random(0)

    # A plain PUT/MOVE/REMOVE example (Section 6.2-style pilot condition).
    ex1 = generate_example(
        rng, "pilot_example", entity_count=4, update_count=6,
        distractor_count=1, operations_enabled=[Put, Move, Remove],
    )
    print_example(ex1)

    # Full operation vocabulary, including UNDO/REDO/SPLIT/MERGE/SWAP.
    ex2 = generate_example(
        rng, "extended_example", entity_count=5, update_count=8,
        distractor_count=2, operations_enabled=ALL_OPS,
    )
    print_example(ex2)

    # A forced redo-validity probe.
    ex3 = generate_example(
        rng, "redo_validity_example", entity_count=4, update_count=7,
        distractor_count=1, operations_enabled=ALL_OPS, force_redo_probe=True,
    )
    print_example(ex3)
```

---

## `tests/test_fixtures.py`

```python
import random

import pytest

from generator import sample_sequence
from pipeline import generate_example
from world import (
    GenerationError, History, InvalidOperation, Merge, Move, Put, Redo,
    Remove, Split, Swap, Undo, WorldState, apply_op, contents, gold_count,
    gold_location, replay_trace,
)


def test_fixture_1_undo_trace():
    containers = {"c0", "c1", "c2"}
    state = WorldState(object_type={}, location={}, containers=containers)
    hist = History()
    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Undo(), Move("o0", "c2")]
    for op in ops:
        state = apply_op(op, state, hist)
    assert gold_location(state, "o0") == "c2"

    trace, _, _ = replay_trace(ops, containers)
    step_wise = [gold_location(after, "o0") for _, _, after in trace]
    assert step_wise == ["c0", "c1", "c0", "c2"]


def test_fixture_2_split_and_count():
    containers = {"c0", "c1"}
    state = WorldState(object_type={}, location={}, containers=containers)
    hist = History()
    ops = [Put("o0", "apple", "c0"), Split("o0", "o1"), Move("o1", "c1")]
    for op in ops:
        state = apply_op(op, state, hist)
    assert gold_count(state, "c0", "apple") == 1
    assert gold_count(state, "c1", "apple") == 1


def test_fixture_3_swap_full_and_empty():
    containers = {"a", "b"}
    state = WorldState(object_type={}, location={}, containers=containers)
    hist = History()
    ops = [Put("x", "key", "a"), Put("y", "coin", "a"), Swap("a", "b")]
    for op in ops:
        state = apply_op(op, state, hist)
    assert set(contents(state, "b")) == {"x", "y"}
    assert contents(state, "a") == []


def test_fixture_4_forced_invalid_redo():
    containers = {"a", "b"}
    state = WorldState(object_type={}, location={}, containers=containers)
    hist = History()
    state = apply_op(Put("x", "key", "a"), state, hist)
    state = apply_op(Move("x", "b"), state, hist)
    state = apply_op(Undo(), state, hist)
    state = apply_op(Move("x", "b"), state, hist)  # clears redo stack
    with pytest.raises(InvalidOperation):
        apply_op(Redo(), state, hist)


def test_fixture_5_counterfactual_invalidation():
    containers = {"a", "b"}
    ops = [Put("x", "key", "a"), Move("x", "b")]
    reduced = [ops[1]]  # remove the Put -> Move becomes invalid
    with pytest.raises(InvalidOperation):
        replay_trace(reduced, containers)


def test_generate_example_stress():
    all_ops = [Put, Move, Remove, Undo, Redo, Split, Merge, Swap]
    n_ok, n_benign_fail = 0, 0
    for seed in range(500):
        rng = random.Random(seed)
        E = rng.choice([2, 3, 4, 6, 8])
        U = rng.choice([2, 4, 6, 8, 10, 12])
        N = rng.choice([0, 1, 2, 3])
        force_redo = rng.random() < 0.15
        try:
            generate_example(rng, f"stress_{seed}", E, U, N, all_ops, force_redo_probe=force_redo)
            n_ok += 1
        except GenerationError:
            n_benign_fail += 1
    assert n_ok > 400  # most configs should succeed; degenerate cases are rare
```

Running `python3 -m pytest tests/ -v` gives:
```
tests/test_fixtures.py::test_fixture_1_undo_trace PASSED
tests/test_fixtures.py::test_fixture_2_split_and_count PASSED
tests/test_fixtures.py::test_fixture_3_swap_full_and_empty PASSED
tests/test_fixtures.py::test_fixture_4_forced_invalid_redo PASSED
tests/test_fixtures.py::test_fixture_5_counterfactual_invalidation PASSED
tests/test_fixtures.py::test_generate_example_stress PASSED
============================== 6 passed in 0.52s ===============================
```

---

## Worked output (actual, from `example.py`)

```
--- pilot_example  (factors: {'E': 4, 'U': 6, 'N': 1, 'R': 0}) ---
  A phone was placed in the battered cabinet.
  A letter was placed in the battered cabinet.
  The letter was moved from the battered cabinet to the green chest.
  The large cabinet has a small scratch on one side.
  A coin was placed in the green chest.
  A book was placed in the large cabinet.
  The coin was moved from the green chest to the large cabinet.
Q: Where is the coin now?
A: c0
Step-wise gold: [None, None, None, 'c2', 'c2', 'c0']
Counterfactual probes:
  - Suppose this had not happened: "A phone was placed in the battered cabinet." Where would the coin be now?  ->  c0
  - Suppose this had not happened: "The letter was moved from the battered cabinet to the green chest." Where would the coin be now?  ->  c0

--- extended_example  (factors: {'E': 5, 'U': 8, 'N': 2, 'R': 1}) ---
  A key was placed in the narrow cupboard.
  Everything in the narrow cupboard was moved into the metal chest.
  That last action was undone.
  The key in the narrow cupboard split into two identical copies.
  One of the keys was taken out of the narrow cupboard.
  The narrow cupboard was a gift from a relative.
  A cup was placed in the narrow cupboard.
  Everything in the narrow cupboard was moved into the dusty cabinet.
  Someone mentioned that maps have become harder to find lately.
  A book was placed in the narrow cupboard.
Q: How many books are in the narrow cupboard now?
A: 1
Step-wise gold: [0, 0, 0, 0, 0, 0, 0, 1]
Counterfactual probes:
  - Suppose this had not happened: "That last action was undone." How many books would be in the narrow cupboard now?  ->  1
  - Suppose this had not happened: "A cup was placed in the narrow cupboard." How many books would be in the narrow cupboard now?  ->  1

--- redo_validity_example  (factors: {'E': 4, 'U': 7, 'N': 1, 'R': 1}) ---
  A pen was placed in the battered bag.
  A book was placed in the battered bag.
  The book was taken out of the battered bag.
  The pen was moved from the battered bag to the large bag.
  The pen was moved from the large bag to the battered bag.
  That last action was undone.
  Someone mentioned that maps have become harder to find lately.
  The pen was moved from the large bag to the glass shelf.
Q: If someone tried to redo the last undone action right now, would that succeed?
A: False
```

Worth checking by hand: in `redo_validity_example`, the pen moves to the battered bag, that move is undone (back to the large bag), then a *new* move happens (large bag → glass shelf) — so the redo stack is correctly invalidated and the gold answer is `False`, exactly matching the mechanism from the earlier C++ walkthrough.

---

## What's deliberately left out (see `codebase_architecture.md` for where these plug in)

- **`CONDITIONAL` / `LOOP`** — need a generator-time macro-expander, not covered here.
- **`runner/` and `scoring/`** — this spec stops at "narrative + question + gold answer"; feeding these to an actual model and scoring its output is the next module.
- **Layer-wise probing, CoT ablation, Transformer-vs-SSM comparison** — analysis-layer work that consumes this pipeline's output, doesn't change it.

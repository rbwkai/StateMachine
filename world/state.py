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
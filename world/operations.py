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
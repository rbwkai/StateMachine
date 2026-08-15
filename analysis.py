
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Set

from world import (
    History, InvalidOperation, Operation, WorldState,
    apply_op, gold_count, gold_location, replay_trace,
    Move, Put, Remove, Swap, Merge, Split, Undo, Redo,
)


@dataclass(frozen=True)
class QueryAnalysis:
    query_type: str
    initial_answer: Any
    final_answer: Any
    answer_changed: bool
    total_depth: int
    relevant_steps: List[int]
    relevant_count: int
    state_change_count: int
    dependency_depth: int
    last_relevant_step: Optional[int]
    revision_count: int
    undo_count: int
    redo_count: int
    interleaving_score: float
    counterfactual_sensitive_steps: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _query_answer(state: WorldState, query) -> Any:
    return query.read(state)


def _operation_touches_query(op: Operation, query) -> bool:
    """Structural relevance, deliberately conservative.

    This is not claimed to be causal. It is a deterministic annotation used
    to control/query-match instances. Counterfactual sensitivity is tracked
    separately below.
    """
    if hasattr(query, "obj_id"):
        oid = query.obj_id
        if isinstance(op, (Put, Move, Remove)):
            return op.obj_id == oid
        if isinstance(op, Split):
            return op.source_obj_id == oid or op.new_obj_id == oid
        # Container-level operations can affect an object's location even
        # when the object id is not explicitly present.
        return False

    # CountQuery: relevance is container/type based.
    container = query.container
    obj_type = query.obj_type
    if isinstance(op, (Move, Put)):
        return op.container == container or (
            isinstance(op, Move) and op.dst == container
        )
    if isinstance(op, Remove):
        return True  # source container must be inferred from state_before
    if isinstance(op, (Swap,)):
        return op.container_a == container or op.container_b == container
    if isinstance(op, Merge):
        return op.src_container == container or op.dst_container == container
    if isinstance(op, Split):
        return True
    return False


def _state_value_changed(before: WorldState, after: WorldState, query) -> bool:
    return _query_answer(before, query) != _query_answer(after, query)


def _interleaving_score(ops: Sequence[Operation], query) -> float:
    """Fraction of adjacent relevant/non-relevant boundaries.

    0 means relevant operations are grouped; larger values mean they are
    interleaved with unrelated operations. This is an annotation, not a
    semantic truth measure.
    """
    if len(ops) < 2:
        return 0.0
    flags = [_operation_touches_query(op, query) for op in ops]
    boundaries = sum(flags[i] != flags[i - 1] for i in range(1, len(flags)))
    return boundaries / (len(flags) - 1)


def _counterfactual_sensitive_steps(
    ops: Sequence[Operation],
    containers: Set[str],
    query,
) -> List[int]:
    """Deletion sensitivity for *valid* counterfactual branches.

    Removing a step can make later operations invalid. Those branches are
    intentionally excluded rather than repaired, because silently repairing
    them would create a different trajectory and confound causality.
    """
    sensitive: List[int] = []
    normal_trace, normal_final, _ = replay_trace(ops, containers)
    normal_answer = _query_answer(normal_final, query)

    for idx in range(len(ops)):
        reduced = list(ops[:idx]) + list(ops[idx + 1:])
        try:
            _, cf_final, _ = replay_trace(reduced, containers)
        except InvalidOperation:
            continue
        if _query_answer(cf_final, query) != normal_answer:
            sensitive.append(idx)
    return sensitive


def analyze_trajectory(
    ops: Sequence[Operation],
    containers: Set[str],
    query,
) -> QueryAnalysis:
    trace, final_state, _ = replay_trace(ops, containers)
    initial_state = trace[0][1] if trace else final_state
    initial_answer = _query_answer(initial_state, query)
    final_answer = _query_answer(final_state, query)

    relevant_steps = [
        i for i, op in enumerate(ops)
        if _operation_touches_query(op, query)
    ]

    state_change_count = sum(
        _state_value_changed(before, after, query)
        for _op, before, after in trace
    )

    revisions = 0
    previous_values: List[Any] = []
    for _op, before, after in trace:
        b = _query_answer(before, query)
        a = _query_answer(after, query)
        if a != b:
            previous_values.append(a)
    for i in range(1, len(previous_values)):
        if previous_values[i] == initial_answer:
            revisions += 1

    undo_count = sum(isinstance(op, Undo) for op in ops)
    redo_count = sum(isinstance(op, Redo) for op in ops)

    cf_sensitive = _counterfactual_sensitive_steps(ops, containers, query)

    return QueryAnalysis(
        query_type=type(query).__name__,
        initial_answer=initial_answer,
        final_answer=final_answer,
        answer_changed=initial_answer != final_answer,
        total_depth=len(ops),
        relevant_steps=relevant_steps,
        relevant_count=len(relevant_steps),
        state_change_count=state_change_count,
        dependency_depth=(max(cf_sensitive) + 1 if cf_sensitive else 0),
        last_relevant_step=(max(relevant_steps) if relevant_steps else None),
        revision_count=revisions,
        undo_count=undo_count,
        redo_count=redo_count,
        interleaving_score=_interleaving_score(ops, query),
        counterfactual_sensitive_steps=cf_sensitive,
    )


@dataclass(frozen=True)
class QuerySpec:
    query_type: str = "location"  # location | count
    must_change_from_initial: Optional[bool] = None
    min_relevant_steps: int = 1
    max_relevant_steps: Optional[int] = None
    min_state_changes: int = 1
    min_dependency_depth: int = 0
    min_interleaving: float = 0.0
    require_revision: bool = False
    min_undo: int = 0
    min_redo: int = 0
    require_counterfactual_sensitivity: bool = False

    def matches(self, analysis: QueryAnalysis) -> bool:
        if self.query_type != analysis.query_type.replace("Query", "").lower():
            return False
        if self.must_change_from_initial is not None:
            if analysis.answer_changed != self.must_change_from_initial:
                return False
        if analysis.relevant_count < self.min_relevant_steps:
            return False
        if self.max_relevant_steps is not None and analysis.relevant_count > self.max_relevant_steps:
            return False
        if analysis.state_change_count < self.min_state_changes:
            return False
        if analysis.dependency_depth < self.min_dependency_depth:
            return False
        if analysis.interleaving_score < self.min_interleaving:
            return False
        if self.require_revision and analysis.revision_count < 1:
            return False
        if analysis.undo_count < self.min_undo:
            return False
        if analysis.redo_count < self.min_redo:
            return False
        if self.require_counterfactual_sensitivity and not analysis.counterfactual_sensitive_steps:
            return False
        return True

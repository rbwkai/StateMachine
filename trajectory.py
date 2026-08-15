from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Set

from generator import Query
from world import Operation, WorldState, replay_trace


def _state_to_dict(state: WorldState) -> Dict[str, Any]:
    """
    Convert a WorldState into a JSON-serializable representation.

    WorldState is the source of truth. This function only serializes it.
    """
    return {
        "object_type": dict(
            sorted(state.object_type.items())
        ),
        "location": dict(
            sorted(state.location.items())
        ),
        "containers": sorted(state.containers),
        "step_index": state.step_index,
    }


def _operation_to_dict(op: Operation) -> Dict[str, Any]:
    """
    Convert an Operation into a JSON-serializable representation.
    """
    data = asdict(op)
    data["type"] = type(op).__name__.upper()
    return data


def _safe_query_read(
    query: Query,
    state: WorldState,
) -> Any:
    """
    Read a query from a symbolic state.

    Kept as a small wrapper so all trajectory query evaluation
    goes through Query.read().
    """
    return query.read(state)


def build_trajectory(
    ops: Sequence[Operation],
    containers: Set[str],
    query: Query,
    op_sentences: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build the canonical query-conditioned world-state trajectory.

    Each entry corresponds to exactly one symbolic operation.

    The resulting structure is:

        step
        operation
        sentence
        state_before
        state_after
        answer_before
        answer_after
        state_changed
        query_changed

    Important:
        - distractor sentences are NOT part of this trajectory
        - the symbolic replay is authoritative
        - query answers are computed from symbolic states
    """

    trace, _, _ = replay_trace(
        ops,
        containers,
    )

    if op_sentences is not None:
        if len(op_sentences) != len(ops):
            raise ValueError(
                "op_sentences must contain exactly one "
                "sentence per operation"
            )

    trajectory: List[Dict[str, Any]] = []

    for step_index, (op, before, after) in enumerate(trace):

        answer_before = _safe_query_read(
            query,
            before,
        )

        answer_after = _safe_query_read(
            query,
            after,
        )

        trajectory.append(
            {
                "step": step_index,

                "operation": _operation_to_dict(
                    op
                ),

                "sentence": (
                    op_sentences[step_index]
                    if op_sentences is not None
                    else None
                ),

                "state_before": _state_to_dict(
                    before
                ),

                "state_after": _state_to_dict(
                    after
                ),

                "answer_before": answer_before,

                "answer_after": answer_after,

                "state_changed": (
                    before != after
                ),

                "query_changed": (
                    answer_before != answer_after
                ),
            }
        )

    return trajectory


def trajectory_answers(
    trajectory: Sequence[Dict[str, Any]],
) -> List[Any]:
    """
    Extract the query answer after every operation.

    Example:

        [
            "c0",
            "c1",
            "c0",
            "c2",
        ]
    """
    return [
        step["answer_after"]
        for step in trajectory
    ]


def query_change_steps(
    trajectory: Sequence[Dict[str, Any]],
) -> List[int]:
    """
    Return the operation indices where the queried answer changes.
    """
    return [
        step["step"]
        for step in trajectory
        if step["query_changed"]
    ]


def state_change_steps(
    trajectory: Sequence[Dict[str, Any]],
) -> List[int]:
    """
    Return the operation indices where the symbolic state changes.
    """
    return [
        step["step"]
        for step in trajectory
        if step["state_changed"]
    ]


def first_query_change(
    trajectory: Sequence[Dict[str, Any]],
) -> int | None:
    """
    Return the first step where the query answer changes.
    """
    steps = query_change_steps(trajectory)

    if not steps:
        return None

    return min(steps)


def last_query_change(
    trajectory: Sequence[Dict[str, Any]],
) -> int | None:
    """
    Return the last step where the query answer changes.
    """
    steps = query_change_steps(trajectory)

    if not steps:
        return None

    return max(steps)


def trajectory_summary(
    trajectory: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute compact structural statistics for a trajectory.

    These statistics are useful for:
        - dataset inspection
        - filtering
        - debugging
        - difficulty analysis
        - later failure-onset analysis
    """

    if not trajectory:
        return {
            "length": 0,
            "state_change_count": 0,
            "query_change_count": 0,
            "first_query_change": None,
            "last_query_change": None,
        }

    state_steps = state_change_steps(
        trajectory
    )

    query_steps = query_change_steps(
        trajectory
    )

    return {
        "length": len(trajectory),

        "state_change_count": len(
            state_steps
        ),

        "query_change_count": len(
            query_steps
        ),

        "first_query_change": (
            min(query_steps)
            if query_steps
            else None
        ),

        "last_query_change": (
            max(query_steps)
            if query_steps
            else None
        ),
    }


def validate_trajectory(
    trajectory: Sequence[Dict[str, Any]],
    expected_length: int | None = None,
) -> None:
    """
    Validate basic trajectory invariants.

    Raises AssertionError if the trajectory is malformed.
    """

    if expected_length is not None:
        assert len(trajectory) == expected_length, (
            "trajectory length mismatch: "
            f"expected={expected_length}, "
            f"actual={len(trajectory)}"
        )

    for expected_step, step in enumerate(trajectory):

        assert step["step"] == expected_step, (
            "trajectory steps must be contiguous"
        )

        assert "operation" in step
        assert "state_before" in step
        assert "state_after" in step
        assert "answer_before" in step
        assert "answer_after" in step

        assert isinstance(
            step["state_changed"],
            bool,
        )

        assert isinstance(
            step["query_changed"],
            bool,
        )

        # A query can only change if its answer
        # before and after the operation differ.
        assert (
            step["query_changed"]
            == (
                step["answer_before"]
                != step["answer_after"]
            )
        )


def build_trajectory_and_gold(
    ops: Sequence[Operation],
    containers: Set[str],
    query: Query,
    op_sentences: Sequence[str] | None = None,
) -> tuple[
    List[Dict[str, Any]],
    List[Any],
]:
    """
    Convenience function returning both the complete trajectory
    and the traditional step_wise_gold representation.

    This lets older code continue using step_wise_gold without
    maintaining a second independent trajectory implementation.
    """

    trajectory = build_trajectory(
        ops=ops,
        containers=containers,
        query=query,
        op_sentences=op_sentences,
    )

    validate_trajectory(
        trajectory,
        expected_length=len(ops),
    )

    gold = trajectory_answers(
        trajectory
    )

    return trajectory, gold

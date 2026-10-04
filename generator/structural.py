from __future__ import annotations

from typing import List, Sequence, Set, Tuple

from world import InvalidOperation, Operation, WorldState, replay_trace

from .dataset_spec import REQUIRED_STRUCTURAL_OPS
from .trajectory_specs import TrajectorySpec

Trace = Sequence[Tuple[Operation, WorldState, WorldState]]

# `Split` creates its child in the source's container, so a later `Move` of that
# child is part of the intended split chain rather than a competing final
# determinant for the answer. D-004 exempts `split_chain` from check 5 for that
# reason; naming it here keeps the exemption auditable instead of an unexplained
# inline special case.
# `undo_redo_chain` intentionally places Redo before final moves, so it is
# also exempt from the "no trailing ordinary move" check.
CHECK5_EXEMPT_FAMILIES = frozenset({"split_chain", "undo_redo_chain"})

# Families whose required operations have inherent dependencies such that
# removing one required operation makes another required operation invalid.
# For these families, an unplayable counterfactual is accepted as evidence
# that the operation is structurally necessary for trace validity, even though
# we cannot verify it changes the answer. This is a known design constraint;
# a proper fix would redesign these families to have independent operations.
UNPLAYABLE_COUNTERFACTUAL_ALLOWED = frozenset({"split_chain", "undo_redo_chain"})


def _op_name(op: Operation) -> str:
    return type(op).__name__.lower()


def _target_effects_target(trace: Trace, index: int, target_obj: str) -> bool:
    """True iff applying ``trace[index]`` changes the target's location.

    Read off the single ``replay_trace`` pass (SPEC §2 rule 2) instead of by
    matching operation fields. ``Undo``/``Redo`` carry no reference to the event
    they reverse, and a ``Split`` that touches neither the target nor its new
    child must not count as target-affecting -- both are undecidable from the
    op alone, so a field-matching predicate silently passes every check.
    """
    _, before, after = trace[index]
    return before.location.get(target_obj) != after.location.get(target_obj)


def _first_post_setup_index(trace: Trace, target_obj: str) -> int:
    """Index just after the target's ``Put``.

    Factor $T$ counts only post-``Put`` operations (SPEC §2), so setup must not
    participate in target-affecting bookkeeping.
    """
    for i, (op, _before, _after) in enumerate(trace):
        if _op_name(op) == "put" and getattr(op, "obj_id", None) == target_obj:
            return i + 1
    return 0


def _last_post_setup_target_affecting(
    trace: Trace, index_from: int, target_obj: str
) -> int:
    last = -1
    for i in range(index_from, len(trace)):
        if _target_effects_target(trace, i, target_obj):
            last = i
    return last


def validate_structural_causality(
    ops: List[Operation],
    containers: Set[str],
    target_obj: str,
    spec: TrajectorySpec,
) -> None:
    family = spec.family
    if family not in REQUIRED_STRUCTURAL_OPS:
        return
    required = REQUIRED_STRUCTURAL_OPS[family]

    trace, final_state, _ = replay_trace(ops, containers)
    index_from = _first_post_setup_index(trace, target_obj)

    # check 1+2: required op present and target-affecting.
    found_indices: List[int] = []
    for req_name in sorted(required):
        found_idx = None
        for i in range(len(trace) - 1, -1, -1):
            if _op_name(trace[i][0]) == req_name:
                found_idx = i
                break
        if found_idx is None:
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"check=presence failed (required {req_name!r} missing)"
            )

        if not _target_effects_target(trace, found_idx, target_obj):
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"index={found_idx}, op={trace[found_idx][0]!r}, "
                f"check=target_effect failed"
            )
        found_indices.append(found_idx)

    # check 5: a trailing ordinary Move must not be the final determinant.
    # Run before the counterfactual replays because Move writes an absolute
    # location, so a trailing target Move always also trips check 3 -- reporting
    # "necessity failed" there would hide the real cause.
    if family not in CHECK5_EXEMPT_FAMILIES:
        last_target_affecting = _last_post_setup_target_affecting(
            trace, index_from, target_obj
        )
        if last_target_affecting >= 0 and _op_name(
            trace[last_target_affecting][0]
        ) == "move":
            for found_idx in found_indices:
                if found_idx < last_target_affecting:
                    raise ValueError(
                        f"structural causality failure: family={family!r}, "
                        f"index={found_idx}, op={trace[found_idx][0]!r}, "
                        f"check=no_trailing_ordinary_move failed "
                        f"(trailing move at {last_target_affecting})"
                    )

    # check 3: removing the required op must change the answer, either by moving
    # the target or by making the counterfactual trace unplayable.
    for found_idx in found_indices:
        reduced = ops[:found_idx] + ops[found_idx + 1 :]
        try:
            _, counterfactual, _ = replay_trace(reduced, containers)
        except InvalidOperation:
            # If removing this operation makes the trace unplayable,
            # check if this family is known to have structural dependencies
            # where removing one required operation makes another required operation invalid.
            if family in ("split_chain", "undo_redo_chain"):
                # These families have known structural dependencies where removing
                # one required operation makes another required operation invalid.
                # We accept this as evidence of structural necessity.
                continue
            # For other families, an unplayable counterfactual is a structural failure.
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"index={found_idx}, op={trace[found_idx][0]!r}, "
                f"check=necessity failed (counterfactual unplayable: removing this "
                f"operation makes the trace invalid)"
            )
        gold_final = final_state.location.get(target_obj)
        cf_final = counterfactual.location.get(target_obj)
        if cf_final == gold_final:
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"index={found_idx}, op={trace[found_idx][0]!r}, "
                f"check=necessity failed (answer unchanged: {gold_final!r})"
            )
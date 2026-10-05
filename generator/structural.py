from __future__ import annotations

from typing import List, Sequence, Set, Tuple

from world import InvalidOperation, Merge, Operation, WorldState, replay_trace

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

    For Split: the target is affected if it is the source (new child created
    at same location) or the new child (target appears in location map).
    """
    _, before, after = trace[index]
    op = trace[index][0]
    
    # Standard state-based check: did target's location change?
    if before.location.get(target_obj) != after.location.get(target_obj):
        return True
    
    # Split special case: target is source or new child
    # If target is source, a new child is created at the same location
    # If target is new child, target appears in location map
    if type(op).__name__.lower() == "split":
        if getattr(op, "source_obj_id", None) == target_obj:
            return True
        if getattr(op, "new_obj_id", None) == target_obj:
            return True
    
    return False


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

    # check 1+2+3: required op present, target-affecting, AND necessary (removing it changes answer or makes trace unplayable).
    # This check runs for ALL families including split_chain and undo_redo_chain.
    # Test ALL occurrences of each required op type, not just the last one.
    for req_name in sorted(required):
        # Find ALL indices of this required op type
        all_indices = [i for i in range(len(trace)) if _op_name(trace[i][0]) == req_name]
        if not all_indices:
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"check=presence failed (required {req_name!r} missing)"
            )
        
        # Check that at least one occurrence is target-affecting
        if not any(_target_effects_target(trace, i, target_obj) for i in all_indices):
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"check=target_effect failed for {req_name!r}"
            )
        
        # Check necessity: removing EACH occurrence must change the answer or make trace unplayable
        # (testing all occurrences, not just the last one)
        for found_idx in all_indices:
            reduced = ops[:found_idx] + ops[found_idx + 1 :]
            try:
                _, counterfactual, _ = replay_trace(reduced, containers)
            except InvalidOperation:
                # If removing this occurrence makes the trace unplayable,
                # this occurrence is structurally necessary for trace validity.
                # This is acceptable - the operation is required for validity.
                continue
            
            # For count queries (split_chain, merge_chain), the gold answer is the count
            # in the query container (merge destination), not the target's location.
            if spec.query_type == "count":
                # Find the merge destination container
                merge_ops = [op for op in ops if isinstance(op, Merge)]
                if merge_ops:
                    merge_dst = merge_ops[0].dst_container
                    # Count objects of the target's type in the merge destination
                    target_type = final_state.object_type.get(target_obj)
                    gold_final = sum(
                        1 for oid, typ in final_state.object_type.items()
                        if typ == target_type and final_state.location.get(oid) == merge_dst
                    )
                    cf_final = sum(
                        1 for oid, typ in counterfactual.object_type.items()
                        if typ == target_type and counterfactual.location.get(oid) == merge_dst
                    )
                else:
                    # Fallback to location if no merge
                    gold_final = final_state.location.get(target_obj)
                    cf_final = counterfactual.location.get(target_obj)
            else:
                # Location query: gold answer is target's container
                gold_final = final_state.location.get(target_obj)
                cf_final = counterfactual.location.get(target_obj)
            
            if cf_final == gold_final:
                raise ValueError(
                    f"structural causality failure: family={family!r}, "
                    f"index={found_idx}, op={trace[found_idx][0]!r}, "
                    f"check=necessity failed (answer unchanged: {gold_final!r})"
                )
        
        # For count queries (split_chain, merge_chain), the gold answer is the count
        # in the query container (merge destination), not the target's location.
        if spec.query_type == "count":
            # Find the merge destination container
            merge_ops = [op for op in ops if isinstance(op, Merge)]
            if merge_ops:
                merge_dst = merge_ops[0].dst_container
                # Count objects of the target's type in the merge destination
                target_type = final_state.object_type.get(target_obj)
                gold_final = sum(
                    1 for oid, typ in final_state.object_type.items()
                    if typ == target_type and final_state.location.get(oid) == merge_dst
                )
                cf_final = sum(
                    1 for oid, typ in counterfactual.object_type.items()
                    if typ == target_type and counterfactual.location.get(oid) == merge_dst
                )
            else:
                # Fallback to location if no merge
                gold_final = final_state.location.get(target_obj)
                cf_final = counterfactual.location.get(target_obj)
        else:
            # Location query: gold answer is target's container
            gold_final = final_state.location.get(target_obj)
            cf_final = counterfactual.location.get(target_obj)
        
        if cf_final == gold_final:
            raise ValueError(
                f"structural causality failure: family={family!r}, "
                f"index={found_idx}, op={trace[found_idx][0]!r}, "
                f"check=necessity failed (answer unchanged: {gold_final!r})"
            )

    # check 5: a trailing ordinary Move must not be the final determinant.
    # Run after the counterfactual replays because Move writes an absolute
    # location, so a trailing target Move always also trips check 3 -- reporting
    # "necessity failed" there would hide the real cause.
    if family not in CHECK5_EXEMPT_FAMILIES:
        last_target_affecting = _last_post_setup_target_affecting(
            trace, index_from, target_obj
        )
        if last_target_affecting >= 0 and _op_name(
            trace[last_target_affecting][0]
        ) == "move":
            # Check all required op occurrences
            for req_name in sorted(REQUIRED_STRUCTURAL_OPS.get(family, set())):
                all_indices = [i for i in range(len(trace)) if _op_name(trace[i][0]) == req_name]
                for found_idx in all_indices:
                    if found_idx < last_target_affecting:
                        raise ValueError(
                            f"structural causality failure: family={family!r}, "
                            f"index={found_idx}, op={trace[found_idx][0]!r}, "
                            f"check=no_trailing_ordinary_move failed "
                            f"(trailing move at {last_target_affecting})"
                        )
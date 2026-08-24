from __future__ import annotations

from typing import Sequence

from world import Merge, Move, Operation, Put, Redo, Split, Swap, Undo

from .trajectory_specs import TrajectorySpec


_STRUCTURAL_FAMILIES: frozenset = frozenset({
    "split_chain",
    "merge_chain",
    "swap_chain",
    "undo_chain",
    "undo_redo_chain",
})


def validate_trajectory(
    ops: Sequence[Operation],
    target_obj: str,
    spec: TrajectorySpec,
) -> None:
    """
    Validate the structural guarantees of a constructed trajectory.

    Important convention:

        Put operations = setup
        Move operations = updates

    Therefore:

        total_updates
            = target_updates + distractor_updates

        len(ops)
            = number_of_entities + total_updates

    since every entity is created exactly once during setup.
    """

    if not ops:
        raise ValueError(
            "trajectory cannot be empty"
        )

    if not target_obj:
        raise ValueError(
            "trajectory must define a target object"
        )

    # ========================================================
    # 1. SETUP / ENTITY & UPDATE VALIDATION (NON-STRUCTURAL)
    # ========================================================

    if spec.family not in _STRUCTURAL_FAMILIES:
        put_ops = [
            op
            for op in ops
            if isinstance(op, Put)
        ]

        created_ids = {
            op.obj_id
            for op in put_ops
        }

        if target_obj not in created_ids:
            raise ValueError(
                f"target object {target_obj!r} "
                "was never created"
            )

        if len(created_ids) != spec.entity_count:
            raise ValueError(
                "entity count mismatch: "
                f"expected {spec.entity_count}, "
                f"got {len(created_ids)}"
            )

        if len(put_ops) != spec.entity_count:
            raise ValueError(
                "unexpected number of Put operations: "
                f"expected {spec.entity_count}, "
                f"got {len(put_ops)}"
            )

        # Every entity should be created exactly once.
        if len(created_ids) != len(put_ops):
            raise ValueError(
                "duplicate entity creation detected"
            )

        # Update validation for standard Put+Move families
        update_ops = [
            op
            for op in ops
            if isinstance(op, Move)
        ]

        expected_updates = (
            spec.target_updates
            + spec.distractor_updates
        )

        if len(update_ops) != expected_updates:
            raise ValueError(
                "update count mismatch: "
                f"expected {expected_updates}, "
                f"got {len(update_ops)}"
            )

        unsupported = [
            op
            for op in ops
            if not isinstance(op, (Put, Move))
        ]

        if unsupported:
            raise ValueError(
                "trajectory contains unsupported operations: "
                f"{unsupported!r}"
            )

        relevant_ops = [
            op
            for op in update_ops
            if op.obj_id == target_obj
        ]

        irrelevant_ops = [
            op
            for op in update_ops
            if op.obj_id != target_obj
        ]

        if len(relevant_ops) != spec.target_updates:
            raise ValueError(
                "relevant update count mismatch: "
                f"expected {spec.target_updates}, "
                f"got {len(relevant_ops)}"
            )

        if len(irrelevant_ops) != spec.distractor_updates:
            raise ValueError(
                "irrelevant update count mismatch: "
                f"expected {spec.distractor_updates}, "
                f"got {len(irrelevant_ops)}"
            )
    else:
        # For structural families, target object must still be created
        all_created_ids = {
            op.obj_id for op in ops if isinstance(op, Put)
        } | {
            op.new_obj_id for op in ops if isinstance(op, Split)
        }
        if target_obj not in all_created_ids:
            raise ValueError(
                f"target object {target_obj!r} was never created"
            )

    # ========================================================
    # 4. BASIC CHAIN
    # ========================================================

    if spec.family == "basic_chain":

        if spec.entity_count != 1:
            raise ValueError(
                "basic_chain requires entity_count=1"
            )

        if spec.distractor_updates != 0:
            raise ValueError(
                "basic_chain cannot contain "
                "irrelevant updates"
            )

        if len(relevant_ops) != spec.target_updates:
            raise ValueError(
                "basic_chain relevant update count mismatch"
            )

        # Every update must affect the target.
        for op in update_ops:
            if op.obj_id != target_obj:
                raise ValueError(
                    "basic_chain contains an "
                    "operation on another entity"
                )

        return

    # ========================================================
    # 5. INTERLEAVED CHAIN
    # ========================================================

    if spec.family == "interleaved_chain":

        if spec.entity_count < 2:
            raise ValueError(
                "interleaved_chain requires "
                "at least 2 entities"
            )

        if spec.target_updates < 1:
            raise ValueError(
                "interleaved_chain requires "
                "at least 1 relevant update"
            )

        if spec.distractor_updates < 1:
            raise ValueError(
                "interleaved_chain requires "
                "at least 1 irrelevant update"
            )

        relevant_positions = [
            i
            for i, op in enumerate(ops)
            if (
                isinstance(op, Move)
                and op.obj_id == target_obj
            )
        ]

        irrelevant_positions = [
            i
            for i, op in enumerate(ops)
            if (
                isinstance(op, Move)
                and op.obj_id != target_obj
            )
        ]

        if not relevant_positions:
            raise ValueError(
                "interleaved_chain has no "
                "relevant moves"
            )

        if not irrelevant_positions:
            raise ValueError(
                "interleaved_chain has no "
                "irrelevant moves"
            )

        # ----------------------------------------------------
        # Actual interleaving
        #
        # Reject:
        #
        # target target target distractor distractor
        #
        # Accept:
        #
        # target distractor target distractor
        # ----------------------------------------------------

        genuinely_interleaved = False

        for i in range(
            len(relevant_positions) - 1
        ):
            left = relevant_positions[i]
            right = relevant_positions[i + 1]

            if any(
                left < d < right
                for d in irrelevant_positions
            ):
                genuinely_interleaved = True
                break

        if not genuinely_interleaved:
            raise ValueError(
                "interleaved_chain does not contain "
                "actual interleaving"
            )

        # ----------------------------------------------------
        # Minimum interleaving constraint
        # ----------------------------------------------------

        between_count = 0

        for i in range(
            len(relevant_positions) - 1
        ):
            left = relevant_positions[i]
            right = relevant_positions[i + 1]

            between_count += sum(
                left < d < right
                for d in irrelevant_positions
            )

        possible = len(update_ops)

        interleaving_score = (
            between_count / possible
            if possible
            else 0.0
        )

        if (
            interleaving_score
            < spec.min_interleaving
        ):
            raise ValueError(
                "interleaving score too low: "
                f"required >= {spec.min_interleaving}, "
                f"got {interleaving_score}"
            )

        return

    # ========================================================
    # 6. REVISION
    # ========================================================

    if spec.family == "revision":

        if spec.entity_count != 1:
            raise ValueError(
                "revision requires entity_count=1"
            )

        if spec.distractor_updates != 0:
            raise ValueError(
                "revision does not currently support "
                "irrelevant updates"
            )

        if spec.target_updates < 3:
            raise ValueError(
                "revision requires at least "
                "3 relevant updates"
            )

        destinations = [
            op.dst
            for op in relevant_ops
        ]

        # ----------------------------------------------------
        # Unique-location requirement
        # ----------------------------------------------------

        unique_locations = len(
            set(destinations)
        )

        if (
            unique_locations
            < spec.min_unique_target_locations
        ):
            raise ValueError(
                "revision has too few unique "
                "target locations: "
                f"required >= "
                f"{spec.min_unique_target_locations}, "
                f"got {unique_locations}"
            )

        # ----------------------------------------------------
        # Actual revision
        #
        # A location must occur again after at least one
        # different location.
        # ----------------------------------------------------

        revised = False

        for i in range(
            len(destinations)
        ):
            for j in range(
                i + 2,
                len(destinations),
            ):
                if (
                    destinations[i]
                    == destinations[j]
                    and len(
                        set(
                            destinations[
                                i + 1:j
                            ]
                        )
                    )
                    > 0
                ):
                    revised = True
                    break

            if revised:
                break

        if not revised:
            raise ValueError(
                "revision trajectory lacks a genuine "
                "revisit after an intervening location"
            )

        return

    # ========================================================
    # 7. SPLIT CHAIN
    # ========================================================

    if spec.family == "split_chain":

        if spec.entity_count != 2:
            raise ValueError(
                "split_chain requires entity_count=2"
            )

        if spec.target_updates < 2:
            raise ValueError(
                "split_chain requires at least "
                "2 target_updates"
            )

        split_ops = [
            op for op in ops if isinstance(op, Split)
        ]

        if not split_ops:
            raise ValueError(
                "split_chain contains no Split operations"
            )

        # The source of every Split must be the target object.
        for sop in split_ops:
            if sop.source_obj_id != target_obj:
                raise ValueError(
                    "split_chain Split must originate "
                    "from the target object; "
                    f"got source={sop.source_obj_id!r}"
                )

        # Target must have at least one Move before and after
        # the first Split.
        first_split_idx = next(
            i for i, op in enumerate(ops)
            if isinstance(op, Split)
        )

        pre_split_target_moves = [
            op for op in ops[:first_split_idx]
            if isinstance(op, Move) and op.obj_id == target_obj
        ]

        if not pre_split_target_moves:
            raise ValueError(
                "split_chain must have at least one "
                "target Move before the Split"
            )

        if spec.target_updates > 2:
            post_split_target_moves = [
                op for op in ops[first_split_idx + 1:]
                if isinstance(op, Move) and op.obj_id == target_obj
            ]

            if not post_split_target_moves:
                raise ValueError(
                    "split_chain must have at least one "
                    "target Move after the Split when target_updates > 2"
                )

        return

    # ========================================================
    # 8. MERGE CHAIN
    # ========================================================

    if spec.family == "merge_chain":

        if spec.entity_count < 2:
            raise ValueError(
                "merge_chain requires entity_count >= 2"
            )

        if spec.target_updates < 2:
            raise ValueError(
                "merge_chain requires at least "
                "2 target_updates"
            )

        merge_ops = [
            op for op in ops if isinstance(op, Merge)
        ]

        if not merge_ops:
            raise ValueError(
                "merge_chain contains no Merge operations"
            )

        # The target must appear in a container that is either
        # src or dst of a Merge op (this is checked via replay
        # in build_trajectory; here we just verify count).
        if len(merge_ops) < 1:
            raise ValueError(
                "merge_chain requires at least one Merge"
            )

        # Verify entity_count matches Put + Split ops (no Split
        # in merge_chain, so just Puts).
        put_count = len(
            [op for op in ops if isinstance(op, Put)]
        )

        if put_count != spec.entity_count:
            raise ValueError(
                "merge_chain entity_count mismatch: "
                f"expected {spec.entity_count} Put ops, "
                f"got {put_count}"
            )

        return

    # ========================================================
    # 9. SWAP CHAIN
    # ========================================================

    if spec.family == "swap_chain":

        if spec.entity_count < 2:
            raise ValueError(
                "swap_chain requires entity_count >= 2"
            )

        if spec.target_updates < 1:
            raise ValueError(
                "swap_chain requires at least "
                "1 target_update"
            )

        swap_ops = [
            op for op in ops if isinstance(op, Swap)
        ]

        if not swap_ops:
            raise ValueError(
                "swap_chain contains no Swap operations"
            )

        # The target object must start in a container that
        # participates in at least one Swap.  Because the
        # canonical replay already verified state consistency,
        # we verify only that a Swap exists.
        return

    # ========================================================
    # 10. UNDO CHAIN
    # ========================================================

    if spec.family == "undo_chain":

        if spec.entity_count != 1:
            raise ValueError(
                "undo_chain requires entity_count=1"
            )

        if spec.target_updates < 2:
            raise ValueError(
                "undo_chain requires at least "
                "2 target_updates (move + undo)"
            )

        undo_ops = [
            op for op in ops if isinstance(op, Undo)
        ]

        if not undo_ops:
            raise ValueError(
                "undo_chain contains no Undo operations"
            )

        redo_ops = [
            op for op in ops if isinstance(op, Redo)
        ]

        if redo_ops:
            raise ValueError(
                "undo_chain must not contain Redo operations"
            )

        # There must be at least one Move before the first Undo.
        first_undo_idx = next(
            i for i, op in enumerate(ops)
            if isinstance(op, Undo)
        )

        pre_undo_moves = [
            op for op in ops[:first_undo_idx]
            if isinstance(op, Move)
        ]

        if not pre_undo_moves:
            raise ValueError(
                "undo_chain must have at least one Move "
                "before the first Undo"
            )

        return

    # ========================================================
    # 11. UNDO_REDO CHAIN
    # ========================================================

    if spec.family == "undo_redo_chain":

        if spec.entity_count != 1:
            raise ValueError(
                "undo_redo_chain requires entity_count=1"
            )

        if spec.target_updates < 3:
            raise ValueError(
                "undo_redo_chain requires at least "
                "3 target_updates (move + undo + redo)"
            )

        undo_ops = [
            op for op in ops if isinstance(op, Undo)
        ]

        redo_ops = [
            op for op in ops if isinstance(op, Redo)
        ]

        if not undo_ops:
            raise ValueError(
                "undo_redo_chain contains no Undo operations"
            )

        if not redo_ops:
            raise ValueError(
                "undo_redo_chain contains no Redo operations"
            )

        # Undo must precede at least one Redo.
        first_undo_idx = next(
            i for i, op in enumerate(ops)
            if isinstance(op, Undo)
        )

        first_redo_idx = next(
            i for i, op in enumerate(ops)
            if isinstance(op, Redo)
        )

        if first_redo_idx <= first_undo_idx:
            raise ValueError(
                "undo_redo_chain: Redo must follow Undo"
            )

        # The Move that was undone must precede the Undo.
        pre_undo_moves = [
            op for op in ops[:first_undo_idx]
            if isinstance(op, Move)
        ]

        if not pre_undo_moves:
            raise ValueError(
                "undo_redo_chain must have at least one "
                "Move before the first Undo"
            )

        return

    raise ValueError(
        "no structural validator exists for "
        f"trajectory family={spec.family!r}"
    )
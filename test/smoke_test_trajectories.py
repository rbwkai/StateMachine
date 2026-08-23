from __future__ import annotations

from pathlib import Path
import random
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.trajectories import (
    available_families,
    build_trajectory,
)
from generator.trajectory_specs import TrajectorySpec
from generator.trajectory_validation import validate_trajectory


def print_trajectory(result):
    for i, op in enumerate(result.ops):
        print(f"  {i}: {op}")

    print("target:", result.target_obj)
    print("final:", result.final_state.location)


# ============================================================
# 1. AVAILABLE FAMILIES
# ============================================================

print("=" * 70)
print("1. AVAILABLE FAMILIES")
print("=" * 70)

families = available_families()

expected_families = [
    "basic_chain",
    "interleaved_chain",
    "merge_chain",
    "revision",
    "split_chain",
    "swap_chain",
    "undo_chain",
    "undo_redo_chain",
]

assert families == expected_families, f"Expected {expected_families}, got {families}"

print(families)
print("PASS")


# ============================================================
# 2. BASIC CHAIN
# ============================================================

print()
print("=" * 70)
print("2. BASIC CHAIN")
print("=" * 70)

basic_spec = TrajectorySpec(
    family="basic_chain",
    entity_count=1,
    num_containers=3,
    total_updates=6,
    target_updates=6,
    distractor_updates=0,
)

basic = build_trajectory(
    random.Random(42),
    basic_spec,
)

print_trajectory(basic)

assert len(basic.ops) == 1 + basic_spec.total_updates
assert basic.target_obj == "o0"
assert len(basic.final_state.location) == 1

validate_trajectory(
    basic.ops,
    basic.target_obj,
    basic.spec,
)

print("PASS")


# ============================================================
# 3. INTERLEAVED CHAIN
# ============================================================

print()
print("=" * 70)
print("3. INTERLEAVED CHAIN")
print("=" * 70)

interleaved_spec = TrajectorySpec(
    family="interleaved_chain",
    entity_count=4,
    num_containers=3,
    total_updates=8,
    target_updates=4,
    distractor_updates=4,
)

interleaved = build_trajectory(
    random.Random(42),
    interleaved_spec,
)

print_trajectory(interleaved)

assert len(interleaved.ops) == 4 + 8
assert interleaved.target_obj == "o0"
assert len(interleaved.final_state.location) == 4

target_moves = [
    op
    for op in interleaved.ops
    if getattr(op, "obj_id", None) == "o0"
    and op.__class__.__name__ == "Move"
]

distractor_moves = [
    op
    for op in interleaved.ops
    if getattr(op, "obj_id", None) != "o0"
    and op.__class__.__name__ == "Move"
]

assert len(target_moves) == 4
assert len(distractor_moves) == 4

validate_trajectory(
    interleaved.ops,
    interleaved.target_obj,
    interleaved.spec,
)

print("PASS")


# ============================================================
# 4. REVISION
# ============================================================

print()
print("=" * 70)
print("4. REVISION")
print("=" * 70)

revision_spec = TrajectorySpec(
    family="revision",
    entity_count=1,
    num_containers=3,
    total_updates=6,
    target_updates=6,
    distractor_updates=0,
)

revision = build_trajectory(
    random.Random(42),
    revision_spec,
)

print_trajectory(revision)

assert len(revision.ops) == 1 + revision_spec.total_updates
assert revision.target_obj == "o0"

destinations = [
    op.dst
    for op in revision.ops
    if op.__class__.__name__ == "Move"
]

print("destinations:", destinations)

assert len(set(destinations)) < len(destinations)

revised = any(
    destinations[i] == destinations[j]
    and j > i + 1
    for i in range(len(destinations))
    for j in range(i + 1, len(destinations))
)

assert revised

validate_trajectory(
    revision.ops,
    revision.target_obj,
    revision.spec,
)

print("PASS")


# ============================================================
# 5. SPLIT CHAIN
# ============================================================

print()
print("=" * 70)
print("5. SPLIT CHAIN")
print("=" * 70)

split_spec = TrajectorySpec(
    family="split_chain",
    entity_count=2,
    num_containers=3,
    total_updates=4,
    target_updates=4,
    distractor_updates=0,
)

split = build_trajectory(
    random.Random(42),
    split_spec,
)

print_trajectory(split)

assert split.target_obj == "o0"
assert len(split.final_state.location) == 2
assert "o0" in split.final_state.location
assert "o1" in split.final_state.location

validate_trajectory(
    split.ops,
    split.target_obj,
    split.spec,
)

print("PASS")


# ============================================================
# 6. MERGE CHAIN
# ============================================================

print()
print("=" * 70)
print("6. MERGE CHAIN")
print("=" * 70)

merge_spec = TrajectorySpec(
    family="merge_chain",
    entity_count=2,
    num_containers=3,
    total_updates=3,
    target_updates=3,
    distractor_updates=0,
)

merge = build_trajectory(
    random.Random(42),
    merge_spec,
)

print_trajectory(merge)

assert merge.target_obj == "o0"
assert len(merge.final_state.location) == 2

validate_trajectory(
    merge.ops,
    merge.target_obj,
    merge.spec,
)

print("PASS")


# ============================================================
# 7. SWAP CHAIN
# ============================================================

print()
print("=" * 70)
print("7. SWAP CHAIN")
print("=" * 70)

swap_spec = TrajectorySpec(
    family="swap_chain",
    entity_count=2,
    num_containers=3,
    total_updates=3,
    target_updates=3,
    distractor_updates=0,
)

swap = build_trajectory(
    random.Random(42),
    swap_spec,
)

print_trajectory(swap)

assert swap.target_obj == "o0"
assert len(swap.final_state.location) == 2

validate_trajectory(
    swap.ops,
    swap.target_obj,
    swap.spec,
)

print("PASS")


# ============================================================
# 8. UNDO CHAIN
# ============================================================

print()
print("=" * 70)
print("8. UNDO CHAIN")
print("=" * 70)

undo_spec = TrajectorySpec(
    family="undo_chain",
    entity_count=1,
    num_containers=3,
    total_updates=4,
    target_updates=4,
    distractor_updates=0,
)

undo = build_trajectory(
    random.Random(42),
    undo_spec,
)

print_trajectory(undo)

assert undo.target_obj == "o0"
assert len(undo.final_state.location) == 1

validate_trajectory(
    undo.ops,
    undo.target_obj,
    undo.spec,
)

print("PASS")


# ============================================================
# 9. UNDO REDO CHAIN
# ============================================================

print()
print("=" * 70)
print("9. UNDO REDO CHAIN")
print("=" * 70)

undo_redo_spec = TrajectorySpec(
    family="undo_redo_chain",
    entity_count=1,
    num_containers=3,
    total_updates=5,
    target_updates=5,
    distractor_updates=0,
)

undo_redo = build_trajectory(
    random.Random(42),
    undo_redo_spec,
)

print_trajectory(undo_redo)

assert undo_redo.target_obj == "o0"
assert len(undo_redo.final_state.location) == 1

validate_trajectory(
    undo_redo.ops,
    undo_redo.target_obj,
    undo_redo.spec,
)

print("PASS")


# ============================================================
# 10. VALIDATION GATE
# ============================================================

print()
print("=" * 70)
print("10. VALIDATION GATE")
print("=" * 70)

# Deliberately construct an invalid specification.
invalid_spec = TrajectorySpec(
    family="basic_chain",
    entity_count=1,
    num_containers=3,
    total_updates=7,
    target_updates=6,
    distractor_updates=1,
)

try:
    validate_trajectory(
        basic.ops,
        basic.target_obj,
        invalid_spec,
    )
except ValueError as exc:
    print("Correctly rejected invalid basic_chain configuration:")
    print(f"  {exc}")
else:
    raise AssertionError(
        "invalid basic_chain configuration was not rejected"
    )

# Also test structural validation rejection: split_chain with entity_count mismatch
invalid_split_spec = TrajectorySpec(
    family="split_chain",
    entity_count=1,
    num_containers=3,
    total_updates=4,
    target_updates=4,
    distractor_updates=0,
)

try:
    validate_trajectory(
        split.ops,
        split.target_obj,
        invalid_split_spec,
    )
except ValueError as exc:
    print("Correctly rejected invalid split_chain entity count:")
    print(f"  {exc}")
else:
    raise AssertionError(
        "invalid split_chain configuration was not rejected"
    )

print("PASS")


# ============================================================
# 11. REPLAY CONSISTENCY & DETERMINISM
# ============================================================

print()
print("=" * 70)
print("11. REPLAY CONSISTENCY & DETERMINISM")
print("=" * 70)

all_results = [
    basic,
    interleaved,
    revision,
    split,
    merge,
    swap,
    undo,
    undo_redo,
]

for result in all_results:
    rebuilt = build_trajectory(
        random.Random(42),
        result.spec,
    )

    assert rebuilt.ops == result.ops, f"Ops mismatch for {result.spec.family}"
    assert (
        rebuilt.final_state.location
        == result.final_state.location
    ), f"Location mismatch for {result.spec.family}"
    assert (
        rebuilt.final_state.object_type
        == result.final_state.object_type
    ), f"Object type mismatch for {result.spec.family}"

print("Same seed -> identical trajectories across all 8 families: PASS")


# ============================================================
# 12. MULTI-SEED VALIDATION
# ============================================================

print()
print("=" * 70)
print("12. MULTI-SEED VALIDATION (10 seeds x 8 families)")
print("=" * 70)

all_specs = [
    basic_spec,
    interleaved_spec,
    revision_spec,
    split_spec,
    merge_spec,
    swap_spec,
    undo_spec,
    undo_redo_spec,
]

for seed in range(10):
    for spec in all_specs:
        result = build_trajectory(
            random.Random(seed),
            spec,
        )

        validate_trajectory(
            result.ops,
            result.target_obj,
            result.spec,
        )

print("10 seeds × 8 families: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ALL 8 TRAJECTORY SMOKE TESTS PASSED")
print("=" * 70)

print("Family registry (8 families) PASS")
print("Basic chain ................ PASS")
print("Interleaved chain .......... PASS")
print("Revision ................... PASS")
print("Split chain ................ PASS")
print("Merge chain ................ PASS")
print("Swap chain ................. PASS")
print("Undo chain ................. PASS")
print("Undo/Redo chain ............ PASS")
print("Spec validation gate ....... PASS")
print("Replay consistency ......... PASS")
print("Determinism ................ PASS")
print("Multi-seed validation ...... PASS")

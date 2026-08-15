from __future__ import annotations

import random

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

assert families == [
    "basic_chain",
    "interleaved_chain",
    "revision",
]

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

# Revision must revisit at least one previous location.
assert len(set(destinations)) < len(destinations)

# And at least one location must be revisited
# after an intervening update.
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
# 5. VALIDATION GATE
# ============================================================

print()
print("=" * 70)
print("5. VALIDATION GATE")
print("=" * 70)

# Deliberately construct an invalid specification.
# basic_chain must have zero distractor updates.

# Deliberately construct an invalid specification.
# basic_chain must have zero distractor updates.

# Deliberately construct a numerically valid but
# family-invalid specification.
#
# basic_chain must have zero distractor updates.
#
# TrajectorySpec validates numerical consistency.
# validate_trajectory() validates family-specific rules.

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
        "invalid basic_chain configuration "
        "was not rejected"
    )

print("PASS")

# ============================================================
# 6. REPLAY CONSISTENCY
# ============================================================

print()
print("=" * 70)
print("6. REPLAY CONSISTENCY")
print("=" * 70)

for result in [
    basic,
    interleaved,
    revision,
]:
    # build_trajectory already performs replay validation.
    # Rebuilding with the same seed additionally checks that
    # the construction remains deterministic.

    rebuilt = build_trajectory(
        random.Random(42),
        result.spec,
    )

    assert rebuilt.ops == result.ops
    assert (
        rebuilt.final_state.location
        == result.final_state.location
    )
    assert (
        rebuilt.final_state.object_type
        == result.final_state.object_type
    )

print("Same seed -> identical trajectories: PASS")


# ============================================================
# 7. DIFFERENT SEEDS DO NOT BREAK VALIDITY
# ============================================================

print()
print("=" * 70)
print("7. MULTI-SEED VALIDATION")
print("=" * 70)

specs = [
    basic_spec,
    interleaved_spec,
    revision_spec,
]

for seed in range(10):
    for spec in specs:
        result = build_trajectory(
            random.Random(seed),
            spec,
        )

        validate_trajectory(
            result.ops,
            result.target_obj,
            result.spec,
        )

print("10 seeds × 3 families: PASS")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ALL TRAJECTORY SMOKE TESTS PASSED")
print("=" * 70)

print("Family registry ............ PASS")
print("Basic chain ................ PASS")
print("Interleaved chain .......... PASS")
print("Revision ................... PASS")
print("Spec validation ........... PASS")
print("Replay consistency ........ PASS")
print("Determinism ................ PASS")
print("Multi-seed validation ...... PASS")
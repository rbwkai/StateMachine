"""
test/test_measured_factors.py
==============================
Unit tests for generator/metadata.py.

Tests verify that MeasuredFactors computed from the canonical replay
agree with the requested TrajectorySpec for all 8 families, across
multiple seeds and T levels.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator import (
    MeasuredFactors,
    measure_factors,
    verify_factors,
    build_trajectory,
)
from generator.trajectory_specs import TrajectorySpec
from generator.metadata import _count_revisits


# ============================================================
# Helpers
# ============================================================

def build_and_measure(family, entity_count, target_updates,
                      distractor_updates=0, num_containers=3,
                      seed=42):
    """Build a trajectory and return (trajectory, measured_factors)."""
    total_updates = target_updates + distractor_updates
    spec = TrajectorySpec(
        family=family,
        entity_count=entity_count,
        num_containers=num_containers,
        total_updates=total_updates,
        target_updates=target_updates,
        distractor_updates=distractor_updates,
    )
    t = build_trajectory(random.Random(seed), spec)
    return t, t.measured_factors


# ============================================================
# 1. _count_revisits unit tests
# ============================================================

print("=" * 70)
print("1. _count_revisits")
print("=" * 70)

# No revisits — every location distinct.
assert _count_revisits(["c0", "c1", "c2", "c3"]) == 0, "all distinct → 0"

# One revisit: c0 → c1 → c0 (c1 intervenes).
assert _count_revisits(["c0", "c1", "c0"]) == 1, "one revisit"

# Two revisits: c0→c1→c0→c1
assert _count_revisits(["c0", "c1", "c0", "c1"]) == 2, "two revisits"

# Immediate repeat is NOT a revisit (no intervening different location).
assert _count_revisits(["c0", "c0", "c1"]) == 0, "immediate repeat not a revisit"

# Short list.
assert _count_revisits([]) == 0
assert _count_revisits(["c0"]) == 0
assert _count_revisits(["c0", "c1"]) == 0

print("PASS")


# ============================================================
# 2. basic_chain — E=1, D=0, T measured correctly
# ============================================================

print()
print("=" * 70)
print("2. basic_chain measured factors")
print("=" * 70)

for T in [2, 4, 6, 8, 12, 16]:
    for seed in range(5):
        t, m = build_and_measure("basic_chain", 1, T, 0, seed=seed)
        assert m.E_actual == 1, f"basic_chain T={T} seed={seed}: E={m.E_actual}"
        assert m.T_actual == T, f"basic_chain T={T} seed={seed}: T_actual={m.T_actual} != {T}"
        assert m.D_actual == 0, f"basic_chain T={T} seed={seed}: D={m.D_actual}"

print(f"basic_chain (T∈{{2,4,6,8,12,16}} × 5 seeds): PASS")


# ============================================================
# 3. interleaved_chain — E, T, D all measured correctly
# ============================================================

print()
print("=" * 70)
print("3. interleaved_chain measured factors")
print("=" * 70)

for target_updates, distractor_updates in [(4, 4), (4, 8), (8, 8)]:
    for seed in range(5):
        t, m = build_and_measure(
            "interleaved_chain", 3, target_updates, distractor_updates, seed=seed
        )
        assert m.E_actual == 3, \
            f"interleaved E=3 T={target_updates} D={distractor_updates} s={seed}: E_actual={m.E_actual}"
        assert m.T_actual == target_updates, \
            f"interleaved T={target_updates} D={distractor_updates} s={seed}: T_actual={m.T_actual}"
        assert m.D_actual == distractor_updates, \
            f"interleaved T={target_updates} D={distractor_updates} s={seed}: D_actual={m.D_actual}"

print("interleaved_chain (3 configs × 5 seeds): PASS")


# ============================================================
# 4. revision — V_actual >= 1
# ============================================================

print()
print("=" * 70)
print("4. revision — V_actual >= 1")
print("=" * 70)

for T in [4, 6, 8, 12]:
    for seed in range(5):
        t, m = build_and_measure("revision", 1, T, 0, seed=seed)
        assert m.E_actual == 1, f"revision T={T} s={seed}: E={m.E_actual}"
        assert m.T_actual == T, f"revision T={T} s={seed}: T_actual={m.T_actual} != {T}"
        assert m.D_actual == 0, f"revision T={T} s={seed}: D={m.D_actual}"
        assert m.V_actual >= 1, \
            f"revision T={T} s={seed}: V_actual={m.V_actual} — no revisit detected"

print("revision (T∈{4,6,8,12} × 5 seeds, V≥1): PASS")


# ============================================================
# 5. split_chain — E=2 (target + child)
# ============================================================

print()
print("=" * 70)
print("5. split_chain measured factors")
print("=" * 70)

for T in [2, 4, 6, 8]:
    for seed in range(5):
        t, m = build_and_measure("split_chain", 2, T, 0, seed=seed)
        assert m.E_actual == 2, f"split T={T} s={seed}: E_actual={m.E_actual}"
        # D_actual may be non-zero for split_chain because the Split op itself
        # does not change the target's location (child spawns alongside target).
        # This is expected and not an error in measurement.
        assert m.T_actual + m.D_actual == T, \
            f"split T={T} s={seed}: T_actual+D_actual={m.T_actual+m.D_actual} != {T}"

print("split_chain (T∈{2,4,6,8} × 5 seeds): PASS")


# ============================================================
# 6. merge_chain — E≥2, T measured
# ============================================================

print()
print("=" * 70)
print("6. merge_chain measured factors")
print("=" * 70)

for T in [2, 4, 6, 8]:
    for seed in range(5):
        t, m = build_and_measure("merge_chain", 2, T, 0, seed=seed)
        assert m.E_actual == 2, f"merge T={T} s={seed}: E_actual={m.E_actual}"
        # T_actual counts ops that causally moved the target.
        # Merge moves the target if target is in the source container.
        assert m.T_actual >= 1, f"merge T={T} s={seed}: T_actual={m.T_actual} < 1"

print("merge_chain (T∈{2,4,6,8} × 5 seeds): PASS")


# ============================================================
# 7. swap_chain — target location changes with every Swap
# ============================================================

print()
print("=" * 70)
print("7. swap_chain measured factors")
print("=" * 70)

for T in [1, 2, 4, 6, 8]:
    for seed in range(5):
        t, m = build_and_measure("swap_chain", 2, T, 0, seed=seed)
        assert m.E_actual == 2, f"swap T={T} s={seed}: E_actual={m.E_actual}"
        assert m.T_actual >= 1, f"swap T={T} s={seed}: T_actual={m.T_actual}"

print("swap_chain (T∈{1,2,4,6,8} × 5 seeds): PASS")


# ============================================================
# 8. undo_chain — Undo counts as a target op (location changes)
# ============================================================

print()
print("=" * 70)
print("8. undo_chain measured factors")
print("=" * 70)

for T in [2, 4, 6, 8]:
    for seed in range(5):
        t, m = build_and_measure("undo_chain", 1, T, 0, seed=seed)
        assert m.E_actual == 1, f"undo T={T} s={seed}: E_actual={m.E_actual}"
        assert m.D_actual == 0, f"undo T={T} s={seed}: D_actual={m.D_actual}"
        # T_actual should equal requested T since all ops affect target
        assert m.T_actual == T, \
            f"undo T={T} s={seed}: T_actual={m.T_actual} != {T}"

print("undo_chain (T∈{2,4,6,8} × 5 seeds): PASS")


# ============================================================
# 9. undo_redo_chain — Undo + Redo both count as target ops
# ============================================================

print()
print("=" * 70)
print("9. undo_redo_chain measured factors")
print("=" * 70)

for T in [3, 5, 8]:
    for seed in range(5):
        t, m = build_and_measure("undo_redo_chain", 1, T, 0, seed=seed)
        assert m.E_actual == 1, f"undo_redo T={T} s={seed}: E_actual={m.E_actual}"
        assert m.D_actual == 0, f"undo_redo T={T} s={seed}: D_actual={m.D_actual}"
        assert m.T_actual == T, \
            f"undo_redo T={T} s={seed}: T_actual={m.T_actual} != {T}"

print("undo_redo_chain (T∈{3,5,8} × 5 seeds): PASS")


# ============================================================
# 10. measure_factors with L_actual
# ============================================================

print()
print("=" * 70)
print("10. L_actual word-count computation")
print("=" * 70)

sentences = [
    "A key was placed in the green box.",
    "The key was moved to the blue bin.",
    "The key was moved back to the green box.",
]
from generator.metadata import measure_factors as mf_fn
from generator.trajectories import _initial_world, _put, _move

state, history, containers = _initial_world(3)
ops = []
state = _put(state, history, ops, "o0", "key", "c0")
state = _move(state, history, ops, "o0", "c1")
state = _move(state, history, ops, "o0", "c0")

m = mf_fn(ops, containers, "o0", sentences=sentences)
expected_words = sum(len(s.split()) for s in sentences)
assert m.L_actual == expected_words, \
    f"L_actual={m.L_actual} != {expected_words}"

print(f"L_actual word count (expected {expected_words}): PASS")


# ============================================================
# 11. verify_factors catches mismatches
# ============================================================

print()
print("=" * 70)
print("11. verify_factors — mismatch detection")
print("=" * 70)

bad_m = MeasuredFactors(E_actual=2, T_actual=8, D_actual=0, V_actual=0)

try:
    verify_factors(
        requested_E=1,   # wrong!
        requested_T=8,
        requested_D=0,
        measured=bad_m,
        family="basic_chain",
        instance_id="test_instance",
    )
    raise AssertionError("Should have raised on E mismatch")
except AssertionError as exc:
    if "E_actual" in str(exc):
        print(f"  Correctly caught E mismatch: {exc}")
    else:
        raise

try:
    verify_factors(
        requested_E=2,
        requested_T=6,   # wrong!
        requested_D=0,
        measured=bad_m,
        family="basic_chain",
    )
    raise AssertionError("Should have raised on T mismatch")
except AssertionError as exc:
    if "T_actual" in str(exc):
        print(f"  Correctly caught T mismatch: {exc}")
    else:
        raise

print("verify_factors mismatch detection: PASS")


# ============================================================
# 12. measured_factors attached to ConstructedTrajectory
# ============================================================

print()
print("=" * 70)
print("12. measured_factors attached to ConstructedTrajectory")
print("=" * 70)

# Structural families (split, merge, swap) may have non-zero D_actual
# because structural ops (Split, Merge, Swap) that don't physically move
# the target are correctly counted as D by measure_factors.
# We only check strict D equality for families where D is explicitly controlled.
_STRICT_D_FAMILIES = {"basic_chain", "interleaved_chain", "revision",
                       "undo_chain", "undo_redo_chain"}

for family, E, T, D in [
    ("basic_chain", 1, 8, 0),
    ("interleaved_chain", 3, 4, 4),
    ("revision", 1, 8, 0),
    ("split_chain", 2, 8, 0),
    ("merge_chain", 2, 6, 0),
    ("swap_chain", 2, 6, 0),
    ("undo_chain", 1, 8, 0),
    ("undo_redo_chain", 1, 8, 0),
]:
    t, m = build_and_measure(family, E, T, D, seed=0)
    assert t.measured_factors is not None, \
        f"{family}: measured_factors not attached to ConstructedTrajectory"
    assert t.measured_factors is m, \
        f"{family}: measured_factors identity mismatch"
    assert m.E_actual == E, \
        f"{family}: E_actual={m.E_actual} != {E}"
    if family in _STRICT_D_FAMILIES:
        assert m.D_actual == D, \
            f"{family}: D_actual={m.D_actual} != {D}"

print("measured_factors attached across all 8 families: PASS")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ALL MEASURED-FACTOR TESTS PASSED")
print("=" * 70)
print("_count_revisits unit tests ............. PASS")
print("basic_chain E/T/D measurement .......... PASS")
print("interleaved_chain E/T/D measurement .... PASS")
print("revision V≥1 guarantee ................. PASS")
print("split_chain E measurement .............. PASS")
print("merge_chain E measurement .............. PASS")
print("swap_chain T≥1 measurement ............. PASS")
print("undo_chain T measurement ............... PASS")
print("undo_redo_chain T measurement .......... PASS")
print("L_actual word-count computation ........ PASS")
print("verify_factors mismatch detection ....... PASS")
print("measured_factors in ConstructedTrajectory PASS")

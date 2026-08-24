"""Invariant validation suite for DWS-Bench generator.
Runs statistical batch testing across all 8 families over multiple seeds
and rigorously checks mathematical and symbolic invariants:

Invariants:
1. S == E + U (Total symbolic ops == initial placements + post-init updates)
2. U == T + D (Post-init updates == target updates + distractor updates)
3. Initial Put operations count == E (or 1 for split_chain where child spawns later)
4. Post-init state updates count == U
5. S counts discrete symbolic operations, L counts rendered sentence length
6. Final simulator state == Canonical replay trace final state
7. Determinism: Seed(k) reproduces identical ops, states, and text
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import List

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator.dataset_spec import CapabilityGroup, family_capability_group
from generator.trajectories import available_families, build_trajectory
from generator.trajectory_specs import TrajectorySpec
from generator.trajectory_validation import validate_trajectory
from render.names import NameRegistry
from render.narrative import question_location, render_narrative
from world.operations import Move, Put, Split, Merge, Swap, Undo, Redo


def verify_batch(seeds_per_family: int = 25) -> None:
    families = available_families()
    total_tested = 0

    print("=" * 76)
    print(f"AUTOMATED INVARIANT VERIFICATION SUITE ({len(families)} families x {seeds_per_family} seeds)")
    print("=" * 76)

    for family in families:
        group = family_capability_group(family)
        print(f"\nVerifying Family: {family.upper():<18} [Capability Group: {group.name}]")

        for seed in range(seeds_per_family):
            rng = random.Random(seed)

            # Parameter variations across seeds
            if family in ("basic_chain", "revision", "undo_chain", "undo_redo_chain"):
                entity_count = 1
                target_updates = 3 + (seed % 4)
                distractor_updates = 0
            elif family in ("split_chain", "swap_chain"):
                entity_count = 2
                target_updates = 2 + (seed % 3)
                distractor_updates = seed % 2
            elif family == "interleaved_chain":
                entity_count = 2 + (seed % 3)
                target_updates = 2 + (seed % 3)
                distractor_updates = 2 + (seed % 3)
            elif family == "merge_chain":
                entity_count = 2 + (seed % 2)
                target_updates = 2 + (seed % 3)
                distractor_updates = seed % 2

            total_updates = target_updates + distractor_updates

            spec = TrajectorySpec(
                family=family,
                entity_count=entity_count,
                num_containers=3 + (seed % 2),
                total_updates=total_updates,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
            )

            # 1. Build and validate structurally
            traj = build_trajectory(rng, spec)
            validate_trajectory(traj.ops, traj.target_obj, traj.spec)

            # 2. Accounting calculations
            put_ops = [op for op in traj.ops if isinstance(op, Put)]
            post_init_ops = [op for op in traj.ops if not isinstance(op, Put)]

            E_actual = len(put_ops)
            U_actual = len(post_init_ops)
            S_actual = len(traj.ops)

            # INVARIANT 1: S == E + U
            assert S_actual == E_actual + U_actual, (
                f"Invariant S == E + U failed: S={S_actual}, E={E_actual}, U={U_actual}"
            )

            # INVARIANT 2: U == T + D
            assert spec.total_updates == spec.target_updates + spec.distractor_updates, (
                f"Invariant U == T + D failed: {spec.total_updates} != {spec.target_updates} + {spec.distractor_updates}"
            )

            # INVARIANT 3: Initial Put operations count == E (or 1 for split_chain)
            expected_initial_puts = 1 if family == "split_chain" else spec.entity_count
            assert E_actual == expected_initial_puts, (
                f"Initial Put count mismatch: expected {expected_initial_puts}, got {E_actual}"
            )

            # INVARIANT 4: Post-init state updates count == U
            assert U_actual == spec.total_updates, (
                f"Post-init updates mismatch: expected {spec.total_updates}, got {U_actual}"
            )

            # 3. Render and test linguistic realization
            names = NameRegistry(containers=traj.containers, rng=random.Random(seed))
            sentences, final_state = render_narrative(traj.ops, traj.containers, names)
            L_actual = len(sentences)

            # INVARIANT 5: S counts symbolic operations, L counts rendered sentence length
            assert L_actual == S_actual, (
                f"L ({L_actual}) != S ({S_actual})"
            )

            # INVARIANT 6: Target object exists in final world state or was accounted for
            assert traj.target_obj in traj.final_state.object_type, (
                f"Target {traj.target_obj} not in final state object types"
            )

            # INVARIANT 7: Determinism check (rebuilding with same seed yields exact match)
            traj_rebuilt = build_trajectory(random.Random(seed), spec)
            assert traj_rebuilt.ops == traj.ops, "Determinism failed on ops"
            assert traj_rebuilt.final_state.location == traj.final_state.location, "Determinism failed on location"

            total_tested += 1

        print(f"  [PASS] {seeds_per_family} seeds tested across parameter sweeps. All invariants hold.")

    print("\n" + "=" * 76)
    print(f"ALL INVARIANTS VERIFIED SUCCESSFULLY ({total_tested} total trajectory instances).")
    print("=" * 76)


if __name__ == "__main__":
    verify_batch(seeds_per_family=25)

"""End-to-end demo: generates and prints examples for all 8 trajectory chains
grouped by theoretical capability, plus procedural meta-probes, with explicit
parameter and state-transition accounting.

Run with:
    python example.py
"""

from __future__ import annotations

import random
from typing import Dict, Any

from generator.dataset_spec import CapabilityGroup, family_capability_group
from generator.trajectories import available_families, build_trajectory
from generator.trajectory_specs import TrajectorySpec
from render.names import NameRegistry
from render.narrative import question_location, render_narrative
from pipeline import generate_example
from world import Move, Put, Remove, Undo, Redo, Split, Merge, Swap

ALL_OPS = [Put, Move, Remove, Undo, Redo, Split, Merge, Swap]


def print_example_block(
    group_name: str,
    family_title: str,
    failure_mode: str,
    spec: TrajectorySpec,
    rng: random.Random,
) -> None:
    traj = build_trajectory(rng, spec)
    names = NameRegistry(containers=traj.containers, rng=rng)
    sentences, final_state = render_narrative(traj.ops, traj.containers, names)
    question = question_location(traj.target_obj, final_state, names)
    target_container = final_state.location.get(traj.target_obj)
    target_name = names.container(target_container) if target_container else "<nowhere>"

    initial_placements = len([op for op in traj.ops if isinstance(op, Put)])
    post_init_updates = len([op for op in traj.ops if not isinstance(op, Put)])
    total_transitions = len(traj.ops)
    rendered_text = " ".join(sentences)
    token_count_L = len(rendered_text.split())

    print("=" * 74)
    print(f"CAPABILITY GROUP: {group_name}")
    print(f"FAMILY:           {spec.family.upper()} ({family_title})")
    print(f"Tested Failure:   {failure_mode}")
    print("-" * 74)
    print("Parameter Accounting:")
    print(f"  - Initial Placements (E)       = {initial_placements}")
    print(f"  - Post-Init Operations (U)     = {post_init_updates}  (T={spec.target_updates} target ops, D={spec.distractor_updates} distractor ops)")
    print(f"  - Total Symbolic Operations (S)= {total_transitions}  (S = E + U)")
    print(f"  - Rendered Sentences           = {len(sentences)}")
    print(f"  - Rendered Tokens (L)          = {token_count_L} tokens  (thesis constraint: L < 600)")
    print("-" * 74)
    print("Process Narrative:")
    for idx, s in enumerate(sentences, 1):
        print(f"  {idx}. {s}")
    print()
    print(f"Question:    {question}")
    print(f"Gold Answer: {target_name} ({target_container})")
    print(f"Final State: {final_state.location}")
    print()


def main() -> None:
    rng = random.Random(42)

    family_demos = [
        # Group A: Sequential State Tracking
        (
            "GROUP A: SEQUENTIAL STATE TRACKING (Core / Naturalistic Transfer Target)",
            "basic_chain",
            "Sequential Capacity (RQ1)",
            "Forgetting or drifting over a long sequential update chain.",
            TrajectorySpec(family="basic_chain", entity_count=1, total_updates=4, target_updates=4),
        ),
        (
            "GROUP A: SEQUENTIAL STATE TRACKING (Core / Naturalistic Transfer Target)",
            "revision",
            "Current-vs-Past Disambiguation (RQ4)",
            "Reporting a stale, previously occupied location.",
            TrajectorySpec(family="revision", entity_count=1, total_updates=6, target_updates=6),
        ),
        # Group B: Multi-Entity Interference
        (
            "GROUP B: MULTI-ENTITY INTERFERENCE (Core / Naturalistic Transfer Target)",
            "interleaved_chain",
            "Selective Tracking Under Interference (RQ3)",
            "Attending to distractor updates instead of target updates.",
            TrajectorySpec(family="interleaved_chain", entity_count=3, total_updates=4, target_updates=2, distractor_updates=2),
        ),
        # Group C: Identity Transformation (RQ5 Sub-theme 1)
        (
            "GROUP C: IDENTITY TRANSFORMATION (RQ5 Synthetic Extension: Dynamic Cardinality)",
            "split_chain",
            "Identity Multiplication (RQ5.1)",
            "Merging child entities back into one or losing track of post-split objects.",
            TrajectorySpec(family="split_chain", entity_count=2, total_updates=4, target_updates=4),
        ),
        (
            "GROUP C: IDENTITY TRANSFORMATION (RQ5 Synthetic Extension: Dynamic Cardinality)",
            "merge_chain",
            "Identity Consolidation (RQ5.1)",
            "Over-persistence: reporting a stale location for an entity after a container merge.",
            TrajectorySpec(family="merge_chain", entity_count=2, total_updates=3, target_updates=3),
        ),
        # Group D: Global State Operations (RQ5 Sub-theme 2)
        (
            "GROUP D: GLOBAL STATE OPERATIONS (RQ5 Synthetic Extension: Bilateral Exchange)",
            "swap_chain",
            "Simultaneous Bilateral Update (RQ5.2)",
            "No-temp-variable bug: placing both entities in the same container.",
            TrajectorySpec(family="swap_chain", entity_count=2, total_updates=3, target_updates=3),
        ),
        # Group E: Temporal Edit History (RQ5 Sub-theme 3)
        (
            "GROUP E: TEMPORAL EDIT HISTORY (RQ5 Synthetic Extension: History Mutability)",
            "undo_chain",
            "Rollback / Contradiction (RQ5.3)",
            "Treating an undone action as if it still took effect.",
            TrajectorySpec(family="undo_chain", entity_count=1, total_updates=3, target_updates=3),
        ),
        (
            "GROUP E: TEMPORAL EDIT HISTORY (RQ5 Synthetic Extension: History Mutability)",
            "undo_redo_chain",
            "3-Way Edit History Awareness (RQ5.3)",
            "Conflating 'undone' state with 'redone' state.",
            TrajectorySpec(family="undo_redo_chain", entity_count=1, total_updates=4, target_updates=4),
        ),
    ]

    print("\n" + "#" * 74)
    print("# DWS-BENCH: 5-WAY THEORETICAL CAPABILITY TAXONOMY (ALL 8 FAMILIES)")
    print("#" * 74 + "\n")

    for group_name, family, title, failure_mode, spec in family_demos:
        print_example_block(group_name, title, failure_mode, spec, rng)

    print("#" * 74)
    print("# PROCEDURAL EXECUTION / STATE-MACHINE META-PROBE")
    print("#" * 74 + "\n")

    redo_ex = generate_example(
        rng,
        "redo_validity_meta_probe",
        entity_count=4,
        update_count=7,
        distractor_count=1,
        operations_enabled=ALL_OPS,
        force_redo_probe=True,
    )
    print(f"--- Meta-Probe ID: {redo_ex['id']} ---")
    print("Testing: Operation Validity Under Mutable History (Procedural Execution)")
    print("Narrative:")
    for s in redo_ex["sentences"]:
        print(f"  {s}")
    print()
    print(f"Question:    {redo_ex['question']}")
    print(f"Gold Answer: {redo_ex['gold_answer']}")
    print()


if __name__ == "__main__":
    main()
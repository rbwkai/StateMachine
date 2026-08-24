"""End-to-end demo: generates and prints examples for all 8 trajectory chains
grouped by the Five-Group Capability Taxonomy, plus procedural meta-probes,
with explicit parameter and measured factor accounting.

Run with:
    python3 example.py
"""

from __future__ import annotations

import random
from typing import Any, Dict

from generator.dataset_spec import CapabilityGroup, family_capability_group
from generator.metadata import measure_factors
from generator.trajectories import available_families, build_trajectory
from generator.trajectory_specs import TrajectorySpec
from pipeline import generate_example
from render.names import NameRegistry
from render.narrative import question_location, render_narrative
from world import Merge, Move, Put, Redo, Remove, Split, Swap, Undo

ALL_OPS = [Put, Move, Remove, Undo, Redo, Split, Merge, Swap]


def print_example_block(
    group_name: str,
    family_title: str,
    failure_hypothesis: str,
    spec: TrajectorySpec,
    rng: random.Random,
) -> None:
    traj = build_trajectory(rng, spec)
    names = NameRegistry(containers=traj.containers, rng=rng)
    sentences, final_state = render_narrative(traj.ops, traj.containers, names)
    question = question_location(traj.target_obj, final_state, names)
    target_container = final_state.location.get(traj.target_obj)
    target_name = names.container(target_container) if target_container else "<nowhere>"

    # Compute measured factors
    m = measure_factors(traj.ops, traj.containers, traj.target_obj, sentences=sentences)

    initial_placements = sum(1 for op in traj.ops if isinstance(op, Put))
    post_init_updates = sum(1 for op in traj.ops if not isinstance(op, Put))
    total_transitions = len(traj.ops)

    print("=" * 74)
    print(f"CAPABILITY GROUP: {group_name}")
    print(f"FAMILY:           {spec.family.upper()} ({family_title})")
    print(f"Targeted Hypoth.: {failure_hypothesis}")
    print("-" * 74)
    print("Parameter Accounting & Measured Factors:")
    print(f"  - Entity Load (E)              = {m.E_actual}")
    print(f"  - Target-Relevant Ops (T)      = {m.T_actual}")
    print(f"  - Distractor Ops (D)           = {m.D_actual}")
    print(f"  - State Revision Count (V)     = {m.V_actual}")
    print(f"  - Post-Init Operations (U)     = {post_init_updates}  (bookkeeping: U = T + D)")
    print(f"  - Total Symbolic Ops (S)       = {total_transitions}  (S = E + U)")
    print(f"  - Rendered Sentences           = {len(sentences)}")
    print(f"  - Textual Distractors (N)      = {m.N_actual}")
    print(f"  - Rendered Length (L_word)     = {m.L_word} words  (proxy for tokenizer tokens)")
    print(f"  - Measured Factor Tuple        = (E={m.E_actual}, T={m.T_actual}, D={m.D_actual}, V={m.V_actual}, L_word={m.L_word}, N={m.N_actual})")
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
            "GROUP A: SEQUENTIAL STATE TRACKING (Core RQ1 / Naturalistic Reference)",
            "basic_chain",
            "Temporal Depth (RQ1)",
            "Degradation as temporal update depth increases under clean baseline conditions.",
            TrajectorySpec(family="basic_chain", entity_count=1, total_updates=4, target_updates=4),
        ),
        (
            "GROUP A: SEQUENTIAL STATE TRACKING (Core RQ2 / Naturalistic Reference)",
            "revision",
            "State Revision (RQ2)",
            "Interference from previously established historical states at matched temporal depth.",
            TrajectorySpec(family="revision", entity_count=1, total_updates=6, target_updates=6),
        ),
        # Group B: Multi-Entity Interference
        (
            "GROUP B: MULTI-ENTITY INTERFERENCE (Core RQ3 / Naturalistic Reference)",
            "interleaved_chain",
            "Distractor Interference (RQ3)",
            "Susceptibility to irrelevant state-transition interference at constant entity load.",
            TrajectorySpec(family="interleaved_chain", entity_count=3, total_updates=4, target_updates=2, distractor_updates=2),
        ),
        # Group C: Identity Transformation (RQ5 Pilot)
        (
            "GROUP C: IDENTITY TRANSFORMATION (RQ5 Structural Pilot: Dynamic Cardinality)",
            "split_chain",
            "Identity Branching (RQ5.1)",
            "Failure to track branched child identity separately from the original target.",
            TrajectorySpec(family="split_chain", entity_count=2, total_updates=4, target_updates=4),
        ),
        (
            "GROUP C: IDENTITY TRANSFORMATION (RQ5 Structural Pilot: Dynamic Cardinality)",
            "merge_chain",
            "State Consolidation (RQ5.1)",
            "Over-persistence: failing to apply container-level relocation to merged entities.",
            TrajectorySpec(family="merge_chain", entity_count=2, total_updates=3, target_updates=3),
        ),
        # Group D: Global State Operations (RQ5 Pilot)
        (
            "GROUP D: GLOBAL STATE OPERATIONS (RQ5 Structural Pilot: Bilateral Exchange)",
            "swap_chain",
            "Bilateral Exchange (RQ5.2)",
            "Relational state exchange failure (unilateral overwrite error).",
            TrajectorySpec(family="swap_chain", entity_count=2, total_updates=3, target_updates=3),
        ),
        # Group E: Temporal Edit History (RQ5 Pilot)
        (
            "GROUP E: TEMPORAL EDIT HISTORY (RQ5 Structural Pilot: History Mutability)",
            "undo_chain",
            "State Rollback (RQ5.3)",
            "Treating an undone action as if it still took effect.",
            TrajectorySpec(family="undo_chain", entity_count=1, total_updates=3, target_updates=3),
        ),
        (
            "GROUP E: TEMPORAL EDIT HISTORY (RQ5 Structural Pilot: History Mutability)",
            "undo_redo_chain",
            "3-Way Edit History Awareness (RQ5.3)",
            "Conflating undone state with redone state during history cycles.",
            TrajectorySpec(family="undo_redo_chain", entity_count=1, total_updates=4, target_updates=4),
        ),
    ]

    print("\n" + "#" * 74)
    print("# DWS-BENCH: FIVE-GROUP CAPABILITY TAXONOMY (ALL 8 TRAJECTORY FAMILIES)")
    print("#" * 74 + "\n")

    for group_name, family, title, failure_hypothesis, spec in family_demos:
        print_example_block(group_name, title, failure_hypothesis, spec, rng)

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
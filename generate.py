"""
Generation CLI / Script for DWS-Bench Trajectories.
Generates full benchmark records for all trajectory families.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from generator.dataset_spec import (
    CapabilityGroup,
    Condition,
    Experiment,
    GenerationStatus,
    family_capability_group,
)
from generator.trajectories import available_families, build_trajectory
from generator.trajectory_specs import TrajectorySpec
from generator.trajectory_validation import validate_trajectory
from render.names import NameRegistry
from render.narrative import question_location, render_narrative
from world.operations import Move, Put


def generate_family_example(
    family: str,
    rng: random.Random,
    example_id: str,
    entity_count: int = 2,
    total_updates: int = 4,
    target_updates: int = 4,
    distractor_updates: int = 0,
    num_containers: int = 3,
) -> Dict[str, Any]:
    """Generate a single benchmark instance from a trajectory specification."""
    # Adjust defaults per family constraints if necessary
    if family in ("basic_chain", "revision", "undo_chain", "undo_redo_chain"):
        entity_count = 1
    elif family in ("split_chain", "swap_chain"):
        entity_count = 2
    elif family in ("interleaved_chain", "merge_chain"):
        entity_count = max(2, entity_count)

    if family in ("undo_redo_chain",) and target_updates < 3:
        target_updates = 4
        total_updates = 4

    if family == "interleaved_chain":
        target_updates = max(1, total_updates // 2)
        distractor_updates = total_updates - target_updates

    spec = TrajectorySpec(
        family=family,
        entity_count=entity_count,
        num_containers=num_containers,
        total_updates=total_updates,
        target_updates=target_updates,
        distractor_updates=distractor_updates,
    )

    # 1. Build symbolic trajectory
    trajectory = build_trajectory(rng, spec)

    # 2. Structural validation
    validate_trajectory(trajectory.ops, trajectory.target_obj, trajectory.spec)

    # 3. Render natural language narrative
    names = NameRegistry(containers=trajectory.containers, rng=rng)
    sentences, final_state = render_narrative(
        trajectory.ops, trajectory.containers, names
    )

    # 4. Target question and answer
    question = question_location(trajectory.target_obj, final_state, names)
    target_container = final_state.location.get(trajectory.target_obj)
    target_name = names.container(target_container) if target_container else None

    # 5. Build step-by-step trace
    step_trace: List[Dict[str, Any]] = []
    for step_idx, (op, sentence) in enumerate(zip(trajectory.ops, sentences)):
        step_trace.append({
            "step": step_idx,
            "op": op.__class__.__name__.upper(),
            "sentence": sentence,
            "details": str(op),
        })

    initial_placements = len([op for op in trajectory.ops if isinstance(op, Put)])
    post_init_updates = len([op for op in trajectory.ops if not isinstance(op, Put)])
    total_transitions = len(trajectory.ops)

    record = {
        "example_id": example_id,
        "family": family,
        "capability_group": family_capability_group(family).name,
        "spec": {
            "entity_count": spec.entity_count,
            "total_updates": spec.total_updates,
            "target_updates": spec.target_updates,
            "distractor_updates": spec.distractor_updates,
            "num_containers": spec.num_containers,
            "initial_placements": initial_placements,
            "post_init_updates": post_init_updates,
            "total_transitions": total_transitions,
        },
        "context": " ".join(sentences),
        "sentences": sentences,
        "question": question,
        "gold_answer": target_name,
        "gold_container": target_container,
        "target_obj": trajectory.target_obj,
        "step_trace": step_trace,
        "final_state": {
            "location": final_state.location,
            "containers": sorted(final_state.containers),
        },
    }

    return record


def main():
    parser = argparse.ArgumentParser(description="Generate DWS-Bench trajectory records.")
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        help="Family name to generate (or 'all' for all 8 families)",
    )
    parser.add_argument("--count", type=int, default=1, help="Number of instances per family")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSONL filepath")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.family == "all":
        families = available_families()
    else:
        if args.family not in available_families():
            raise ValueError(f"Unknown family: {args.family}. Available: {available_families()}")
        families = [args.family]

    generated_records = []

    for fam in families:
        print(f"\n{'='*70}\nFAMILY: {fam}\n{'='*70}")
        for i in range(args.count):
            ex_id = f"{fam}_{args.seed}_{i}"
            rec = generate_family_example(family=fam, rng=rng, example_id=ex_id)
            generated_records.append(rec)

            print(f"[{ex_id}]")
            print(f"Story:")
            for s in rec["sentences"]:
                print(f"  - {s}")
            print(f"Question:    {rec['question']}")
            print(f"Gold Answer: {rec['gold_answer']} ({rec['gold_container']})")
            print(f"Final State: {rec['final_state']['location']}")
            print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in generated_records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nWrote {len(generated_records)} records to {args.output}")


if __name__ == "__main__":
    main()

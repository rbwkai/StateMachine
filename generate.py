"""
generate.py
===========
Generation CLI / Script for DWS-Bench Trajectories.

Every record this CLI emits comes from ``generator.instance`` — the same
request -> validated-record path ``experiments/_common.py`` uses, including the
structural, factor and length gate (SPEC §3, requirements.md §7). The CLI adds
no fields of its own: ``build_validated_instance`` owns the record schema, so
``generate.py`` output and the experiment sweeps are interchangeable and
``eval/eval_harness.py`` / ``run_eval.py`` read the same ``instance_id`` key from
both.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from generator.dataset_spec import family_capability_group
from generator.instance import generate_instance_with_retry
from generator.trajectories import available_families
from generator.trajectory_specs import TrajectorySpec


# Records are not part of an RQ sweep, but the unified schema carries an
# `experiment` / `condition_id` pair, so a CLI run is labelled rather than left
# empty and becomes aggregatable with the sweep data by generate_all.py.
CLI_EXPERIMENT_TAG = "generate_cli"


def generate_family_example(
    family: str,
    rng: random.Random,
    instance_id: str,
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

    # One draw from the caller's stream becomes this instance's seed, so
    # successive CLI instances differ while a given --seed still reproduces the
    # same records. Trajectory construction, rendering, measurement and the gate
    # all happen inside the shared path; nothing is re-derived here.
    instance_seed = rng.getrandbits(32)

    return generate_instance_with_retry(
        seed=instance_seed,
        spec=spec,
        instance_id=instance_id,
        experiment=CLI_EXPERIMENT_TAG,
        condition_id=(
            f"{family}_T{target_updates}_D{distractor_updates}_E{entity_count}"
        ),
    )


def _print_record(record: Dict[str, Any]) -> None:
    print(f"[{record['instance_id']}]")
    print("Story:")
    for sentence in record["sentences"]:
        print(f"  - {sentence}")
    print(f"Question:    {record['question']}")
    print(
        f"Gold Answer: {record['gold_answer']} "
        f"({record['gold_container']})"
    )
    print(f"Final State: {record['final_state']['location']}")
    print(f"Requested:  {json.dumps(record['requested_factors'])}")
    print(f"Measured:   {json.dumps(record['measured_factors'])}")
    print(f"trace_hash: {record['trace_hash']}")
    print(
        "canonical_trace: "
        f"{len(record['canonical_trace'])} ops, "
        f"step_wise_gold: {json.dumps(record['step_wise_gold'])}"
    )
    print(
        "versions: "
        f"generator={record['generator_version']} "
        f"renderer={record['renderer_version']} "
        f"scoring={record['scoring_version']}"
    )


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

    generated_records: List[Dict[str, Any]] = []

    for fam in families:
        print(f"\n{'='*70}\nFAMILY: {fam}\n{'='*70}")
        for i in range(args.count):
            instance_id = f"{fam}_{args.seed}_{i}"
            rec = generate_family_example(
                family=fam,
                rng=rng,
                instance_id=instance_id,
            )
            generated_records.append(rec)

            _print_record(rec)
            print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in generated_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(generated_records)} records to {args.output}")

    print("\nFamilies covered: " + ", ".join(
        f"{fam} ({family_capability_group(fam).name})" for fam in families
    ))


if __name__ == "__main__":
    main()
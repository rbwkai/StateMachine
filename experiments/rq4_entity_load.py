"""
experiments/rq4_entity_load.py
===============================
RQ4: Entity-load sweep with a fixed target dependency chain.

Design:
    family             : interleaved_chain
    entity counts      : 2, 3, 4, 5
    target updates     : 8
    distractor updates : 4
    instances/condition: 100
    total              : 400

The target receives the same requested number of updates in every condition.
Additional entities increase the pool of distractor entities while the total
number of distractor updates remains fixed. Realized word length is measured
and must be reported as a possible residual confound.

Usage:
    python3 experiments/rq4_entity_load.py
    python3 experiments/rq4_entity_load.py --dry-run
    python3 experiments/rq4_entity_load.py --instances 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import generate_condition, probe_reachability, write_jsonl


FAMILY = "interleaved_chain"
ENTITY_LEVELS = [2, 3, 4, 5]
TARGET_UPDATES = 8
DISTRACTOR_UPDATES = 4
NUM_CONTAINERS = 6
INSTANCES_PER_CONDITION = 50
EXPERIMENT_TAG = "rq4_entity_load"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ4 Entity Load Sweep — E∈{2,3,4,5}, T=8, D=4"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_CONDITION)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq4_entity_load" / "rq4_entity_load.jsonl"
    )

    print("=" * 70)
    print("RQ4 — Entity Load Sweep")
    print(f"  family        : {FAMILY}")
    print(f"  E levels      : {ENTITY_LEVELS}")
    print(f"  T (fixed)     : {TARGET_UPDATES}")
    print(f"  D (fixed)     : {DISTRACTOR_UPDATES}")
    print(f"  instances/lvl : {args.instances}")
    print(f"  total target  : {len(ENTITY_LEVELS) * args.instances}")
    print("=" * 70)

    print("\n[Step 1] Reachability probe (10 seeds per entity level)...")
    all_ok = True
    for entity_count in ENTITY_LEVELS:
        ok = probe_reachability(
            family=FAMILY,
            entity_count=entity_count,
            target_updates=TARGET_UPDATES,
            distractor_updates=DISTRACTOR_UPDATES,
            num_containers=NUM_CONTAINERS,
            n_seeds=10,
        )
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit("RQ4 reachability probe failed; no data generated.")

    if args.dry_run:
        print("\n[DRY-RUN] All entity-load conditions are reachable.")
        return

    print("\n[Step 2] Generating instances...")
    records = []
    failures = 0
    start = time.perf_counter()

    for index, entity_count in enumerate(ENTITY_LEVELS):
        condition_id = f"E{entity_count}"
        condition_label = (
            f"{FAMILY} E={entity_count} T={TARGET_UPDATES} D={DISTRACTOR_UPDATES}"
        )
        condition_records, condition_failures = generate_condition(
            family=FAMILY,
            entity_count=entity_count,
            target_updates=TARGET_UPDATES,
            distractor_updates=DISTRACTOR_UPDATES,
            num_instances=args.instances,
            experiment_tag=EXPERIMENT_TAG,
            condition_id=condition_id,
            num_containers=NUM_CONTAINERS,
            condition_label=condition_label,
        )
        records.extend(condition_records)
        failures += condition_failures

    write_jsonl(records, output_path)
    elapsed = time.perf_counter() - start
    print("\n" + "=" * 70)
    print("RQ4 GENERATION COMPLETE")
    print(f"  Total generated : {len(records)}")
    print(f"  Total failures  : {failures}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print(f"  Output          : {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
experiments/rq2_revision.py
============================
RQ2: Revision Complexity Sweep (E=1, D=0, V≥2)

Design:
    family          : revision
    entity_count    : 1
    distractor_updates: 0
    target_updates  : 4, 8, 12, 16
    instances/cond  : 50
    total           : 200

Control (V=0) is the RQ1 basic_chain data at the same T values.
No duplicate generation needed.

The `revision` trajectory family naturally produces location revisits.
We filter to keep only instances with measured V_actual ≥ MIN_V_ACTUAL (2)
to satisfy the V=2 condition from §5.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import generate_condition, probe_reachability, write_jsonl


# ============================================================
# Experimental design (from §5 of the research plan)
# ============================================================

FAMILY           = "revision"
ENTITY_COUNT     = 1
DISTRACTOR_UPDATES = 0
NUM_CONTAINERS   = 3
INSTANCES_PER_CONDITION = 50
EXPERIMENT_TAG   = "rq2_revision"
MIN_V_ACTUAL_DEFAULT = 2

T_LEVELS = [4, 8, 12, 16]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Revision Sweep — E=1, D=0, V≥2, T∈{4,8,12,16}"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_CONDITION)
    parser.add_argument(
        "--min-v", type=int, default=MIN_V_ACTUAL_DEFAULT,
        help=f"Minimum measured V_actual (default: {MIN_V_ACTUAL_DEFAULT})",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq2_revision" / "rq2_revision.jsonl"
    )

    print("=" * 70)
    print("RQ2 — Revision Complexity Sweep")
    print(f"  family        : {FAMILY}")
    print(f"  E             : {ENTITY_COUNT}")
    print(f"  D             : {DISTRACTOR_UPDATES}")
    print(f"  V ≥           : {args.min_v}")
    print(f"  T levels      : {T_LEVELS}")
    print(f"  instances/lvl : {args.instances}")
    print(f"  total target  : {len(T_LEVELS) * args.instances}")
    if args.dry_run:
        print("  MODE          : DRY-RUN")
    else:
        print(f"  output        : {output_path}")
    print("=" * 70)

    if args.dry_run:
        print("\nProbing reachability (10 seeds per T level)…")
        for T in T_LEVELS:
            probe_reachability(
                family=FAMILY,
                entity_count=ENTITY_COUNT,
                target_updates=T,
                distractor_updates=0,
                n_seeds=10,
            )
        print("\n[NOTE] V filtering is applied at generation time, not here.")
        return

    all_records: List[Dict[str, Any]] = []
    total_failures = 0
    t0 = time.perf_counter()

    for T in T_LEVELS:
        condition_id = f"T{T}"
        records, failures = generate_condition(
            family=FAMILY,
            entity_count=ENTITY_COUNT,
            target_updates=T,
            distractor_updates=0,
            num_instances=args.instances,
            experiment_tag=EXPERIMENT_TAG,
            condition_id=condition_id,
            num_containers=NUM_CONTAINERS,
            condition_label=f"revision E=1 T={T} V≥{args.min_v}",
            min_v=args.min_v,
        )
        all_records.extend(records)
        total_failures += failures

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 70)
    print("RQ2 GENERATION COMPLETE")
    print(f"  Total generated : {len(all_records)}")
    print(f"  Total failures  : {total_failures}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 70)

    print("\nPer-condition summary:")
    for T in T_LEVELS:
        cond = [r for r in all_records if r["requested_factors"]["T"] == T]
        v_vals = [r["measured_factors"]["V_actual"] for r in cond]
        if v_vals:
            print(
                f"  T={T:2d} : {len(cond):3d} instances  "
                f"V_actual ∈ [{min(v_vals)}, {max(v_vals)}]"
            )
        else:
            print(f"  T={T:2d} : 0 instances (all failed)")

    write_jsonl(all_records, output_path)


if __name__ == "__main__":
    main()

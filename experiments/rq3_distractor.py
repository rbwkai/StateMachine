"""
experiments/rq3_distractor.py
==============================
RQ3: Distractor / Interference Sweep (E=3, T=8, V=0)

Design:
    family          : interleaved_chain
    entity_count    : 3
    target_updates  : 8 (fixed)
    distractor_updates: 4, 8, 16   (D=0 is already in RQ1 at T=8)
    instances/cond  : 100
    new instances   : 300

The D=0 condition at T=8 is available from RQ1.  This script generates
only the three non-zero D conditions.

Usage:
    python3 experiments/rq3_distractor.py
    python3 experiments/rq3_distractor.py --dry-run
    python3 experiments/rq3_distractor.py --include-d0   # also generate D=0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import (
    generate_condition,
    probe_reachability,
    write_jsonl,
)


# ============================================================
# Experimental design (from §6 of the research plan)
# ============================================================

FAMILY          = "interleaved_chain"
ENTITY_COUNT    = 3
TARGET_UPDATES  = 8     # T fixed across all conditions
NUM_CONTAINERS  = 4     # More containers for higher D levels
INSTANCES_PER_CONDITION = 100
EXPERIMENT_TAG  = "rq3_distractor"
BASE_SEED       = 3000

# D=0 already exists in RQ1 — only generate D>0 by default.
D_LEVELS_DEFAULT = [4, 8, 16]
D_LEVELS_WITH_D0 = [0, 4, 8, 16]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ3 Distractor Sweep — E=3, T=8, D∈{4,8,16}"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_CONDITION)
    parser.add_argument(
        "--include-d0", action="store_true",
        help="Also generate D=0 condition (already in RQ1 data at T=8)",
    )
    args = parser.parse_args()

    D_LEVELS: List[int] = D_LEVELS_WITH_D0 if args.include_d0 else D_LEVELS_DEFAULT

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq3_distractor" / "rq3_distractor.jsonl"
    )

    print("=" * 70)
    print("RQ3 — Distractor / Interference Sweep")
    print(f"  family        : {FAMILY}")
    print(f"  E             : {ENTITY_COUNT}")
    print(f"  T (fixed)     : {TARGET_UPDATES}")
    print(f"  D levels      : {D_LEVELS}")
    print(f"  containers    : {NUM_CONTAINERS}")
    print(f"  instances/lvl : {args.instances}")
    print(f"  total         : {len(D_LEVELS) * args.instances}")
    if args.dry_run:
        print("  MODE          : DRY-RUN")
    else:
        print(f"  output        : {output_path}")
    print("=" * 70)

    if args.dry_run:
        print("\nProbing reachability (10 seeds per D level)…")
        all_ok = True
        for D in D_LEVELS:
            ok = probe_reachability(
                family=FAMILY,
                entity_count=ENTITY_COUNT,
                target_updates=TARGET_UPDATES,
                distractor_updates=D,
                num_containers=NUM_CONTAINERS,
                n_seeds=10,
            )
            if not ok:
                all_ok = False
        if all_ok:
            print("\n[OK] All D levels reachable.")
        else:
            print("\n[WARN] Some D levels had failures.")
        return

    all_records = []
    total_failures = 0
    t0 = time.perf_counter()

    for D in D_LEVELS:
        condition_label = (
            f"interleaved_chain E={ENTITY_COUNT} T={TARGET_UPDATES} D={D}"
        )

        records, failures = generate_condition(
            family=FAMILY,
            entity_count=ENTITY_COUNT,
            target_updates=TARGET_UPDATES,
            distractor_updates=D,
            num_instances=args.instances,
            experiment_tag=EXPERIMENT_TAG,
            base_seed=BASE_SEED + D * 100,
            num_containers=NUM_CONTAINERS,
            condition_label=condition_label,
        )

        all_records.extend(records)
        total_failures += failures

    elapsed = time.perf_counter() - t0

    print()
    print("=" * 70)
    print("RQ3 GENERATION COMPLETE")
    print(f"  Total generated : {len(all_records)}")
    print(f"  Total failures  : {total_failures}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 70)

    print("\nPer-condition summary:")
    for D in D_LEVELS:
        cond = [r for r in all_records if r["requested_factors"]["D"] == D]
        T_actuals = [r["measured_factors"]["T_actual"] for r in cond]
        D_actuals = [r["measured_factors"]["D_actual"] for r in cond]
        if cond:
            print(
                f"  D={D:2d} : {len(cond):3d} instances  "
                f"T_actual ∈ {{{min(T_actuals)},{max(T_actuals)}}}  "
                f"D_actual ∈ {{{min(D_actuals)},{max(D_actuals)}}}"
            )
        else:
            print(f"  D={D:2d} : 0 instances (all failed)")

    write_jsonl(all_records, output_path)


if __name__ == "__main__":
    main()

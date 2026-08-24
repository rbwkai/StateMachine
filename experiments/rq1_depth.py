"""
experiments/rq1_depth.py
========================
RQ1: Temporal Depth Sweep (E=1, D=0, V=0)

Design:
    family          : basic_chain
    entity_count    : 1          ← clean causal interpretation (§4)
    distractor_updates: 0
    target_updates  : 2, 4, 6, 8, 12, 16
    instances/cond  : 100
    total           : 600

Each instance uses a distinct seed (base_seed + instance_idx) for
structural diversity.

Usage:
    python3 experiments/rq1_depth.py                  # full run (600)
    python3 experiments/rq1_depth.py --dry-run        # 10-seed reachability probe
    python3 experiments/rq1_depth.py --output data/rq1_depth/custom.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import (
    generate_condition,
    probe_reachability,
    write_jsonl,
)


# ============================================================
# Experimental design (from §3 of the research plan)
# ============================================================

FAMILY           = "basic_chain"
ENTITY_COUNT     = 1          # E=1 — clean single-entity baseline (§4)
DISTRACTOR_UPDATES = 0        # D=0 — no interference
NUM_CONTAINERS   = 3
INSTANCES_PER_CONDITION = 100
EXPERIMENT_TAG   = "rq1_depth"
BASE_SEED        = 1000       # Distinct from other experiments

T_LEVELS = [2, 4, 6, 8, 12, 16]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ1 Temporal Depth Sweep — E=1, D=0, T∈{2,4,6,8,12,16}"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe reachability (10 seeds/condition) without full generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: data/rq1_depth/rq1_depth.jsonl)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=INSTANCES_PER_CONDITION,
        help=f"Instances per T level (default: {INSTANCES_PER_CONDITION})",
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq1_depth" / "rq1_depth.jsonl"
    )

    print("=" * 70)
    print("RQ1 — Temporal Depth Sweep")
    print(f"  family        : {FAMILY}")
    print(f"  E             : {ENTITY_COUNT}")
    print(f"  D             : {DISTRACTOR_UPDATES}")
    print(f"  T levels      : {T_LEVELS}")
    print(f"  instances/lvl : {args.instances}")
    print(f"  total         : {len(T_LEVELS) * args.instances}")
    if args.dry_run:
        print("  MODE          : DRY-RUN (reachability probe only)")
    else:
        print(f"  output        : {output_path}")
    print("=" * 70)

    # ----------------------------------------------------------
    # Dry-run: reachability probe
    # ----------------------------------------------------------

    if args.dry_run:
        print("\nProbing reachability (10 seeds per T level)…")
        all_ok = True
        for T in T_LEVELS:
            ok = probe_reachability(
                family=FAMILY,
                entity_count=ENTITY_COUNT,
                target_updates=T,
                distractor_updates=DISTRACTOR_UPDATES,
                num_containers=NUM_CONTAINERS,
                n_seeds=10,
            )
            if not ok:
                all_ok = False

        if all_ok:
            print("\n[OK] All T levels are reachable. Safe to run full generation.")
        else:
            print("\n[WARN] Some T levels had failures. Review before full generation.")
        return

    # ----------------------------------------------------------
    # Full generation
    # ----------------------------------------------------------

    all_records = []
    total_failures = 0
    t0 = time.perf_counter()

    for T in T_LEVELS:
        condition_label = (
            f"basic_chain E={ENTITY_COUNT} T={T} D={DISTRACTOR_UPDATES}"
        )

        records, failures = generate_condition(
            family=FAMILY,
            entity_count=ENTITY_COUNT,
            target_updates=T,
            distractor_updates=DISTRACTOR_UPDATES,
            num_instances=args.instances,
            experiment_tag=EXPERIMENT_TAG,
            base_seed=BASE_SEED + T * 1000,
            num_containers=NUM_CONTAINERS,
            condition_label=condition_label,
        )

        all_records.extend(records)
        total_failures += failures

    elapsed = time.perf_counter() - t0

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print(f"RQ1 GENERATION COMPLETE")
    print(f"  Total generated : {len(all_records)}")
    print(f"  Total failures  : {total_failures}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 70)

    # Per-condition breakdown
    print("\nPer-condition summary:")
    for T in T_LEVELS:
        cond_records = [
            r for r in all_records
            if r["requested_factors"]["T"] == T
        ]
        T_actuals = [r["measured_factors"]["T_actual"] for r in cond_records]
        if T_actuals:
            assert all(t == T for t in T_actuals), \
                f"T_actual mismatch at requested T={T}: {set(T_actuals)}"
        print(
            f"  T={T:2d} : {len(cond_records):3d} instances"
            + (f"  [T_actual verified ✓]" if cond_records else "")
        )

    if total_failures > 0:
        print(
            f"\n[WARN] {total_failures} instances failed to generate. "
            "Check stderr for details."
        )

    write_jsonl(all_records, output_path)


if __name__ == "__main__":
    main()

"""
experiments/rq5_pilot.py
=========================
RQ5 Structural Pilot: T=8 across 5 structural families

Design (from §7 of the research plan):

    Family         E   T   D   V   S=E+U
    split_chain    2   8   0   0   9
    merge_chain    2   8   0   0   10
    swap_chain     2   8   0   0   10
    undo_chain     1   8   0   0   9
    undo_redo_chain 1  8   0   0   9

    100 instances per family = 500 total.

Purpose (§7):
    Can these additional operation families be generated reliably
    at a common depth T=8 and evaluated using the same framework?

Before generating 100 instances, a 10-seed reachability probe runs for
each family.  If any family fails the probe, the pilot halts with a
diagnostic report rather than silently producing partial data (§8).

Usage:
    python3 experiments/rq5_pilot.py                # full run (500)
    python3 experiments/rq5_pilot.py --dry-run      # reachability probe only
    python3 experiments/rq5_pilot.py --family split_chain  # one family only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import (
    generate_condition,
    probe_reachability,
    write_jsonl,
)


# ============================================================
# Family configurations (§7)
# ============================================================

# Each entry: (family, entity_count, target_updates, distractor_updates,
#              num_containers, description)
PILOT_FAMILIES: List[Tuple] = [
    ("split_chain",     2, 8, 0, 3, "Identity multiplication via Split"),
    ("merge_chain",     2, 8, 0, 3, "Identity consolidation via Merge"),
    ("swap_chain",      2, 8, 0, 3, "Bilateral exchange via Swap"),
    ("undo_chain",      1, 8, 0, 3, "Rollback/contradiction via Undo"),
    ("undo_redo_chain", 1, 8, 0, 3, "3-way edit history via Undo+Redo"),
]

INSTANCES_PER_FAMILY = 100
EXPERIMENT_TAG       = "rq5_pilot"
BASE_SEED            = 5000


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RQ5 Structural Pilot — T=8, 5 families × 100 instances"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Reachability probe only (10 seeds/family)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_FAMILY)
    parser.add_argument(
        "--family", type=str, default=None,
        help="Generate one family only (e.g. split_chain)",
    )
    args = parser.parse_args()

    # Filter families if --family specified.
    families = PILOT_FAMILIES
    if args.family:
        families = [f for f in PILOT_FAMILIES if f[0] == args.family]
        if not families:
            names = [f[0] for f in PILOT_FAMILIES]
            print(f"Unknown family {args.family!r}. Available: {names}")
            sys.exit(1)

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq5_pilot" / "rq5_pilot.jsonl"
    )

    print("=" * 70)
    print("RQ5 Structural Pilot")
    print(f"  T (fixed)     : 8")
    print(f"  families      : {[f[0] for f in families]}")
    print(f"  instances/fam : {args.instances}")
    print(f"  total target  : {len(families) * args.instances}")
    if args.dry_run:
        print("  MODE          : DRY-RUN (reachability probe)")
    else:
        print(f"  output        : {output_path}")
    print("=" * 70)

    # ----------------------------------------------------------
    # STEP 1: Reachability probe (§8 requirement)
    # Always run before full generation.
    # ----------------------------------------------------------

    print("\n[Step 1] Reachability probe (10 seeds × each family)…")

    probe_results: Dict[str, bool] = {}

    for (family, E, T, D, num_containers, desc) in families:
        ok = probe_reachability(
            family=family,
            entity_count=E,
            target_updates=T,
            distractor_updates=D,
            num_containers=num_containers,
            n_seeds=10,
        )
        probe_results[family] = ok

    failed_families = [f for f, ok in probe_results.items() if not ok]

    if failed_families:
        print()
        print("=" * 70)
        print("[HALT] The following families failed the T=8 reachability probe:")
        for f in failed_families:
            print(f"  ✗  {f}")
        print()
        print(
            "These families cannot reliably generate valid trajectories at T=8.\n"
            "Review and fix the trajectory constructors before running the pilot.\n"
            "Pilot aborted — no instances generated."
        )
        print("=" * 70)
        sys.exit(1)

    print("\n  All families passed the reachability probe ✓")

    if args.dry_run:
        print("\n[DRY-RUN] Probe complete. No instances generated.")
        return

    # ----------------------------------------------------------
    # STEP 2: Full generation
    # ----------------------------------------------------------

    print("\n[Step 2] Generating instances…")

    all_records = []
    total_failures = 0
    family_stats: Dict[str, Dict] = {}

    t0 = time.perf_counter()

    for i, (family, E, T, D, num_containers, desc) in enumerate(families):
        print(f"\n  Family {i+1}/{len(families)}: {family}")
        print(f"  Description: {desc}")

        records, failures = generate_condition(
            family=family,
            entity_count=E,
            target_updates=T,
            distractor_updates=D,
            num_instances=args.instances,
            experiment_tag=EXPERIMENT_TAG,
            base_seed=BASE_SEED + i * 200,
            num_containers=num_containers,
            condition_label=f"{family} E={E} T={T} D={D}",
        )

        all_records.extend(records)
        total_failures += failures

        family_stats[family] = {
            "generated": len(records),
            "failures": failures,
            "T_actuals": [r["measured_factors"]["T_actual"] for r in records],
            "E_actuals": [r["measured_factors"]["E_actual"] for r in records],
        }

    elapsed = time.perf_counter() - t0

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print()
    print("=" * 70)
    print("RQ5 PILOT COMPLETE")
    print(f"  Total generated : {len(all_records)}")
    print(f"  Total failures  : {total_failures}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 70)

    print("\nPer-family summary:")
    for family, stats in family_stats.items():
        T_a = stats["T_actuals"]
        E_a = stats["E_actuals"]
        T_range = f"[{min(T_a)},{max(T_a)}]" if T_a else "—"
        E_range = f"[{min(E_a)},{max(E_a)}]" if E_a else "—"
        print(
            f"  {family:20s} : {stats['generated']:3d} generated  "
            f"{stats['failures']:2d} failed  "
            f"T_actual∈{T_range}  E_actual∈{E_range}"
        )

    if total_failures > 0:
        print(f"\n[WARN] {total_failures} instances failed. Check stderr.")

    write_jsonl(all_records, output_path)


if __name__ == "__main__":
    main()

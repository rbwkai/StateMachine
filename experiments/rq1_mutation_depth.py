"""
experiments/rq1_mutation_depth.py
==================================
RQ1: Sequential Dependency and Mutation Structure

How does increasing target-relevant state depth affect the ability of small
instruction-tuned language models to recover the final world state, and does
degradation depend on the type of state mutation rather than depth alone?

Core experiment:
  Vary T.
  Compare matched mutation structures.
  Keep E, D, N, and ideally L_tok controlled.
  Compare ordinary forward updates against revision and identity-changing
  operations.
  Analyze both final accuracy and first divergence.

Key hypothesis: At matched depth, different transition structures produce
different effective tracking horizons.

Design:
  family × T grid:
    basic_chain     : T ∈ {2, 4, 6, 8, 12, 16}  (ordinary forward updates)
    revision        : T ∈ {4, 8, 12, 16}         (revisits/supersession)
    split_chain     : T ∈ {4, 8, 12, 16}         (identity multiplication)
    merge_chain     : T ∈ {4, 8, 12, 16}         (identity consolidation)
    swap_chain      : T ∈ {4, 8, 12, 16}         (bilateral exchange)
    undo_chain      : T ∈ {4, 8, 12, 16}         (rollback/contradiction)
    undo_redo_chain : T ∈ {4, 8, 12, 16}         (3-way edit history)

  E=1 for single-entity families, E=2 for structural families
  D=0, N=0
  instances/cond: 50

Usage:
  python3 experiments/rq1_mutation_depth.py                  # full run
  python3 experiments/rq1_mutation_depth.py --dry-run        # reachability probe
  python3 experiments/rq1_mutation_depth.py --family basic_chain  # one family only
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
    ProbeResult,
    build_manifest,
    generate_condition,
    probe_reachability,
    write_jsonl,
    write_manifest,
    verify_generated_records,
    print_factor_summary,
)


# ============================================================
# Experimental design
# ============================================================

# Family configurations: (family, entity_count, T_levels, description)
RQ1_FAMILIES: List[Tuple[str, int, List[int], str]] = [
    ("basic_chain",     1, [2, 4, 6, 8, 12, 16], "Ordinary forward updates (Move chain)"),
    ("revision",        1, [4, 8, 12, 16],       "Revisits/supersession (revisit depth)"),
    ("split_chain",     2, [4, 8, 12, 16],       "Identity multiplication via Split"),
    ("merge_chain",     2, [4, 8, 12, 16],       "Identity consolidation via Merge"),
    ("swap_chain",      2, [4, 8, 12, 16],       "Bilateral exchange via Swap"),
    ("undo_chain",      1, [4, 8, 12, 16],       "Rollback/contradiction via Undo"),
    ("undo_redo_chain", 1, [6, 8, 12, 16],       "3-way edit history via Undo+Redo"),
]

DISTRACTOR_UPDATES = 0
TEXTUAL_DISTRACTORS = 0
INSTANCES_PER_CONDITION = 50
EXPERIMENT_TAG = "rq1_mutation_depth"
condition_id = ""

# Family-specific container defaults
FAMILY_CONTAINERS = {
    "basic_chain": 3,
    "revision": 3,
    "split_chain": 3,
    "merge_chain": 3,
    "swap_chain": 3,
    "undo_chain": 3,
    "undo_redo_chain": 3,
}

# Family-specific query types
FAMILY_QUERY_TYPES = {
    "split_chain": "count",
    "merge_chain": "count",
    "swap_chain": "count",
    "undo_chain": "count",
    "basic_chain": "location",
    "revision": "location",
    "interleaved_chain": "location",
    "undo_redo_chain": "location",
}


def main():
    parser = argparse.ArgumentParser(
        description="RQ1 Sequential Dependency & Mutation Structure"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Reachability probe only (10 seeds/condition)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_CONDITION)
    parser.add_argument(
        "--family", type=str, default=None,
        help="Generate one family only (e.g. basic_chain)"
    )
    args = parser.parse_args()

    # Filter families if --family specified
    families = RQ1_FAMILIES
    if args.family:
        families = [f for f in RQ1_FAMILIES if f[0] == args.family]
        if not families:
            names = [f[0] for f in RQ1_FAMILIES]
            print(f"Unknown family {args.family!r}. Available: {names}")
            sys.exit(1)

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq1_mutation_depth" / "rq1_mutation_depth.jsonl"
    )

    print("=" * 75)
    print("RQ1 — Sequential Dependency and Mutation Structure")
    print(f"  families      : {[f[0] for f in families]}")
    print(f"  D             : {DISTRACTOR_UPDATES}")
    print(f"  N             : {TEXTUAL_DISTRACTORS}")
    print(f"  instances/cond: {args.instances}")
    total_conditions = sum(len(f[2]) for f in families)
    print(f"  total target  : {total_conditions * args.instances}")
    if args.dry_run:
        print("  MODE          : DRY-RUN (reachability probe)")
    else:
        print(f"  output        : {output_path}")
    print("=" * 75)

    # ----------------------------------------------------------
    # STEP 1: Reachability probe (always run)
    # ----------------------------------------------------------

    print("\n[Step 1] Reachability probe (10 seeds × each family×T)…")

    probe_results: Dict[str, ProbeResult] = {}

    for (family, E, T_levels, desc) in families:
        num_containers = FAMILY_CONTAINERS[family]
        for T in T_levels:
            key = f"{family}_T{T}"
            query_type = FAMILY_QUERY_TYPES.get(family, "location")
            probe_results[key] = probe_reachability(
                family=family,
                entity_count=E,
                target_updates=T,
                distractor_updates=DISTRACTOR_UPDATES,
                num_containers=num_containers,
                n_seeds=10,
                query_type=query_type,
            )

    # A cell that no probe seed can generate is excluded from the sweep and
    # named loudly here and in the manifest. A cell that generates on some
    # seeds is kept: generate_condition() counts the failures, so a partial
    # condition shows up as a shortfall rather than as a missing one.
    unreachable = sorted(k for k, r in probe_results.items() if r.excluded)
    degraded = sorted(
        k for k, r in probe_results.items() if not r.excluded and not r.reachable
    )

    if unreachable:
        print()
        print("=" * 75)
        print("[EXCLUDED] These conditions cannot generate a validated instance:")
        for key in unreachable:
            print(f"  ✗  {key}")
        print("They are excluded from this run and listed in the manifest.")
        print("=" * 75)

    if degraded:
        print(f"\n[WARN] {len(degraded)} condition(s) generate on some seeds only "
              f"({', '.join(degraded)}); expect a shortfall against "
              f"{args.instances} instances/cond.")

    if unreachable or degraded:
        kept = len(probe_results) - len(unreachable)
        print(f"\n  {kept}/{len(probe_results)} conditions passed the reachability probe ✓")
    else:
        print("\n  All conditions passed the reachability probe ✓")

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

    for i, (family, E, T_levels, desc) in enumerate(families):
        num_containers = FAMILY_CONTAINERS[family]
        print(f"\n  Family {i+1}/{len(families)}: {family}")
        print(f"  Description: {desc}")

        family_T_actuals = []
        family_E_actuals = []
        family_generated = 0
        family_failures = 0

        for T in T_levels:
            # Use default condition_id which includes family
            condition_label = (
                f"{family} E={E} T={T} D={DISTRACTOR_UPDATES}"
            )

            if f"{family}_T{T}" in unreachable:
                print(f"  [SKIP] {condition_label} — excluded by the "
                      f"reachability probe")
                continue

            query_type = FAMILY_QUERY_TYPES.get(family, "location")

            records, failures = generate_condition(
                family=family,
                entity_count=E,
                target_updates=T,
                distractor_updates=DISTRACTOR_UPDATES,
                num_instances=args.instances,
                experiment_tag=EXPERIMENT_TAG,
                condition_id=condition_id,
                num_containers=num_containers,
                condition_label=condition_label,
                textual_distractor_count=TEXTUAL_DISTRACTORS,
                query_type=query_type,
            )

            all_records.extend(records)
            total_failures += failures
            family_generated += len(records)
            family_failures += failures

            for r in records:
                family_T_actuals.append(r["measured_factors"]["T_actual"])
                family_E_actuals.append(r["measured_factors"]["E_actual"])

        family_stats[family] = {
            "generated": family_generated,
            "failures": family_failures,
            "T_actuals": family_T_actuals,
            "E_actuals": family_E_actuals,
        }

    elapsed = time.perf_counter() - t0

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print()
    print("=" * 75)
    print("RQ1 GENERATION COMPLETE")
    print(f"  Total generated : {len(all_records)}")
    print(f"  Total failures  : {total_failures}")
    print(f"  Excluded cells  : {len(unreachable)}"
          + (f" ({', '.join(unreachable)})" if unreachable else ""))
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 75)

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

    # ----------------------------------------------------------
    # Verification
    # ----------------------------------------------------------

    structural_families = ["split_chain", "merge_chain", "swap_chain",
                           "undo_chain", "undo_redo_chain", "revision"]
    if not verify_generated_records(all_records, EXPERIMENT_TAG,
                                     structural_families=structural_families):
        print("\n[WARN] Post-generation verification failed")
        sys.exit(1)

    # Detailed factor summary
    print_factor_summary(all_records)

    write_jsonl(all_records, output_path)

    # The manifest travels with the dataset so a release can be traced back to
    # the code, the pinned models and the hashes that produced it.
    write_manifest(
        build_manifest(
            all_records,
            experiment_tag=EXPERIMENT_TAG,
            dataset_path=output_path,
            excluded_conditions=unreachable,
        ),
        output_path.parent / "manifest.json",
    )


if __name__ == "__main__":
    main()
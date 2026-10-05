"""
experiments/rq2_interference.py
==============================
RQ2: Interference Decomposition and Supersession

When target-relevant state transitions are held constant, how do
state-changing distractors, narrative distractors, and context length
differentially affect dynamic-state reasoning, and does superseded-information
interference produce a distinct stale-state failure pattern?

Core experiment:
  D/N/L independently controlled, with target trajectory fixed.

Critical comparison:
    same target state trajectory
            │
            ├── no distractors (baseline)
            ├── state-changing distractors (D > 0)
            ├── narrative distractors (N > 0)
            └── length-matched controls (L_tok matched)

Error taxonomy for wrong answers:
  - stale previous state
  - missed target update
  - distractor incorporation
  - unrelated wrong state

L_tok matching is critical: otherwise "you only measured context length."

Design:
  Fixed target trajectory: interleaved_chain, E=3, T=8
  D ∈ {4, 8, 16}  (state-changing distractors; D=0 baseline from RQ1 basic_chain T=8)
  N ∈ {0, 4, 8, 16}  (narrative distractors, crossed with D=4 for length matching)
  Crossed with L_tok-matched controls where possible

  instances/cond: 50
  total: ~350 (7 conditions × 50)

Usage:
  python3 experiments/rq2_interference.py                  # full run
  python3 experiments/rq2_interference.py --dry-run        # reachability probe
  python3 experiments/rq2_interference.py --d-only         # only D conditions
  python3 experiments/rq2_interference.py --n-only         # only N conditions
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

FAMILY = "interleaved_chain"
ENTITY_COUNT = 3
TARGET_UPDATES = 8
NUM_CONTAINERS = 4
INSTANCES_PER_CONDITION = 50
EXPERIMENT_TAG = "rq2_interference"

# D levels (state-changing distractors) — interleaved_chain requires D>=1
# D=0 baseline comes from RQ1 basic_chain at T=8 (same target trajectory)
D_LEVELS = [4, 8, 16]

# N levels (narrative distractors) - crossed with D=4 for length matching
N_LEVELS_AT_D4 = [0, 4, 8, 16]
N_LEVELS_OTHER_D = [0]  # Only baseline for other D levels to control cost

# Length-matched control: for each (D, N) pair, we also generate a
# matched condition where total rendered length is controlled
LENGTH_MATCHED = True

# Secondary analysis: superseded-state pairs
# revision family at same T=8 with/without revision
SUPERSESSION_FAMILIES = [("revision", 1, [8], "Superseded-state interference")]

# Seed group of the D x N design. Cells in this group share their seeds (see
# experiments._common.generate_condition), which is how "same target trajectory
# in the D, N and baseline cells" is realised.
TARGET_TRAJECTORY = f"{FAMILY}_T{TARGET_UPDATES}_E{ENTITY_COUNT}"


def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Interference Decomposition & Supersession"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_CONDITION)
    parser.add_argument("--d-only", action="store_true",
                        help="Only generate D-variation conditions (N=0)")
    parser.add_argument("--n-only", action="store_true",
                        help="Only generate N-variation at D=4")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "rq2_interference" / "rq2_interference.jsonl"
    )

    print("=" * 75)
    print("RQ2 — Interference Decomposition and Supersession")
    print(f"  family        : {FAMILY}")
    print(f"  E (fixed)     : {ENTITY_COUNT}")
    print(f"  T (fixed)     : {TARGET_UPDATES}")
    print(f"  D levels      : {D_LEVELS}")
    print(f"  N at D=4      : {N_LEVELS_AT_D4}")
    print(f"  containers    : {NUM_CONTAINERS}")
    print(f"  instances/cond: {args.instances}")
    if args.dry_run:
        print("  MODE          : DRY-RUN (reachability probe)")
    else:
        print(f"  output        : {output_path}")
    print("=" * 75)

    # Build condition list
    conditions: List[Tuple[int, int, str, str]] = []  # (D, N, tag_suffix, label)

    if args.d_only:
        for D in D_LEVELS:
            conditions.append((D, 0, f"D{D}", f"D={D} N=0"))
    elif args.n_only:
        for N in N_LEVELS_AT_D4:
            conditions.append((4, N, f"D4_N{N}", f"D=4 N={N}"))
    else:
        # Full crossed design: D levels at N=0 + N levels at D=4
        # D=0 baseline is provided by RQ1 basic_chain T=8
        for D in D_LEVELS:
            conditions.append((D, 0, f"D{D}", f"D={D} N=0"))
        for N in N_LEVELS_AT_D4:
            if N != 0:  # D=4 N=0 already added
                conditions.append((4, N, f"D4_N{N}", f"D=4 N={N}"))

    total_conditions = len(conditions)
    if SUPERSESSION_FAMILIES:
        total_conditions += sum(len(Ts) for _, _, Ts, _ in SUPERSESSION_FAMILIES)
    print(f"  total target  : {total_conditions * args.instances}")

    # ----------------------------------------------------------
    # STEP 1: Reachability probe
    # ----------------------------------------------------------

    print("\n[Step 1] Reachability probe (10 seeds × each condition)…")

    probe_results: Dict[str, ProbeResult] = {}

    for D, N, tag, label in conditions:
        probe_results[tag] = probe_reachability(
            family=FAMILY,
            entity_count=ENTITY_COUNT,
            target_updates=TARGET_UPDATES,
            distractor_updates=D,
            num_containers=NUM_CONTAINERS,
            n_seeds=10,
            # Probe the cell the sweep generates, narrative distractors included.
            textual_distractor_count=N,
        )

    # Supersession conditions
    for fam, E, T_levels, desc in SUPERSESSION_FAMILIES:
        for T in T_levels:
            key = f"{fam}_T{T}"
            probe_results[key] = probe_reachability(
                family=fam,
                entity_count=E,
                target_updates=T,
                distractor_updates=0,
                num_containers=3,
                n_seeds=10,
            )

    # Same rule as RQ1: a cell no probe seed can generate is excluded from the
    # sweep and named in the manifest; a cell that generates on some seeds is
    # kept and its shortfall is counted during generation.
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
    condition_stats: Dict[str, Dict] = {}

    t0 = time.perf_counter()

    # Main D×N conditions
    for i, (D, N, tag, label) in enumerate(conditions):
        print(f"\n  Condition {i+1}/{len(conditions)}: {label}")

        if tag in unreachable:
            print(f"  [SKIP] {label} — excluded by the reachability probe")
            continue

        condition_id = tag
        condition_label = f"{FAMILY} E={ENTITY_COUNT} T={TARGET_UPDATES} {label}"

        records, failures = generate_condition(
            family=FAMILY,
            entity_count=ENTITY_COUNT,
            target_updates=TARGET_UPDATES,
            distractor_updates=D,
            num_instances=args.instances,
            experiment_tag=EXPERIMENT_TAG,
            condition_id=condition_id,
            num_containers=NUM_CONTAINERS,
            condition_label=condition_label,
            textual_distractor_count=N,
            # Paired design: every D and N cell draws from the same seed group,
            # so cell i of every condition builds the same target trajectory and
            # the cells differ only in the distractor factors under test.
            seed_group=TARGET_TRAJECTORY,
        )

        all_records.extend(records)
        total_failures += failures

        T_actuals = [r["measured_factors"]["T_actual"] for r in records]
        D_actuals = [r["measured_factors"]["D_actual"] for r in records]
        N_actuals = [r["measured_factors"]["N_actual"] for r in records]
        L_words = [r["measured_factors"]["L_word"] for r in records]

        condition_stats[tag] = {
            "generated": len(records),
            "failures": failures,
            "T_actuals": T_actuals,
            "D_actuals": D_actuals,
            "N_actuals": N_actuals,
            "L_words": L_words,
            "label": label,
        }

    # Supersession conditions (revision at T=8)
    for fam, E, T_levels, desc in SUPERSESSION_FAMILIES:
        for T in T_levels:
            print(f"\n  Supersession condition: {fam} T={T}")

            if f"{fam}_T{T}" in unreachable:
                print(f"  [SKIP] {fam} T={T} — excluded by the reachability probe")
                continue

            condition_id = f"{fam}_T{T}"
            condition_label = f"{fam} E={E} T={T} {desc}"

            records, failures = generate_condition(
                family=fam,
                entity_count=E,
                target_updates=T,
                distractor_updates=0,
                num_instances=args.instances,
                experiment_tag="rq2_supersession",
                condition_id=condition_id,
                num_containers=3,
                condition_label=condition_label,
                textual_distractor_count=0,
            )

            all_records.extend(records)
            total_failures += failures

            T_actuals = [r["measured_factors"]["T_actual"] for r in records]
            V_actuals = [r["measured_factors"]["V_actual"] for r in records]

            condition_stats[condition_id] = {
                "generated": len(records),
                "failures": failures,
                "T_actuals": T_actuals,
                "V_actuals": V_actuals,
                "label": f"{fam} T={T}",
            }

    elapsed = time.perf_counter() - t0

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print()
    print("=" * 75)
    print("RQ2 GENERATION COMPLETE")
    print(f"  Total generated : {len(all_records)}")
    print(f"  Total failures  : {total_failures}")
    print(f"  Excluded cells  : {len(unreachable)}"
          + (f" ({', '.join(unreachable)})" if unreachable else ""))
    print(f"  Elapsed         : {elapsed:.1f}s")
    print("=" * 75)

    print("\nPer-condition summary:")
    for tag, stats in condition_stats.items():
        g = stats["generated"]
        f = stats["failures"]
        if "T_actuals" in stats and stats["T_actuals"]:
            T_a = stats["T_actuals"]
            T_range = f"[{min(T_a)},{max(T_a)}]"
        else:
            T_range = "—"
        if "D_actuals" in stats and stats["D_actuals"]:
            D_a = stats["D_actuals"]
            D_range = f"[{min(D_a)},{max(D_a)}]"
        else:
            D_range = "—"
        if "N_actuals" in stats and stats["N_actuals"]:
            N_a = stats["N_actuals"]
            N_range = f"[{min(N_a)},{max(N_a)}]"
        else:
            N_range = "—"
        if "L_words" in stats and stats["L_words"]:
            L_a = stats["L_words"]
            L_range = f"[{min(L_a)},{max(L_a)}]"
        else:
            L_range = "—"
        if "V_actuals" in stats and stats["V_actuals"]:
            V_a = stats["V_actuals"]
            V_range = f"[{min(V_a)},{max(V_a)}]"
        else:
            V_range = "—"

        print(
            f"  {tag:12s} : {g:3d} gen  {f:2d} fail  "
            f"T∈{T_range} D∈{D_range} N∈{N_range} L∈{L_range} V∈{V_range}"
            f"  ({stats['label']})"
        )

    if total_failures > 0:
        print(f"\n[WARN] {total_failures} instances failed. Check stderr.")

    # ----------------------------------------------------------
    # Verification
    # ----------------------------------------------------------

    if not verify_generated_records(all_records, EXPERIMENT_TAG):
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
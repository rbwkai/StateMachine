"""
experiments/rq2_revision.py
============================
RQ2: Revision Complexity Sweep (E=1, D=0, V≥2)

Design:
    family          : revision
    entity_count    : 1
    distractor_updates: 0
    target_updates  : 4, 8, 12, 16
    instances/cond  : 100
    total           : 400

Control (V=0) is the RQ1 basic_chain data at the same T values.
No duplicate generation needed.

The `revision` trajectory family naturally produces location revisits.
We filter to keep only instances with measured V_actual ≥ MIN_V_ACTUAL (2)
to satisfy the V=2 condition from §5.  Failed or low-V instances are
retried (up to MAX_ATTEMPTS_PER_INSTANCE × instances).

Usage:
    python3 experiments/rq2_revision.py
    python3 experiments/rq2_revision.py --dry-run
    python3 experiments/rq2_revision.py --min-v 1   # relax V threshold
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import random

from generator import build_trajectory, measure_factors
from generator.trajectory_specs import TrajectorySpec
from render.names import NameRegistry
from render.narrative import question_location, render_narrative
from world import replay_trace
from world.operations import Put

from experiments._common import probe_reachability, write_jsonl


# ============================================================
# Experimental design (from §5 of the research plan)
# ============================================================

FAMILY           = "revision"
ENTITY_COUNT     = 1
DISTRACTOR_UPDATES = 0
NUM_CONTAINERS   = 3
INSTANCES_PER_CONDITION = 100
EXPERIMENT_TAG   = "rq2_revision"
BASE_SEED        = 2000
MIN_V_ACTUAL_DEFAULT = 2

T_LEVELS = [4, 8, 12, 16]


# ============================================================
# Single-instance generation with V filter
# ============================================================

def generate_revision_instance(
    seed: int,
    T: int,
    min_v: int,
    max_attempts: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    Generate one revision instance with V_actual >= min_v.

    Because V is an emergent property of the trajectory pattern,
    we retry until we get sufficient revisits.
    """
    rng = random.Random(seed)

    for attempt in range(max_attempts):
        # Advance the rng so each attempt gets a different trajectory.
        sub_rng = random.Random(rng.randint(0, 2**31))

        try:
            spec = TrajectorySpec(
                family=FAMILY,
                entity_count=ENTITY_COUNT,
                num_containers=NUM_CONTAINERS,
                total_updates=T,
                target_updates=T,
                distractor_updates=0,
            )

            t = build_trajectory(sub_rng, spec)

            names = NameRegistry(containers=t.containers, rng=sub_rng)
            sentences, final_state = render_narrative(
                t.ops, t.containers, names
            )

            m = measure_factors(
                t.ops, t.containers, t.target_obj, sentences=sentences
            )

            # Filter on measured V.
            if m.V_actual < min_v:
                continue

            trace, _, _ = replay_trace(t.ops, t.containers)
            step_wise_gold = [
                after.location.get(t.target_obj)
                for _, _, after in trace
            ]

            question = question_location(t.target_obj, final_state, names)
            target_container = final_state.location.get(t.target_obj)
            gold_answer = names.container(target_container) if target_container else None
            initial_placements = sum(1 for op in t.ops if isinstance(op, Put))

            instance_id = (
                f"{EXPERIMENT_TAG}_{FAMILY}_T{T}_V{m.V_actual}_s{seed:04d}"
            )

            canonical_trace = []
            for op in t.ops:
                d = {"op_type": type(op).__name__.upper()}
                for attr in ("obj_id", "dst", "obj_type", "container",
                             "src_container", "dst_container",
                             "container_a", "container_b",
                             "source_obj_id", "new_obj_id"):
                    if hasattr(op, attr):
                        d[attr] = getattr(op, attr)
                canonical_trace.append(d)

            return {
                "instance_id": instance_id,
                "family": FAMILY,
                "experiment": EXPERIMENT_TAG,
                "seed": seed,
                "attempt": attempt,
                "requested_factors": {"E": ENTITY_COUNT, "T": T, "D": 0, "V_min": min_v},
                "measured_factors": m.to_dict(),
                "spec": {
                    "entity_count": ENTITY_COUNT,
                    "target_updates": T,
                    "distractor_updates": 0,
                    "num_containers": NUM_CONTAINERS,
                    "total_updates": T,
                    "initial_placements": initial_placements,
                    "total_transitions": len(t.ops),
                },
                "canonical_trace": canonical_trace,
                "sentences": sentences,
                "context": " ".join(sentences),
                "question": question,
                "query_entity": t.target_obj,
                "gold_container": target_container,
                "gold_answer": gold_answer,
                "step_wise_gold": step_wise_gold,
                "final_state": {
                    "location": final_state.location,
                    "containers": sorted(final_state.containers),
                },
            }

        except Exception:
            continue

    return None


def generate_revision_condition(
    T: int,
    num_instances: int,
    min_v: int,
    base_seed: int,
) -> tuple[List[Dict[str, Any]], int]:
    """Generate num_instances for one revision condition at target depth T."""

    print(f"\n  Generating {num_instances} × [revision E=1 T={T} V≥{min_v}]")
    t0 = time.perf_counter()

    records: List[Dict[str, Any]] = []
    failures = 0

    for i in range(num_instances):
        seed = base_seed + i
        rec = generate_revision_instance(seed=seed, T=T, min_v=min_v)

        if rec is not None:
            records.append(rec)
        else:
            failures += 1

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{num_instances} …", end="\r")

    elapsed = time.perf_counter() - t0
    print(
        f"    Done: {len(records)}/{num_instances} succeeded, "
        f"{failures} failed  ({elapsed:.1f}s)"
    )
    return records, failures


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
        records, failures = generate_revision_condition(
            T=T,
            num_instances=args.instances,
            min_v=args.min_v,
            base_seed=BASE_SEED + T * 1000,
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

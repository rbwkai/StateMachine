"""
experiments/_common.py
=======================
Shared utilities for DWS-Bench experiment generation scripts.

All experiment scripts use this module to:
  - Generate and render one fully-verified benchmark instance
  - Write JSONL output with a standardised schema including measured_factors
  - Report progress and timing
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of where the script is run from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generator import (
    MeasuredFactors,
    build_trajectory,
    measure_factors,
)
from generator.trajectory_specs import TrajectorySpec
from render.names import NameRegistry
from render.narrative import question_location, render_narrative
from world.operations import Put


# ============================================================
# Single-instance generation
# ============================================================

def generate_instance(
    rng: random.Random,
    instance_id: str,
    family: str,
    entity_count: int,
    target_updates: int,
    distractor_updates: int,
    num_containers: int,
    experiment_tag: str,
    seed: int,
    max_attempts: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Generate one fully-verified DWS-Bench instance.

    Returns a JSON-serialisable record on success, or None if
    generation fails after max_attempts.

    The record schema is:

        instance_id        : str
        family             : str
        experiment         : str
        seed               : int
        requested_factors  : {E, T, D}
        measured_factors   : {E_actual, T_actual, D_actual, V_actual, L_actual}
        canonical_trace    : list[{op_type, ...}]
        sentences          : list[str]
        context            : str (joined sentences)
        question           : str
        query_entity       : str
        gold_container     : str   (symbolic container id)
        gold_answer        : str   (rendered container name)
        step_wise_gold     : list[str]  (gold container after each op)
        final_state        : {location, containers}
        spec               : {entity_count, target_updates, ...}
    """

    last_exc: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            total_updates = target_updates + distractor_updates

            spec = TrajectorySpec(
                family=family,
                entity_count=entity_count,
                num_containers=num_containers,
                total_updates=total_updates,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
            )

            t = build_trajectory(rng, spec)

            # Render natural-language narrative.
            names = NameRegistry(containers=t.containers, rng=rng)
            sentences, final_state = render_narrative(
                t.ops, t.containers, names
            )

            # Recompute measured factors now that sentences exist (for L_actual).
            m = measure_factors(
                t.ops,
                t.containers,
                t.target_obj,
                sentences=sentences,
            )

            # Canonical trace (op type + fields).
            canonical_trace = []
            for op in t.ops:
                d = {
                    "op_type": type(op).__name__.upper(),
                }
                if hasattr(op, "obj_id"):
                    d["obj_id"] = op.obj_id
                if hasattr(op, "dst"):
                    d["dst"] = op.dst
                if hasattr(op, "obj_type"):
                    d["obj_type"] = op.obj_type
                if hasattr(op, "container"):
                    d["container"] = op.container
                if hasattr(op, "src_container"):
                    d["src_container"] = op.src_container
                if hasattr(op, "dst_container"):
                    d["dst_container"] = op.dst_container
                if hasattr(op, "container_a"):
                    d["container_a"] = op.container_a
                if hasattr(op, "container_b"):
                    d["container_b"] = op.container_b
                if hasattr(op, "source_obj_id"):
                    d["source_obj_id"] = op.source_obj_id
                if hasattr(op, "new_obj_id"):
                    d["new_obj_id"] = op.new_obj_id
                canonical_trace.append(d)

            # Step-wise gold: target's symbolic container after each op.
            from world import replay_trace
            trace, _, _ = replay_trace(t.ops, t.containers)
            step_wise_gold = [
                after.location.get(t.target_obj)
                for _, _, after in trace
            ]

            # Target question + answer.
            question = question_location(t.target_obj, final_state, names)
            target_container = final_state.location.get(t.target_obj)
            gold_answer = names.container(target_container) if target_container else None

            # Initial placements count.
            initial_placements = sum(
                1 for op in t.ops if isinstance(op, Put)
            )

            record = {
                "instance_id": instance_id,
                "family": family,
                "experiment": experiment_tag,
                "seed": seed,
                "attempt": attempt,

                "requested_factors": {
                    "E": entity_count,
                    "T": target_updates,
                    "D": distractor_updates,
                },

                "measured_factors": m.to_dict(),

                "spec": {
                    "entity_count": entity_count,
                    "target_updates": target_updates,
                    "distractor_updates": distractor_updates,
                    "num_containers": num_containers,
                    "total_updates": total_updates,
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

            return record

        except Exception as exc:
            last_exc = exc
            continue

    print(
        f"  [WARN] {instance_id}: failed after {max_attempts} attempts "
        f"— last error: {last_exc!r}",
        file=sys.stderr,
    )
    return None


# ============================================================
# Batch generation
# ============================================================

def generate_condition(
    family: str,
    entity_count: int,
    target_updates: int,
    distractor_updates: int,
    num_instances: int,
    experiment_tag: str,
    base_seed: int = 0,
    num_containers: int = 3,
    condition_label: str = "",
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate num_instances for one experimental condition.

    Each instance uses seed = base_seed + instance_index so that
    trajectories are structurally diverse.

    Returns (records, failure_count).
    """

    label = condition_label or (
        f"{family} E={entity_count} T={target_updates} D={distractor_updates}"
    )

    print(f"\n  Generating {num_instances} × [{label}]")
    t0 = time.perf_counter()

    records: List[Dict[str, Any]] = []
    failures = 0

    for i in range(num_instances):
        seed = base_seed + i
        rng = random.Random(seed)
        instance_id = (
            f"{experiment_tag}_{family}"
            f"_T{target_updates}_D{distractor_updates}"
            f"_s{seed:04d}"
        )

        rec = generate_instance(
            rng=rng,
            instance_id=instance_id,
            family=family,
            entity_count=entity_count,
            target_updates=target_updates,
            distractor_updates=distractor_updates,
            num_containers=num_containers,
            experiment_tag=experiment_tag,
            seed=seed,
        )

        if rec is not None:
            records.append(rec)
        else:
            failures += 1

        # Progress dot every 10 instances.
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{num_instances} …", end="\r")

    elapsed = time.perf_counter() - t0
    success = len(records)
    print(
        f"    Done: {success}/{num_instances} succeeded, "
        f"{failures} failed  ({elapsed:.1f}s)"
    )

    return records, failures


# ============================================================
# JSONL output
# ============================================================

def write_jsonl(
    records: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Write records to a JSONL file, one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n  Wrote {len(records)} records → {path}")


# ============================================================
# Quick dry-run reachability probe
# ============================================================

def probe_reachability(
    family: str,
    entity_count: int,
    target_updates: int,
    distractor_updates: int,
    num_containers: int = 3,
    n_seeds: int = 10,
) -> bool:
    """
    Test whether a condition is reliably reachable (≥8/10 seeds succeed).

    Returns True if reachable, False otherwise.
    Prints a diagnostic summary.
    """
    successes = 0
    errors = []

    for seed in range(n_seeds):
        rng = random.Random(seed)
        try:
            total_updates = target_updates + distractor_updates
            spec = TrajectorySpec(
                family=family,
                entity_count=entity_count,
                num_containers=num_containers,
                total_updates=total_updates,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
            )
            build_trajectory(rng, spec)
            successes += 1
        except Exception as exc:
            errors.append(f"seed={seed}: {exc!r}")

    rate = successes / n_seeds
    status = "OK" if rate >= 0.8 else "FAIL"

    print(
        f"  [{status}] {family} E={entity_count} T={target_updates} "
        f"D={distractor_updates} — {successes}/{n_seeds} succeeded"
    )

    if errors:
        for e in errors[:3]:
            print(f"    {e}")

    return rate >= 0.8

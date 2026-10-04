"""
experiments/_common.py
=======================
Shared utilities for DWS-Bench experiment generation scripts.

All experiment scripts use this module to:
  - Generate and render one fully-verified benchmark instance
  - Write JSONL output with a standardised schema including measured_factors
  - Report progress and timing
  - Run post-generation validation gates
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of where the script is run from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generator import (
    DEFAULT_MAX_ATTEMPTS,
    TrajectorySpec,
    # probe_reachability() below asks whether a condition can be CONSTRUCTED at
    # all, so it calls the public builder directly; it emits no instance record
    # and therefore has no gate to run.
    build_trajectory,
    generate_instance_with_retry,
    reset_deduplication_registry,
)
from world import GenerationError


# ============================================================
# Single-instance generation
# ============================================================

def generate_instance(
    seed: int,
    instance_id: str,
    family: str,
    entity_count: int,
    target_updates: int,
    distractor_updates: int,
    num_containers: int,
    experiment_tag: str,
    condition_id: str = "",
    min_v: Optional[int] = None,
    intended_v: Optional[int] = None,
    textual_distractor_count: int = 0,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Generate one fully-verified DWS-Bench instance.

    The record and the whole validation gate live in
    ``generator.instance.build_validated_instance``; this function only resolves
    the requested factor levels into a TrajectorySpec and runs the bounded retry
    around it. There is no second copy of the gate here (SPEC §3, requirements.md
    §7).

    Returns a JSON-serialisable record, or raises GenerationError once
    ``max_attempts`` rejections have been spent. The error message carries the
    failure-reason histogram and the first concrete failures, so an unreachable
    condition is diagnosable instead of silently yielding ``None``.
    """

    total_updates = target_updates + distractor_updates

    spec = TrajectorySpec(
        family=family,
        entity_count=entity_count,
        num_containers=num_containers,
        total_updates=total_updates,
        target_updates=target_updates,
        distractor_updates=distractor_updates,
    )

    return generate_instance_with_retry(
        seed=seed,
        spec=spec,
        instance_id=instance_id,
        experiment=experiment_tag,
        condition_id=condition_id,
        textual_distractor_count=textual_distractor_count,
        min_v=min_v,
        intended_v=intended_v,
        max_attempts=max_attempts,
        rng=rng,
    )


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
    condition_id: str = "",
    base_seed: int = 0,
    num_containers: int = 3,
    condition_label: str = "",
    min_v: Optional[int] = None,
    intended_v: Optional[int] = None,
    textual_distractor_count: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate num_instances for one experimental condition.

    Seed calculation:
        seed = int(sha1(f"{experiment_tag}|{condition_id}|{i}")[:8], 16)
    """

    # Reset deduplication registry for each new condition batch
    reset_deduplication_registry()

    if not condition_id:
        condition_id = f"T{target_updates}_D{distractor_updates}_E{entity_count}"

    label = condition_label or (
        f"{family} {condition_id}"
    )

    print(f"\n  Generating {num_instances} × [{label}]")
    t0 = time.perf_counter()

    records: List[Dict[str, Any]] = []
    failures = 0

    for i in range(num_instances):
        seed_key = f"{experiment_tag}|{condition_id}|{i}"
        seed = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)

        instance_id = (
            f"{experiment_tag}_{family}"
            f"_{condition_id}"
            f"_i{i:03d}_s{seed:08x}"
        )

        try:
            rec = generate_instance(
                seed=seed,
                instance_id=instance_id,
                family=family,
                entity_count=entity_count,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
                num_containers=num_containers,
                experiment_tag=experiment_tag,
                condition_id=condition_id,
                min_v=min_v,
                intended_v=intended_v,
                textual_distractor_count=textual_distractor_count,
            )
        except GenerationError as exc:
            # One unreachable instance must not abort the whole sweep: count it
            # and print the gate's reason histogram so the shortfall is visible
            # rather than silently under-counted.
            rec = None
            failures += 1
            print(f"  [WARN] {exc}", file=sys.stderr)

        if rec is not None:
            records.append(rec)

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

    for i in range(n_seeds):
        seed_key = f"probe|{family}|{i}"
        seed = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)
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


# ============================================================
# Post-generation validation gates
# ============================================================

def verify_generated_records(
    records: List[Dict[str, Any]],
    experiment_tag: str,
    structural_families: Optional[List[str]] = None,
) -> bool:
    """
    Run post-generation verification on a set of records.

    Checks:
    1. Requested E/T/D match measured E_actual/T_actual/D_actual exactly
    2. If min_v specified, V_actual >= min_v
    3. Rendered L_word <= L_MAX (600)
    4. Structural causality for structural families
    5. Trace hash uniqueness per condition (>= 80% distinct)
    6. Seed uniqueness across all records
    7. Answer leakage: gold answer not in final distractor sentences
    8. Step-wise gold consistency with final gold

    Returns True if all checks pass, False otherwise.
    """
    if not records:
        print("  [FAIL] No records to verify")
        return False

    all_ok = True
    L_MAX = 600  # from generator/constants.py

    # 1. Seed uniqueness (per condition)
    condition_seeds: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in records:
        exp = r.get("experiment", experiment_tag)
        cond_id = r.get("condition_id", str(r.get("requested_factors")))
        condition_seeds[(exp, cond_id)].append(r["seed"])
    
    seed_dup_failures = 0
    for (exp, cond_id), seeds in condition_seeds.items():
        unique_seeds = set(seeds)
        if len(unique_seeds) != len(seeds):
            dup_count = len(seeds) - len(unique_seeds)
            print(f"  [FAIL] {dup_count} duplicate seeds in {exp}/{cond_id}")
            seed_dup_failures += 1
            all_ok = False
    
    if seed_dup_failures == 0:
        total_seeds = sum(len(s) for s in condition_seeds.values())
        print(f"  [✓] All {total_seeds} seeds unique per condition")

    # 2. Per-condition trace hash distinctness
    condition_hashes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in records:
        exp = r.get("experiment", experiment_tag)
        cond_id = r.get("condition_id", str(r.get("requested_factors")))
        condition_hashes[(exp, cond_id)].append(r.get("trace_hash", ""))

    all_conds_ok = True
    print("\n  Per-condition trace hash distinctness (>= 80% required):")
    for (exp, cond_id), hashes in sorted(condition_hashes.items()):
        n = len(hashes)
        distinct = len(set(hashes))
        ratio = distinct / n if n > 0 else 0
        status = "✓" if ratio >= 0.8 else "FAIL"
        print(f"    [{status}] {exp:20s} / {cond_id:15s}: {distinct:3d}/{n:3d} ({ratio:6.1%})")
        if ratio < 0.8:
            all_conds_ok = False
            all_ok = False

    if all_conds_ok:
        print("  [✓] All conditions pass trace hash distinctness")

    # 3. Factor verification
    print("\n  Factor verification (requested vs measured):")
    factor_mismatches = 0
    for r in records:
        req = r["requested_factors"]
        meas = r["measured_factors"]
        mismatches = []
        if req["E"] != meas["E_actual"]:
            mismatches.append(f"E: req={req['E']} meas={meas['E_actual']}")
        if req["T"] != meas["T_actual"]:
            mismatches.append(f"T: req={req['T']} meas={meas['T_actual']}")
        if req["D"] != meas["D_actual"]:
            mismatches.append(f"D: req={req['D']} meas={meas['D_actual']}")

        if mismatches:
            factor_mismatches += 1
            all_ok = False

    if factor_mismatches == 0:
        print("  [✓] All E/T/D exact matches")
    else:
        print(f"  [FAIL] {factor_mismatches} records with E/T/D mismatch")
        all_ok = False

    # 4. V floor check (for revision families)
    if structural_families is None:
        structural_families = []
    v_floor_failures = 0
    for r in records:
        min_v = r.get("spec", {}).get("min_v") or r.get("spec", {}).get("V_min")
        intended_v = r.get("spec", {}).get("intended_v") or r.get("spec", {}).get("V")
        meas = r["measured_factors"]
        v_actual = meas.get("V_actual", 0)

        if min_v is not None and v_actual < min_v:
            v_floor_failures += 1
            all_ok = False
        if intended_v is not None and v_actual != intended_v:
            v_floor_failures += 1
            all_ok = False

    if v_floor_failures == 0:
        if any(r.get("spec", {}).get("min_v") or r.get("spec", {}).get("V_min") for r in records):
            print("  [✓] All V_actual meet minimum floor")
    else:
        print(f"  [FAIL] {v_floor_failures} records fail V floor")

    # 5. Length verification
    length_failures = 0
    for r in records:
        l_word = r["measured_factors"].get("L_word", -1)
        if l_word > L_MAX:
            length_failures += 1
            all_ok = False

    if length_failures == 0:
        print(f"  [✓] All L_word <= {L_MAX}")
    else:
        print(f"  [FAIL] {length_failures} records exceed L_MAX={L_MAX}")

# 6. Structural causality verification
    if structural_families:
        print("\n  Structural causality (re-verify):")
        struct_failures = 0
        from generator.structural import validate_structural_causality
        from generator.trajectory_specs import TrajectorySpec as TSpec
        from world import replay_trace
        
        for r in records:
            if r["family"] in structural_families:
                try:
                    spec = TSpec(
                        family=r["family"],
                        entity_count=r["spec"]["entity_count"],
                        num_containers=r["spec"]["num_containers"],
                        total_updates=r["spec"]["total_updates"],
                        target_updates=r["spec"]["target_updates"],
                        distractor_updates=r["spec"]["distractor_updates"],
                    )
                    ops = []
                    for op_dict in r["canonical_trace"]:
                        # Don't mutate the record - use get() instead of pop()
                        op_type = op_dict.get("op_type", "").lower()
                        # Filter out op_type from the dict before passing to constructor
                        op_kwargs = {k: v for k, v in op_dict.items() if k != "op_type"}
                        if op_type == "put":
                            from world import Put
                            ops.append(Put(**op_kwargs))
                        elif op_type == "move":
                            from world import Move
                            ops.append(Move(**op_kwargs))
                        elif op_type == "split":
                            from world import Split
                            ops.append(Split(**op_kwargs))
                        elif op_type == "merge":
                            from world import Merge
                            ops.append(Merge(**op_kwargs))
                        elif op_type == "swap":
                            from world import Swap
                            ops.append(Swap(**op_kwargs))
                        elif op_type == "undo":
                            from world import Undo
                            ops.append(Undo())
                        elif op_type == "redo":
                            from world import Redo
                            ops.append(Redo())
                        elif op_type == "remove":
                            from world import Remove
                            ops.append(Remove(**op_kwargs))
                    validate_structural_causality(
                        ops,
                        set(r["final_state"]["containers"]),
                        r["query_entity"],
                        spec,
                    )
                except Exception as exc:
                    struct_failures += 1
                    all_ok = False
                    print(f"    [FAIL] {r['instance_id']}: {exc}")

        if struct_failures == 0:
            print("  [✓] All structural families pass causality re-check")
        else:
            print(f"  [FAIL] {struct_failures} structural records fail causality")

    # 7. Answer leakage check
    print("\n  Answer leakage check (gold not in final 3 distractor sentences):")
    leakage_failures = 0
    import re
    distractor_pattern = re.compile(r"^Someone mentioned that .* have become harder to find lately\.$")
    for r in records:
        sentences = r["sentences"]
        gold = r["gold_answer"]
        num_ops = len(r["canonical_trace"])

        # The generator's leakage check validates RAW distractors (before splicing).
        # We approximate by checking only sentences that match the distractor pattern.
        if gold:
            norm_gold = str(gold).lower().strip()
            # Check all sentences that look like distractors
            distractor_sents = [s for s in sentences if distractor_pattern.match(s)]
            if distractor_sents:
                tail = distractor_sents[-3:] if len(distractor_sents) >= 3 else distractor_sents
                for sent in tail:
                    if norm_gold in sent.lower():
                        leakage_failures += 1
                        all_ok = False
                        break

    if leakage_failures == 0:
        print("  [✓] No answer leakage detected")
    else:
        print(f"  [FAIL] {leakage_failures} records have answer leakage")

    # 8. Step-wise gold consistency (compare rendered answers)
    print("\n  Step-wise gold consistency:")
    gold_mismatches = 0
    for r in records:
        step_gold_answers = r.get("step_wise_gold_answers", [])
        final_gold = r["gold_answer"]
        if step_gold_answers and step_gold_answers[-1] != final_gold:
            gold_mismatches += 1
            all_ok = False

    if gold_mismatches == 0:
        print("  [✓] All step-wise gold final answers match final gold")
    else:
        print(f"  [FAIL] {gold_mismatches} records have step-wise/final gold mismatch")

    # Summary
    print("\n  " + "=" * 60)
    if all_ok:
        print("  [VERIFICATION PASSED] All post-generation checks OK")
    else:
        print("  [VERIFICATION FAILED] One or more checks failed")
    print("  " + "=" * 60)

    return all_ok


def print_factor_summary(records: List[Dict[str, Any]]) -> None:
    """Print detailed factor summary statistics."""
    if not records:
        return

    print("\n  Factor statistics (measured):")
    for factor in ["E_actual", "T_actual", "D_actual", "V_actual", "L_word", "N_actual", "U_actual"]:
        vals = [r["measured_factors"].get(factor, 0) for r in records if factor in r["measured_factors"]]
        if vals:
            print(f"    {factor:12s}: min={min(vals):3d} max={max(vals):3d} mean={sum(vals)/len(vals):.1f} n={len(vals)}")

    print("\n  Requested factor distribution:")
    req_t = Counter(r["requested_factors"].get("T", 0) for r in records)
    req_d = Counter(r["requested_factors"].get("D", 0) for r in records)
    req_e = Counter(r["requested_factors"].get("E", 0) for r in records)
    print(f"    T: {dict(sorted(req_t.items()))}")
    print(f"    D: {dict(sorted(req_d.items()))}")
    print(f"    E: {dict(sorted(req_e.items()))}")

    # Family distribution
    fam_dist = Counter(r["family"] for r in records)
    print("\n  Family distribution:")
    for fam, count in sorted(fam_dist.items()):
        print(f"    {fam:20s}: {count:4d}")

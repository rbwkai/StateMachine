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
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Ensure repo root is importable regardless of where the script is run from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from generator import (
    DEFAULT_MAX_ATTEMPTS,
    TrajectorySpec,
    generate_instance_with_retry,
    reset_deduplication_registry,
)
from world import GenerationError


# How many gate attempts one probe seed may spend. The probe asks "can this cell
# produce a validated instance at all", so a handful of attempts per seed is
# enough: the release generator keeps DEFAULT_MAX_ATTEMPTS.
PROBE_MAX_ATTEMPTS = 10

# Fraction of probe seeds that must pass the gate for a cell to count as
# reliably reachable.
PROBE_SUCCESS_RATE = 0.8


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
    query_type: str = "location",
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
        query_type=query_type,
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
    query_type: str = "location",
    seed_group: str = "",
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Generate num_instances for one experimental condition.

    Seed calculation:
        seed = int(sha1(f"{experiment_tag}|{seed_group or condition_id}|{i}")[:8], 16)

    ``seed_group`` names a paired design: conditions in the same group take
    their seeds from the group key instead of their own condition id, so the
    cells of one group build the same trace at the same instance index. That is
    what makes an RQ cell pair differ only in the factor under test, instead of
    also differing in the random draw (SPEC §5, RQ2 "same target trajectory").
    """

    # Reset deduplication registry for each new condition batch
    reset_deduplication_registry()

    if not condition_id:
        condition_id = f"{family}_T{target_updates}_D{distractor_updates}_E{entity_count}"

    label = condition_label or (
        f"{family} {condition_id}"
    )

    print(f"\n  Generating {num_instances} × [{label}]")
    t0 = time.perf_counter()

    records: List[Dict[str, Any]] = []
    failures = 0

    for i in range(num_instances):
        seed_key = f"{experiment_tag}|{seed_group or condition_id}|{i}"
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
                query_type=query_type,
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
# Dataset manifest
# ============================================================

# Provenance is read from git, never from the environment: a manifest that
# records an unpinned commit is worse than one that records none.
_UNKNOWN_COMMIT = "unknown"


def _git(*args: str) -> str:
    """Run one read-only git command in the repo, returning trimmed stdout.

    Returns an empty string outside a git checkout instead of raising, so the
    manifest is still writable from an unpacked source tree.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _last_commit_touching(*paths: str) -> str:
    """The commit that last touched ``paths``, or ``"unknown"``."""
    commit = _git("log", "-1", "--format=%H", "--", *paths)
    return commit or _UNKNOWN_COMMIT


def _safe_dataset_name(path: Optional[Path]) -> str:
    """A dataset name that cannot leak a home directory or machine name.

    Paths come from ``--output``; only the name, or the path relative to the
    repository root, ever reaches an artifact (AGENTS.md §9).
    """
    if path is None:
        return "dataset"
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _records_digest(records: Sequence[Dict[str, Any]]) -> str:
    """sha1 over the serialised records: the same bytes ``write_jsonl`` writes."""
    payload = "".join(
        json.dumps(rec, ensure_ascii=False) + "\n" for rec in records
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_eval_registry() -> Dict[str, Any]:
    """The eval model registry, loaded without importing the ``eval`` package.

    ``eval/__init__.py`` eagerly imports the inference engine, which imports
    ``torch``. Generation runs are offline and must not need torch (the
    offline-generation test enforces it), and ``eval/models.py`` itself only
    needs ``re`` and ``dataclasses``. The registry stays the single source of
    pinned revisions; it is executed, never copied.
    """
    import importlib.util

    path = _REPO_ROOT / "eval" / "models.py"
    if not path.exists():
        print(f"  [WARN] no model registry at {path.name}; manifest records no "
              f"model revisions")
        return {}
    spec = importlib.util.spec_from_file_location("_dws_eval_models", path)
    if spec is None or spec.loader is None:
        print(f"  [WARN] could not load {path.name}; manifest records no "
              f"model revisions")
        return {}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(getattr(module, "CORE_MODELS", {}))


def build_manifest(
    records: List[Dict[str, Any]],
    experiment_tag: str = "",
    dataset_path: Optional[Path] = None,
    model_keys: Optional[Sequence[str]] = None,
    excluded_conditions: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Provenance for one generated dataset (checklist 11, SPEC §8 OPEN-12).

    Records the commits that produced the data, the pinned model revisions the
    data is meant to be evaluated with, and the per-RQ dataset hashes. Nothing
    here depends on wall-clock time, so re-running a generation writes a
    byte-identical manifest (determinism, AGENTS.md §6 rule 5).
    """
    registry = _load_eval_registry()

    by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_condition[str(record.get("condition_id", ""))].append(record)

    dataset_name = _safe_dataset_name(dataset_path) or experiment_tag

    model_commits: Dict[str, Dict[str, str]] = {}
    for key in model_keys or sorted(registry):
        config = registry.get(key)
        if config is None:
            continue
        model_commits[key] = {
            "hf_model_id": config.hf_model_id,
            "revision": config.revision,
        }

    return {
        "experiment": experiment_tag,
        "dataset": dataset_name,
        "records": len(records),
        "generator_commit": _last_commit_touching("generator", "world", "render"),
        "evaluator_commit": _last_commit_touching("eval"),
        "repo_commit": _git("rev-parse", "HEAD") or _UNKNOWN_COMMIT,
        # A dirty tree means the commit alone does not reproduce the dataset.
        "repo_clean": not _git("status", "--porcelain"),
        "model_commits": model_commits,
        "dataset_hashes": {
            dataset_name: {
                "sha1": _records_digest(records),
                "records": len(records),
            },
            "by_condition": {
                condition: {
                    "sha1": _records_digest(condition_records),
                    "records": len(condition_records),
                }
                for condition, condition_records in sorted(by_condition.items())
            },
        },
        "versions": {
            "generator": records[0].get("generator_version", "") if records else "",
            "renderer": records[0].get("renderer_version", "") if records else "",
            "scoring": records[0].get("scoring_version", "") if records else "",
        },
        "conditions": {
            condition: {
                "family": condition_records[0].get("family", ""),
                "instances": len(condition_records),
            }
            for condition, condition_records in sorted(by_condition.items())
        },
        # A cell the reachability probe found unreachable is reported here, so a
        # reader of the dataset sees the gap instead of inferring a missing cell.
        "excluded_conditions": sorted(excluded_conditions or ()),
    }


def write_manifest(
    manifest: Dict[str, Any],
    path: Path,
) -> None:
    """Write a dataset manifest as JSON next to the dataset it describes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  Wrote manifest → {path}")


# ============================================================
# Quick dry-run reachability probe
# ============================================================

@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one reachability probe.

    Truthy when the cell is reliably reachable, so ``if not probe_reachability(...)``
    keeps reading as it did before this became a value object. ``excluded`` is the
    stricter question a generation sweep needs: a cell where not one seed
    produced a validated instance cannot generate anything at all, while a cell
    that succeeds on some seeds still generates (and simply loses some attempts).
    """

    family: str
    entity_count: int
    target_updates: int
    distractor_updates: int
    attempted: int
    successes: int
    query_type: str = "location"

    @property
    def rate(self) -> float:
        return self.successes / self.attempted if self.attempted else 0.0

    @property
    def reachable(self) -> bool:
        return self.rate >= PROBE_SUCCESS_RATE

    @property
    def excluded(self) -> bool:
        return self.successes == 0

    def __bool__(self) -> bool:
        return self.reachable


def probe_reachability(
    family: str,
    entity_count: int,
    target_updates: int,
    distractor_updates: int,
    num_containers: int = 3,
    n_seeds: int = 10,
    query_type: str = "location",
    textual_distractor_count: int = 0,
) -> ProbeResult:
    """
    Test whether a condition is reachable through the generation gate.

    The probe runs the same validated-instance path the real generation loop
    runs, not ``build_trajectory``. A constructor that honours the requested
    factors can still be rejected by the gate afterwards (``SPEC.md`` OPEN-20:
    that is exactly how ``merge_chain`` fails), and a probe that stops at the
    constructor reports such a cell as reachable and the grid then ships an
    empty condition.

    Returns a ``ProbeResult``; prints a diagnostic summary.
    """
    successes = 0
    errors = []

    for i in range(n_seeds):
        seed_key = f"probe|{family}|T{target_updates}|D{distractor_updates}|{query_type}|{i}"
        seed = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:8], 16)
        # The dedup registry is process-global and rejects a repeat trace_hash;
        # each probe seed must be judged on its own.
        reset_deduplication_registry()
        try:
            generate_instance(
                seed=seed,
                instance_id=f"probe_{family}_T{target_updates}_D{distractor_updates}_i{i}",
                family=family,
                entity_count=entity_count,
                target_updates=target_updates,
                distractor_updates=distractor_updates,
                num_containers=num_containers,
                experiment_tag="probe",
                condition_id=f"{family}_T{target_updates}_D{distractor_updates}",
                textual_distractor_count=textual_distractor_count,
                max_attempts=PROBE_MAX_ATTEMPTS,
                query_type=query_type,
            )
            successes += 1
        except GenerationError as exc:
            errors.append(f"seed={seed}: {exc}")
        except Exception as exc:  # a builder bug must not read as "reachable"
            errors.append(f"seed={seed}: {exc!r}")

    result = ProbeResult(
        family=family,
        entity_count=entity_count,
        target_updates=target_updates,
        distractor_updates=distractor_updates,
        attempted=n_seeds,
        successes=successes,
        query_type=query_type,
    )

    print(
        f"  [{'OK' if result.reachable else 'FAIL'}] {family} E={entity_count} "
        f"T={target_updates} D={distractor_updates} — "
        f"{successes}/{n_seeds} succeeded"
    )

    if errors:
        for e in errors[:3]:
            print(f"    {e}")

    return result


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
        # min_v and intended_v are passed at generation time but not stored in the record's spec.
        # They are only available if the experiment script stores them. For now, skip this check
        # unless the record explicitly has them.
        min_v = r.get("spec", {}).get("min_v")
        intended_v = r.get("spec", {}).get("intended_v")
        meas = r["measured_factors"]
        v_actual = meas.get("V_actual", 0)

        if min_v is not None and v_actual < min_v:
            v_floor_failures += 1
            all_ok = False
        if intended_v is not None and v_actual != intended_v:
            v_floor_failures += 1
            all_ok = False

    if v_floor_failures == 0:
        if any(r.get("spec", {}).get("min_v") is not None or r.get("spec", {}).get("intended_v") is not None for r in records):
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
                        # The re-check must ask the question the record asked.
                        # Omitting it defaulted to "location", so a count cell
                        # was re-validated as a location query and "necessity
                        # failed" fired on every split_chain instance, making
                        # RQ1 exit non-zero after generating.
                        query_type=r["spec"].get("query_type", "location"),
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
        # For count queries (split_chain), step_wise_gold_answers are container names
        # while final_gold is a count number - they won't match. Skip check for count queries.
        query_type = r.get("spec", {}).get("query_type", "location")
        if query_type == "count":
            continue
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

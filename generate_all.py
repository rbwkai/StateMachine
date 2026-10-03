"""
generate_all.py
===============
Master script to generate the complete DWS-Bench benchmark dataset suite:
    - RQ1: Temporal Depth Sweep (300 instances)
    - RQ2: Revision Complexity Sweep (200 instances)
    - RQ3: Distractor / Interference Sweep (200 instances)
    - RQ4: Entity Load Sweep (200 instances)
    - RQ5: Structural Operation Pilot (250 instances)
    - Full Aggregated Benchmark: data/full_benchmark.jsonl (1,150 instances)

Usage:
  python3 generate_all.py --write     # Generate all datasets, verify invariants, and write files
  python3 generate_all.py --dry-run   # Run reachability probes across all experiments
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent

EXPERIMENT_SCRIPTS = [
    ("RQ1 Temporal Depth Sweep", _REPO_ROOT / "experiments" / "rq1_depth.py", _REPO_ROOT / "data" / "rq1_depth" / "rq1_depth.jsonl", 300),
    ("RQ2 Revision Sweep", _REPO_ROOT / "experiments" / "rq2_revision.py", _REPO_ROOT / "data" / "rq2_revision" / "rq2_revision.jsonl", 200),
    ("RQ3 Distractor Sweep", _REPO_ROOT / "experiments" / "rq3_distractor.py", _REPO_ROOT / "data" / "rq3_distractor" / "rq3_distractor.jsonl", 200),
    ("RQ4 Entity Load Sweep", _REPO_ROOT / "experiments" / "rq4_entity_load.py", _REPO_ROOT / "data" / "rq4_entity_load" / "rq4_entity_load.jsonl", 200),
    ("RQ5 Structural Pilot", _REPO_ROOT / "experiments" / "rq5_pilot.py", _REPO_ROOT / "data" / "rq5_pilot" / "rq5_pilot.jsonl", 250),
]


def run_dry_runs() -> bool:
    print("=" * 75)
    print("RUNNING DRY-RUN REACHABILITY PROBES ACROSS ALL EXPERIMENTS")
    print("=" * 75)
    all_ok = True
    for name, script, _, _ in EXPERIMENT_SCRIPTS:
        print(f"\n>>> Probing {name} ({script.name})...")
        res = subprocess.run([sys.executable, str(script), "--dry-run"])
        if res.returncode != 0:
            all_ok = False
            print(f"[FAIL] {name} probe failed!")
    return all_ok


def verify_dataset_invariants(records: List[Dict[str, Any]]) -> bool:
    print("\n" + "=" * 75)
    print("POST-GENERATION DATASET INVARIANT VERIFICATION")
    print("=" * 75)

    all_seeds = [r["seed"] for r in records]
    unique_seeds = set(all_seeds)
    if len(unique_seeds) != len(all_seeds):
        duplicates = [s for s, count in Counter(all_seeds).items() if count > 1]
        print(
            f"[FAIL] Duplicate seeds found! {len(all_seeds) - len(unique_seeds)} duplicates. "
            f"Examples: {duplicates[:5]}"
        )
        return False
    print(f"  [✓] All {len(all_seeds)} seeds are unique across the combined file.")

    # Group records by condition: (experiment, condition_id)
    condition_hashes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in records:
        exp = r.get("experiment", "")
        cond_id = r.get("condition_id", str(r.get("requested_factors")))
        condition_hashes[(exp, cond_id)].append(r.get("trace_hash", ""))

    all_conds_ok = True
    print("\n  Per-condition distinct trace_hash check (required >= 80%):")
    for (exp, cond_id), hashes in sorted(condition_hashes.items()):
        n = len(hashes)
        distinct_count = len(set(hashes))
        ratio = distinct_count / n if n > 0 else 0
        status = "✓" if ratio >= 0.8 else "FAIL"
        print(f"    [{status}] {exp:25s} / {cond_id:15s}: {distinct_count:3d}/{n:3d} distinct traces ({ratio:6.1%})")
        if ratio < 0.8:
            all_conds_ok = False

    if not all_conds_ok:
        print("\n[FAIL] One or more conditions fell below 80% distinct trace_hash threshold!")
        return False

    print("\n  [✓] All conditions passed distinct trace_hash check (>= 80%).")
    return True


def generate_all(instances_scale: float = 1.0) -> None:
    t_start = time.perf_counter()
    print("=" * 75)
    print("GENERATING FULL DWS-BENCH BENCHMARK DATASET")
    print("=" * 75)

    all_generated_files = []

    for name, script, output_file, expected_count in EXPERIMENT_SCRIPTS:
        print(f"\n{'#' * 75}")
        print(f"# Executing: {name}")
        print(f"{'#' * 75}")
        cmd = [sys.executable, str(script)]
        if instances_scale != 1.0:
            target_inst = max(1, int(100 * instances_scale))
            cmd.extend(["--instances", str(target_inst)])

        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] Failed executing {script.name} with return code {res.returncode}")
            sys.exit(res.returncode)

        if output_file.exists():
            all_generated_files.append(output_file)

    # Aggregate into full_benchmark.jsonl
    full_benchmark_path = _REPO_ROOT / "data" / "full_benchmark.jsonl"
    combined_records = []
    for f in all_generated_files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    combined_records.append(json.loads(line))

    # Post-generation invariant check
    if not verify_dataset_invariants(combined_records):
        print("\n[ERROR] Post-generation dataset verification failed! Not saving full_benchmark.jsonl.")
        sys.exit(1)

    with open(full_benchmark_path, "w", encoding="utf-8") as fp:
        for rec in combined_records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 75)
    print("FULL BENCHMARK DATASET GENERATION SUMMARY")
    print("=" * 75)
    print(f"Total time elapsed : {t_total:.2f}s")
    print(f"Aggregated file    : {full_benchmark_path}")
    print(f"Total instances    : {len(combined_records)}")
    print("\nDistribution by Trajectory Family:")
    for fam, count in Counter(r["family"] for r in combined_records).most_common():
        print(f"  - {fam:20s}: {count:4d} instances")

    print("\nDistribution by Experiment Sweep:")
    for exp, count in Counter(r["experiment"] for r in combined_records).most_common():
        print(f"  - {exp:20s}: {count:4d} instances")

    print("\nVerification Checklist:")
    print(f"  [✓] RQ1 Depth Sweep:       {sum(1 for r in combined_records if r['experiment'] == 'rq1_depth')} / 300")
    print(f"  [✓] RQ2 Revision Sweep:    {sum(1 for r in combined_records if r['experiment'] == 'rq2_revision')} / 200")
    print(f"  [✓] RQ3 Distractor Sweep:  {sum(1 for r in combined_records if r['experiment'] in {'rq3_distractor', 'rq3_narrative_distractor'})} / 200")
    print(f"  [✓] RQ4 Entity Load Sweep:  {sum(1 for r in combined_records if r['experiment'] == 'rq4_entity_load')} / 200")
    print(f"  [✓] RQ5 Structural Pilot:  {sum(1 for r in combined_records if r['experiment'] == 'rq5_pilot')} / 250")
    print(f"  [✓] Total Benchmark Suite: {len(combined_records)} / 1150")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Generate complete DWS-Bench benchmark dataset suite.")
    parser.add_argument("--dry-run", action="store_true", help="Run reachability probes only")
    parser.add_argument("--write", action="store_true", help="Write generated dataset files to data/")
    args = parser.parse_args()

    if args.dry_run:
        ok = run_dry_runs()
        sys.exit(0 if ok else 1)
    elif args.write:
        generate_all()
    else:
        print("Please specify --write to generate and write dataset files to data/, or --dry-run for reachability probes.")
        sys.exit(1)


if __name__ == "__main__":
    main()


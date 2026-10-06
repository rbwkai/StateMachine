"""Re-score saved DWS-Bench generations and validate provenance.

Refuses to produce a CSV if any prediction id is unresolvable, if any trace_hash
mismatches (prediction made against a different trace), or if the scoring
version used by the predictions differs from the current eval/scoring.py —
*unless* --allow-scoring-version-drift is passed. This prevents the silent
re-scoring under different rules that previously made historical numbers
incomparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    # Run as a script, sys.path[0] is analysis/, whose statistics.py shadows
    # the stdlib module imported below.
    _SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() != _SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statistics import mean, median  # noqa: E402

from eval.scoring import (
    candidate_answers,
    extract_instance_answer,
    normalize_text,
    score_prediction,
)
from generator.constants import SCORING_VERSION


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_instances(path: Path) -> Dict[str, Dict[str, Any]]:
    return {row["instance_id"]: row for row in load_jsonl(path)}


# Run-condition resolution lives in analysis/run_layout.py (one resolver).
from analysis.run_layout import infer_condition  # noqa: E402,F401



def aggregate(rows: Iterable[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output = []
    for values, bucket in sorted(buckets.items(), key=lambda item: str(item[0])):
        token_values = [row.get("generated_tokens") for row in bucket if row.get("generated_tokens") is not None]
        output.append({
            **dict(zip(keys, values)),
            "instances": len(bucket),
            "semantic_accuracy": mean(bool(row["semantic_correct"]) for row in bucket),
            "strict_accuracy": mean(bool(row["strict_correct"]) for row in bucket),
            "protocol_compliance": mean(bool(row["protocol_compliant"]) for row in bucket),
            "correct_given_protocol": (
                sum(bool(row["strict_correct"]) for row in bucket)
                / sum(bool(row["protocol_compliant"]) for row in bucket)
                if any(row["protocol_compliant"] for row in bucket) else None
            ),
            "invalid_rate": mean(not bool(row["protocol_compliant"]) for row in bucket),
            "truncation_rate": mean(row.get("finish_reason") == "length" for row in bucket),
            "mean_generated_tokens": mean(token_values) if token_values else None,
            "median_generated_tokens": median(token_values) if token_values else None,
        })
    return output


def make_condition_summary(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("model"), row.get("cot"), row.get("max_new_tokens"))
        buckets.setdefault(key, []).append(row)
    output = []
    for (model, cot, max_new_tokens), bucket in sorted(buckets.items(), key=lambda item: (str(item[0][0]), bool(item[0][1]), int(item[0][2] or 0))):
        step_accs = [row["step_accuracy"] for row in bucket if row.get("step_accuracy") is not None]
        output.append({
            "model": model,
            "cot": cot,
            "max_new_tokens": max_new_tokens,
            # Strict, as in the harness, run_eval and post_run_sanity.
            "accuracy": mean(bool(row["strict_correct"]) for row in bucket),
            "semantic_accuracy": mean(bool(row["semantic_correct"]) for row in bucket),
            "stepwise_accuracy": mean(step_accs) if step_accs else None,
            "instances": len(bucket),
        })
    return output


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    if not rows:
        return
    fields = list(fieldnames) if fieldnames is not None else sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_provenance(
    predictions: List[Dict[str, Any]],
    instances: Dict[str, Dict[str, Any]],
    allow_scoring_version_drift: bool,
) -> Tuple[int, int]:
    """
    Validate every prediction against the dataset.

    Returns (fallback_count, mismatched_scoring_version). A prediction with no
    scoring_version counts as drift: the rules that scored it are unknown.
    Raises ValueError on unresolvable instance_id or trace_hash mismatch.
    The drift abort itself is applied by the caller, which owns the flag.
    """
    missing_ids = [
        pred.get("instance_id") for pred in predictions
        if not pred.get("instance_id") or pred.get("instance_id") not in instances
    ]
    if missing_ids:
        raise ValueError(
            f"Prediction instance_id(s) not found in dataset: {missing_ids}. "
            f"Run generation first or point --dataset at the matching file."
        )

    fallback_count = 0
    mismatched_scoring_version = 0
    for pred in predictions:
        iid = pred["instance_id"]
        instance = instances[iid]

        # Trace hash: must match exactly. If prediction lacks it, it predates the
        # provenance upgrade and we cannot verify — treat as failure.
        pred_trace = pred.get("trace_hash")
        inst_trace = instance.get("trace_hash")
        if pred_trace is None or inst_trace is None or pred_trace != inst_trace:
            raise ValueError(
                f"Trace hash mismatch for {iid!r}: "
                f"prediction={pred_trace!r} instance={inst_trace!r}. "
                f"Cannot re-score — the gold answer may be for a different trace."
            )

        if pred.get("scoring_version") != SCORING_VERSION:
            mismatched_scoring_version += 1
        # Fallback counting: if instance lacks step_wise_gold_answers, the
        # candidate builder will fall back to same-family gold answers.
        if not instance.get("step_wise_gold_answers"):
            fallback_count += 1

    return fallback_count, mismatched_scoring_version


def rescore_predictions(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    instances: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Score (path, prediction) pairs under each row's own run condition."""
    dataset_list = list(instances.values())
    rows: List[Dict[str, Any]] = []
    for prediction_path, prediction in loaded:
        condition = infer_condition(prediction_path, prediction)
        row = score_prediction(
            prediction,
            instances[prediction["instance_id"]],
            condition["cot"],
            dataset_context=dataset_list,
        )
        # The condition already prefers row-level fields, so this cannot
        # replace a row's own model with a path-derived one.
        row.update(condition)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score saved predictions and validate provenance."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/full_benchmark.jsonl"))
    parser.add_argument("--predictions-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument(
        "--allow-scoring-version-drift",
        action="store_true",
        help=(
            "Permit predictions whose scoring_version differs from the current "
            "eval/scoring.py (SCORING_VERSION). Use when intentionally re-scoring "
            "old predictions under new rules. Default: abort on version drift."
        ),
    )
    args = parser.parse_args()

    instances = load_instances(args.dataset)
    prediction_files = sorted(args.predictions_root.glob("**/*_predictions.jsonl"))
    if not prediction_files:
        raise FileNotFoundError("No saved prediction JSONL files found.")

    loaded = [
        (prediction_path, prediction)
        for prediction_path in prediction_files
        for prediction in load_jsonl(prediction_path)
    ]

    # Validation gate: must pass before scoring or writing any output.
    fallback_count, mismatched_scoring_version = validate_provenance(
        [prediction for _, prediction in loaded],
        instances,
        allow_scoring_version_drift=args.allow_scoring_version_drift,
    )

    if mismatched_scoring_version and not args.allow_scoring_version_drift:
        sys.stderr.write(
            f"\nABORT: {mismatched_scoring_version} predictions carry "
            f"scoring_version != {SCORING_VERSION} (missing counts as drift). "
            f"Re-scoring them under different rules produces incomparable numbers.\n"
            f"Pass --allow-scoring-version-drift to proceed, or regenerate predictions "
            f"with the current pipeline first.\n"
        )
        sys.exit(1)

    all_rows = rescore_predictions(loaded, instances)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", all_rows)
    write_csv(args.output_dir / "summary.csv", aggregate(all_rows, ["model", "cot", "max_new_tokens", "condition"]))
    write_csv(args.output_dir / "by_family.csv", aggregate(all_rows, ["model", "cot", "max_new_tokens", "family"]))
    write_csv(args.output_dir / "by_depth.csv", aggregate(all_rows, ["model", "cot", "max_new_tokens", "depth"]))

    condition_summary = make_condition_summary(all_rows)
    write_csv(
        args.output_dir / "condition_summary.csv",
        condition_summary,
        fieldnames=[
            "model", "cot", "max_new_tokens", "accuracy", "semantic_accuracy",
            "stepwise_accuracy", "instances",
        ],
    )

    # Provenance sidecar — what produced these numbers?
    provenance = {
        "scoring_version": SCORING_VERSION,
        "dataset_file": str(args.dataset),
        "predictions_root": str(args.predictions_root),
        "prediction_files_count": len(prediction_files),
        "total_predictions_scored": len(all_rows),
        "fallback_candidates_used": fallback_count,
        "mismatched_scoring_version": mismatched_scoring_version,
        "allow_scoring_version_drift": args.allow_scoring_version_drift,
    }
    with (args.output_dir / "provenance.json").open("w", encoding="utf-8") as f:
        json.dump(provenance, f, ensure_ascii=False, indent=2)

    print(f"Scored {len(all_rows)} predictions across {len(prediction_files)} conditions.")
    print(f"  fallback candidates used: {fallback_count}")
    print(f"  scoring version mismatches: {mismatched_scoring_version}")
    print(f"  saved outputs to {args.output_dir} (includes provenance.json)")


if __name__ == "__main__":
    main()
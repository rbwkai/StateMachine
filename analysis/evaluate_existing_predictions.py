"""Re-score saved DWS-Bench generations with the instance-aware Evaluator v2."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.scoring import (
    candidate_answers,
    extract_instance_answer,
    normalize_text,
    score_prediction,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_instances(path: Path) -> Dict[str, Dict[str, Any]]:
    return {row["instance_id"]: row for row in load_jsonl(path)}


def infer_condition(prediction_path: Path) -> Dict[str, Any]:
    condition = prediction_path.parent.parent.name
    match = re.match(r"(?P<model>.+?)_(?P<cot>no_cot|cot)_(?P<tokens>\d+)$", condition)
    if not match:
        raise ValueError(f"Cannot infer model condition from {prediction_path}")
    return {
        "model": match.group("model"),
        "cot": match.group("cot") == "cot",
        "max_new_tokens": int(match.group("tokens")),
        "condition": condition,
    }


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


def make_corrected_condition_summary(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            "accuracy": mean(bool(row["semantic_correct"]) for row in bucket),
            "stepwise_accuracy": mean(step_accs) if step_accs else None,
            "instances": len(bucket),
            "runtime_minutes": None,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/full_benchmark.jsonl"))
    parser.add_argument("--predictions-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluator_v2_results"))
    args = parser.parse_args()

    instances = load_instances(args.dataset)
    dataset_list = list(instances.values())
    prediction_files = sorted(args.predictions_root.glob("**/*_predictions.jsonl"))
    if not prediction_files:
        raise FileNotFoundError("No saved prediction JSONL files found.")

    all_rows: List[Dict[str, Any]] = []
    for prediction_path in prediction_files:
        condition = infer_condition(prediction_path)
        for prediction in load_jsonl(prediction_path):
            instance = instances[prediction["instance_id"]]
            row = score_prediction(prediction, instance, condition["cot"], dataset_context=dataset_list)
            row.update(condition)
            all_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluator_v2_predictions.csv", all_rows)
    write_csv(args.output_dir / "evaluator_v2_summary.csv", aggregate(all_rows, ["model", "cot", "max_new_tokens", "condition"]))
    write_csv(args.output_dir / "evaluator_v2_family.csv", aggregate(all_rows, ["model", "cot", "max_new_tokens", "family"]))
    write_csv(args.output_dir / "evaluator_v2_depth.csv", aggregate(all_rows, ["model", "cot", "max_new_tokens", "depth"]))

    corrected_summary = make_corrected_condition_summary(all_rows)
    write_csv(
        args.output_dir / "corrected_condition_summary.csv",
        corrected_summary,
        fieldnames=["model", "cot", "max_new_tokens", "accuracy", "stepwise_accuracy", "instances", "runtime_minutes"],
    )

    print(f"Scored {len(all_rows)} predictions across {len(prediction_files)} conditions.")
    print(f"Saved Evaluator v2 outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

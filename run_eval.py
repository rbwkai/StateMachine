"""
run_eval.py
===========
Evaluation CLI for running Small Language Models (SLMs) and LLMs on DWS-Bench.

Features:
- 5 Core SLMs & optional model configs
- Supports full benchmark, individual RQ sweeps, or custom JSONL dataset paths
- HuggingFace pipeline with quantization (4-bit, 8-bit, bfloat16, float16)
- Mock inference mode for validation and testing
- Curve fitting, failure onset (L_f) calculation, and error taxonomy analysis
- Exports predictions JSONL, metrics JSON, and a formatted Markdown report.

Usage Examples:
  # Mock dry-run on full benchmark
  python3 run_eval.py --model qwen2.5-0.5b --dataset full --mock

  # Run real SLM on RQ1 depth sweep with GPU and 4-bit quantization
  python3 run_eval.py --model qwen2.5-0.5b --dataset rq1 --device cuda --precision 4bit

  # Run Llama-3.2-3B on full benchmark
  python3 run_eval.py --model llama-3.2-3b --dataset full --device auto --precision bfloat16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.failure_onset import (
    best_fitting_curve,
    compare_curves,
    compute_failure_onset,
)
from analysis.first_error import analyze_first_error
from eval.engine import HuggingFaceEngine, MockInferenceEngine
from eval.eval_harness import extract_answer, format_prompt
from eval.models import CORE_MODELS, OPTIONAL_MODELS, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_SHORTCUTS = {
    "full": _REPO_ROOT / "data" / "full_benchmark.jsonl",
    "rq1": _REPO_ROOT / "data" / "rq1_depth" / "rq1_depth.jsonl",
    "rq2": _REPO_ROOT / "data" / "rq2_revision" / "rq2_revision.jsonl",
    "rq3": _REPO_ROOT / "data" / "rq3_distractor" / "rq3_distractor.jsonl",
    "rq5": _REPO_ROOT / "data" / "rq5_pilot" / "rq5_pilot.jsonl",
}


def resolve_dataset_path(dataset_arg: str) -> Path:
    if dataset_arg in DATASET_SHORTCUTS:
        path = DATASET_SHORTCUTS[dataset_arg]
    else:
        path = Path(dataset_arg)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            f"Please generate the dataset first by running: python3 generate_all.py"
        )
    return path


def resolve_model_config(model_arg: str) -> ModelConfig:
    all_models = {**CORE_MODELS, **OPTIONAL_MODELS}
    if model_arg in all_models:
        return all_models[model_arg]

    # Custom HF model ID
    return ModelConfig(
        name=model_arg.replace("/", "_").lower(),
        hf_model_id=model_arg,
        family="custom",
        parameter_count_b=0.0,
    )


def load_dataset(dataset_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def run_evaluation(
    model_config: ModelConfig,
    dataset_records: List[Dict[str, Any]],
    dataset_name: str,
    output_dir: Path,
    device: str = "auto",
    precision: str = "bfloat16",
    batch_size: int = 8,
    chain_of_thought: bool = False,
    mock: bool = False,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # 1. Initialize engine
    if mock:
        logger.info(f"[MOCK MODE] Initializing Mock Engine for {model_config.name}...")
        engine = MockInferenceEngine(model_name=model_config.name)
    else:
        logger.info(f"Initializing HuggingFace Engine for {model_config.hf_model_id} on {device} ({precision})...")
        engine = HuggingFaceEngine(
            model_config=model_config,
            device=device,
            precision=precision,
            hf_token=hf_token,
        )

    # 2. Run batched inference
    total_samples = len(dataset_records)
    logger.info(f"Starting inference on {total_samples} instances (batch size: {batch_size})...")

    raw_predictions: List[str] = []
    prompts: List[str] = []

    for rec in dataset_records:
        prompt = format_prompt(
            context=rec["context"],
            question=rec["question"],
            system_prompt=model_config.system_prompt,
            chain_of_thought=chain_of_thought,
        )
        prompts.append(prompt)

    for i in range(0, total_samples, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = engine.generate_batch(
            batch_prompts,
            max_new_tokens=model_config.max_new_tokens,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            do_sample=model_config.do_sample,
        )
        raw_predictions.extend(batch_responses)
        if (i + len(batch_prompts)) % (batch_size * 5) == 0 or (i + len(batch_prompts)) == total_samples:
            print(f"  Processed {min(i + len(batch_prompts), total_samples)} / {total_samples} ...", end="\r")

    print()
    elapsed = time.perf_counter() - t0
    logger.info(f"Inference completed in {elapsed:.2f}s ({total_samples / max(elapsed, 0.001):.1f} samples/sec).")

    # 3. Evaluate predictions
    instance_results = []
    correct_count = 0

    family_buckets: Dict[str, List[Dict[str, Any]]] = {}
    rq1_by_t: Dict[int, List[bool]] = {}
    rq2_by_t: Dict[int, List[bool]] = {}
    rq3_by_d: Dict[int, List[bool]] = {}

    for rec, raw_pred, prompt_text in zip(dataset_records, raw_predictions, prompts):
        iid = rec["instance_id"]
        family = rec.get("family", "unknown")
        exp = rec.get("experiment", "unknown")
        gold_answer = str(rec.get("gold_answer", "")).strip()
        gold_container = str(rec.get("gold_container", "")).strip()

        # Extract answer against available containers
        containers = rec.get("final_state", {}).get("containers", [])
        extracted = extract_answer(raw_pred, candidate_containers=containers)

        is_correct = (
            extracted.strip().lower() == gold_answer.strip().lower()
            or extracted.strip().lower() == gold_container.strip().lower()
            or raw_pred.strip().lower() == gold_answer.strip().lower()
        )

        if is_correct:
            correct_count += 1

        result_item = {
            "instance_id": iid,
            "family": family,
            "experiment": exp,
            "requested_factors": rec.get("requested_factors", {}),
            "measured_factors": rec.get("measured_factors", {}),
            "question": rec.get("question"),
            "gold_answer": gold_answer,
            "gold_container": gold_container,
            "raw_prediction": raw_pred,
            "extracted_answer": extracted,
            "is_correct": is_correct,
        }
        instance_results.append(result_item)

        # Buckets for breakdown
        family_buckets.setdefault(family, []).append(result_item)

        # Factor specific sweeps
        req = rec.get("requested_factors", {})
        if exp == "rq1_depth" or family == "basic_chain":
            t_val = req.get("T")
            if t_val is not None:
                rq1_by_t.setdefault(t_val, []).append(is_correct)

        if exp == "rq2_revision" or family == "revision":
            t_val = req.get("T")
            if t_val is not None:
                rq2_by_t.setdefault(t_val, []).append(is_correct)

        if exp == "rq3_distractor" or family == "interleaved_chain":
            d_val = req.get("D")
            if d_val is not None:
                rq3_by_d.setdefault(d_val, []).append(is_correct)

    overall_accuracy = correct_count / total_samples if total_samples > 0 else 0.0

    # Family accuracies
    family_accuracies = {}
    for fam, items in family_buckets.items():
        fam_correct = sum(1 for it in items if it["is_correct"])
        family_accuracies[fam] = {
            "total": len(items),
            "correct": fam_correct,
            "accuracy": fam_correct / len(items),
        }

    # RQ1 Temporal Depth Curve & Failure Onset
    rq1_curve = {}
    for t_val in sorted(rq1_by_t.keys()):
        arr = rq1_by_t[t_val]
        rq1_curve[t_val] = sum(1 for c in arr if c) / len(arr) if arr else 0.0

    l_t_onset = None
    if rq1_curve:
        sorted_t = sorted(rq1_curve.keys())
        sorted_acc = [rq1_curve[t] for t in sorted_t]
        l_t_onset = compute_failure_onset(sorted_t, sorted_acc, tau=0.70)

    # RQ2 Revision Curve
    rq2_curve = {}
    for t_val in sorted(rq2_by_t.keys()):
        arr = rq2_by_t[t_val]
        rq2_curve[t_val] = sum(1 for c in arr if c) / len(arr) if arr else 0.0

    # RQ3 Distractor Curve
    rq3_curve = {}
    for d_val in sorted(rq3_by_d.keys()):
        arr = rq3_by_d[d_val]
        rq3_curve[d_val] = sum(1 for c in arr if c) / len(arr) if arr else 0.0

    l_d_onset = None
    if rq3_curve:
        sorted_d = sorted(rq3_curve.keys())
        sorted_d_acc = [rq3_curve[d] for d in sorted_d]
        l_d_onset = compute_failure_onset(sorted_d, sorted_d_acc, tau=0.70)

    metrics = {
        "model_name": model_config.name,
        "hf_model_id": model_config.hf_model_id,
        "parameter_count_b": model_config.parameter_count_b,
        "dataset": dataset_name,
        "total_instances": total_samples,
        "overall_accuracy": overall_accuracy,
        "elapsed_seconds": elapsed,
        "family_accuracies": family_accuracies,
        "rq1_depth_curve": rq1_curve,
        "rq1_failure_onset_L_T": l_t_onset,
        "rq2_revision_curve": rq2_curve,
        "rq3_distractor_curve": rq3_curve,
        "rq3_failure_onset_L_D": l_d_onset,
    }

    # 4. Save artifacts
    model_output_dir = output_dir / model_config.name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    pred_file = model_output_dir / f"{dataset_name}_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for it in instance_results:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    metrics_file = model_output_dir / f"{dataset_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report_file = model_output_dir / f"{dataset_name}_report.md"
    generate_markdown_report(metrics, report_file)

    logger.info(f"Saved predictions → {pred_file}")
    logger.info(f"Saved metrics     → {metrics_file}")
    logger.info(f"Saved report      → {report_file}")

    return metrics


def generate_markdown_report(metrics: Dict[str, Any], report_path: Path) -> None:
    lines = [
        f"# Evaluation Report: `{metrics['model_name']}` on `{metrics['dataset']}`",
        "",
        f"- **Model ID**: `{metrics['hf_model_id']}` ({metrics['parameter_count_b']}B params)",
        f"- **Total Instances**: {metrics['total_instances']}",
        f"- **Overall Accuracy (A_final)**: **{metrics['overall_accuracy'] * 100:.2f}%**",
        f"- **Runtime**: {metrics['elapsed_seconds']:.2f}s",
        "",
        "## Trajectory Family Accuracies",
        "",
        "| Trajectory Family | Instances | Correct | Accuracy |",
        "|---|---|---|---|",
    ]

    for fam, stats in metrics.get("family_accuracies", {}).items():
        lines.append(f"| `{fam}` | {stats['total']} | {stats['correct']} | **{stats['accuracy'] * 100:.1f}%** |")

    if metrics.get("rq1_depth_curve"):
        lines.extend([
            "",
            "## RQ1 Temporal Depth Degradation Curve",
            f"**Failure Onset (L_T @ τ=0.70)**: `{metrics.get('rq1_failure_onset_L_T')}`",
            "",
            "| Depth (T) | Accuracy |",
            "|---|---|",
        ])
        for t_val, acc in metrics["rq1_depth_curve"].items():
            lines.append(f"| T = {t_val} | {acc * 100:.1f}% |")

    if metrics.get("rq2_revision_curve"):
        lines.extend([
            "",
            "## RQ2 Revision Complexity Curve",
            "",
            "| Depth (T) | Accuracy (V ≥ 2) |",
            "|---|---|",
        ])
        for t_val, acc in metrics["rq2_revision_curve"].items():
            lines.append(f"| T = {t_val} | {acc * 100:.1f}% |")

    if metrics.get("rq3_distractor_curve"):
        lines.extend([
            "",
            "## RQ3 Distractor Interference Curve",
            f"**Failure Onset (L_D @ τ=0.70)**: `{metrics.get('rq3_failure_onset_L_D')}`",
            "",
            "| Distractors (D) | Accuracy |",
            "|---|---|",
        ])
        for d_val, acc in metrics["rq3_distractor_curve"].items():
            lines.append(f"| D = {d_val} | {acc * 100:.1f}% |")

    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run SLM evaluation harness on DWS-Bench.")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-0.5b",
        help=f"Model name or HF ID. Core models: {list(CORE_MODELS.keys())}",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full",
        help="Dataset shortcut ('full', 'rq1', 'rq2', 'rq3', 'rq5') or path to JSONL",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cuda', 'cpu', 'mps'")
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "4bit", "8bit"],
        help="Model precision / quantization",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--cot", action="store_true", help="Use Chain-of-Thought prompting")
    parser.add_argument("--mock", action="store_true", help="Run mock evaluation (dry-run without model weights)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N dataset records")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save evaluation artifacts")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face access token")

    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset)
    dataset_name = dataset_path.stem
    model_cfg = resolve_model_config(args.model)
    records = load_dataset(dataset_path, limit=args.limit)

    output_dir = Path(args.output_dir)

    print("=" * 75)
    print("DWS-BENCH SLM EVALUATION")
    print(f"  Model       : {model_cfg.name} ({model_cfg.hf_model_id})")
    print(f"  Dataset     : {dataset_name} ({len(records)} instances)")
    print(f"  Device      : {args.device}")
    print(f"  Precision   : {args.precision}")
    print(f"  Mode        : {'MOCK' if args.mock else 'LIVE'}")
    print(f"  CoT         : {args.cot}")
    print(f"  Output Dir  : {output_dir}")
    print("=" * 75)

    metrics = run_evaluation(
        model_config=model_cfg,
        dataset_records=records,
        dataset_name=dataset_name,
        output_dir=output_dir,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        chain_of_thought=args.cot,
        mock=args.mock,
        hf_token=args.hf_token,
    )

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Overall Accuracy : {metrics['overall_accuracy'] * 100:.2f}% ({metrics['total_instances']} instances)")
    print(f"Runtime          : {metrics['elapsed_seconds']:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()

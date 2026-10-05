"""
experiments/rq3_scale_reasoning.py
===================================
RQ3: Model Scale and Reasoning Strategy

How do model scale and explicit reasoning prompting alter the accuracy and
failure profile of dynamic-state reasoning, and does the effect of reasoning
prompting depend on mutation structure?

Core experiment:
  0.5B / 3B / 7B  ×  Direct / Structured
  evaluated across representative conditions from RQ1 and RQ2.

Primary quantity:
  Δ_prompt = Acc_structured - Acc_direct
  calculated by error category.

Error-category table:
  | Condition | Direct | Structured |  Δ  | Dominant failure |
  | --------- | -----: | ---------: | -: | ---------------- |
  | MOVE      |        |            |    |                  |
  | SWAP      |        |            |    |                  |
  | UNDO      |        |            |    |                  |
  | high D    |        |            |    |                  |
  | high T    |        |            |    |                  |
  | revision  |        |            |    |                  |

Scale conclusions remain explicitly: within the Qwen2.5 model family.

This script does NOT generate data — it runs evaluation on existing RQ1/RQ2
datasets. Data generation is handled by rq1_mutation_depth.py and
rq2_interference.py.

Usage:
  python3 experiments/rq3_scale_reasoning.py --model qwen2.5-3b --prompt direct
  python3 experiments/rq3_scale_reasoning.py --model qwen2.5-3b --prompt structured
  python3 experiments/rq3_scale_reasoning.py --all-models --all-prompts
  python3 experiments/rq3_scale_reasoning.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import eval modules
from eval.engine import create_engine
from eval.eval_harness import evaluate_predictions, format_prompt
from eval.models import ModelConfig, CORE_MODELS
from eval.scoring import score_prediction


# ============================================================
# Evaluation design
# ============================================================

# Models to evaluate (from eval/models.py)
MODELS = ["qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b"]

# Prompt modes
PROMPT_MODES = ["direct", "structured"]  # zero-shot vs CoT

# Representative conditions from RQ1/RQ2 for evaluation
# (experiment_tag, condition_id, description)
REPRESENTATIVE_CONDITIONS: List[Tuple[str, str, str]] = [
    # RQ1: mutation structure at T=8
    ("rq1_mutation_depth", "T8", "basic_chain T=8 (ordinary forward)"),
    ("rq1_mutation_depth", "T8", "revision T=8 (revisits)"),
    ("rq1_mutation_depth", "T8", "split_chain T=8 (identity mult)"),
    ("rq1_mutation_depth", "T8", "merge_chain T=8 (identity cons)"),
    ("rq1_mutation_depth", "T8", "swap_chain T=8 (bilateral exch)"),
    ("rq1_mutation_depth", "T8", "undo_chain T=8 (rollback)"),
    ("rq1_mutation_depth", "T8", "undo_redo_chain T=8 (edit history)"),
    # RQ1: depth variation (basic_chain)
    ("rq1_mutation_depth", "T2", "basic_chain T=2"),
    ("rq1_mutation_depth", "T4", "basic_chain T=4"),
    ("rq1_mutation_depth", "T8", "basic_chain T=8"),
    ("rq1_mutation_depth", "T16", "basic_chain T=16"),
    # RQ2: interference
    ("rq2_interference", "D4", "D=4 N=0 (state distractors)"),
    ("rq2_interference", "D8", "D=8 N=0"),
    ("rq2_interference", "D16", "D=16 N=0"),
    ("rq2_interference", "D4_N4", "D=4 N=4 (narrative distractors)"),
    ("rq2_interference", "D4_N8", "D=4 N=8"),
    ("rq2_interference", "D4_N16", "D=4 N=16"),
    # RQ2: supersession - revision at T=8
    ("rq2_interference", "revision_T8", "Superseded state (revision T=8)"),
]

DEVICE = "cuda"
PRECISION = "bfloat16"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8


def load_dataset(experiment_tag: str, condition_id: str) -> List[Dict[str, Any]]:
    """Load a specific condition from JSONL files.
    
    Only matches on condition_id to avoid loading wrong data when
    experiment_tags overlap. Tries paths in order of specificity.
    """
    paths = [
        _REPO_ROOT / "data" / experiment_tag / f"{condition_id}.jsonl",  # Most specific
        _REPO_ROOT / "data" / experiment_tag / f"{experiment_tag}.jsonl",  # Experiment-level
        _REPO_ROOT / "data" / f"{experiment_tag}.jsonl",  # Legacy flat structure
    ]

    records = []
    for path in paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        if rec.get("condition_id") == condition_id:
                            records.append(rec)
            if records:
                print(f"  Loaded {len(records)} records from {path}")
                return records

    print(f"  [WARN] No data found for {experiment_tag}/{condition_id}")
    return []


def run_evaluation(
    model_key: str,
    prompt_mode: str,
    conditions: List[Tuple[str, str, str]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run evaluation for one model×prompt combination."""
    model_cfg = CORE_MODELS[model_key]
    engine = create_engine(
        model_key,
        device=DEVICE,
        precision=PRECISION,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    all_results = {}
    all_predictions = []

    for exp_tag, cond_id, desc in conditions:
        print(f"\n  Evaluating {model_key} / {prompt_mode} / {exp_tag}:{cond_id} — {desc}")
        records = load_dataset(exp_tag, cond_id)
        if not records:
            continue

        # Format prompts
        prompts = []
        instance_ids = []
        for rec in records:
            chain_of_thought = (prompt_mode == "structured")
            prompt = format_prompt(
                context=rec["context"],
                question=rec["question"],
                system_prompt=model_cfg.system_prompt,
                chain_of_thought=chain_of_thought,
                prompt_version="v2",
            )
            prompts.append(prompt)
            instance_ids.append(rec["instance_id"])

        # Run inference
        try:
            raw_predictions = engine.generate_batch(prompts, max_new_tokens=MAX_NEW_TOKENS)
        except Exception as e:
            print(f"    [ERROR] Generation failed: {e}")
            continue

        # Convert raw predictions to prediction dicts
        predictions = []
        for iid, raw_pred in zip(instance_ids, raw_predictions):
            predictions.append({
                "instance_id": iid,
                "pred_answer": raw_pred,
                "raw_prediction": raw_pred,
            })

        # Score predictions
        scored = evaluate_predictions(records, predictions, chain_of_thought=(prompt_mode == "structured"))
        all_results[f"{exp_tag}:{cond_id}"] = scored
        all_predictions.extend(predictions)

        # Save predictions
        pred_path = output_dir / f"{model_key}_{prompt_mode}_{exp_tag}_{cond_id}_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return {
        "model": model_key,
        "prompt_mode": prompt_mode,
        "conditions": all_results,
        "all_predictions": all_predictions,
    }


def analyze_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Analyze and print error-category table."""
    print("\n" + "=" * 80)
    print("RQ3 ANALYSIS — Scale × Prompting Error-Category Table")
    print("=" * 80)

    # Build error category breakdown
    for res in results:
        model = res["model"]
        prompt = res["prompt_mode"]
        print(f"\n### {model} / {prompt}")

        for cond_key, scored in res["conditions"].items():
            instance_results = scored.get("instance_results", [])
            if not instance_results:
                continue

            correct = sum(1 for r in instance_results if r.get("is_correct"))
            total = len(instance_results)
            acc = correct / total if total > 0 else 0

            # Error categorization from scored results
            error_cats = Counter()
            for r in instance_results:
                if not r.get("is_correct", False):
                    # Extract error type if available
                    etype = r.get("error_type", "UNKNOWN")
                    error_cats[etype] += 1

            dominant = error_cats.most_common(1)
            dom_str = f"{dominant[0][0]} ({dominant[0][1]})" if dominant else "—"

            print(f"  {cond_key:30s}  Acc={acc:.3f}  n={total}  dom={dom_str}")

    # Compute Δ_prompt per condition
    print("\n### Prompt Effect (Δ = Structured - Direct)")
    direct_results = {r["model"]: {k: v for k, v in r["conditions"].items()} for r in results if r["prompt_mode"] == "direct"}
    structured_results = {r["model"]: {k: v for k, v in r["conditions"].items()} for r in results if r["prompt_mode"] == "structured"}

    for model in MODELS:
        if model not in direct_results or model not in structured_results:
            continue
        print(f"\n  {model}:")
        for cond in set(direct_results[model].keys()) | set(structured_results[model].keys()):
            d_acc = 0
            s_acc = 0
            if cond in direct_results[model]:
                dr = direct_results[model][cond].get("instance_results", [])
                d_acc = sum(1 for r in dr if r.get("is_correct")) / len(dr) if dr else 0
            if cond in structured_results[model]:
                sr = structured_results[model][cond].get("instance_results", [])
                s_acc = sum(1 for r in sr if r.get("is_correct")) / len(sr) if sr else 0
            delta = s_acc - d_acc
            print(f"    {cond:30s}  Direct={d_acc:.3f}  Structured={s_acc:.3f}  Δ={delta:+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="RQ3 Model Scale and Reasoning Strategy Evaluation"
    )
    parser.add_argument("--model", type=str, default=None, choices=MODELS,
                        help="Model to evaluate (default: all)")
    parser.add_argument("--prompt", type=str, default=None, choices=PROMPT_MODES,
                        help="Prompt mode (default: both)")
    parser.add_argument("--all-models", action="store_true",
                        help="Evaluate all models")
    parser.add_argument("--all-prompts", action="store_true",
                        help="Evaluate both prompt modes")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results instead of running evaluation")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    if not args.analyze and not (args.model or args.all_models):
        parser.error("Must specify --model, --all-models, or --analyze")

    output_dir = Path(args.output_dir) if args.output_dir else (
        _REPO_ROOT / "results" / "rq3_scale_reasoning"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_eval = MODELS if args.all_models else ([args.model] if args.model else [])
    prompts_to_eval = PROMPT_MODES if args.all_prompts else ([args.prompt] if args.prompt else [])

    if args.analyze:
        # Load and analyze existing results
        results = []
        for model in MODELS:
            for prompt in PROMPT_MODES:
                result_file = output_dir / f"{model}_{prompt}_results.json"
                if result_file.exists():
                    with open(result_file) as f:
                        results.append(json.load(f))
        if results:
            analyze_results(results, output_dir)
        else:
            print("No existing results found to analyze")
        return

    print("=" * 75)
    print("RQ3 — Model Scale and Reasoning Strategy")
    print(f"  Models        : {models_to_eval}")
    print(f"  Prompts       : {prompts_to_eval}")
    print(f"  Conditions    : {len(REPRESENTATIVE_CONDITIONS)} representative conditions")
    print(f"  Output dir    : {output_dir}")
    print("=" * 75)

    all_results = []

    for model_key in models_to_eval:
        for prompt_mode in prompts_to_eval:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_key} / {prompt_mode}")
            print(f"{'='*60}")

            t0 = time.perf_counter()
            res = run_evaluation(model_key, prompt_mode, REPRESENTATIVE_CONDITIONS, output_dir)
            elapsed = time.perf_counter() - t0

            # Save results
            result_file = output_dir / f"{model_key}_{prompt_mode}_results.json"
            with open(result_file, "w", encoding="utf-8") as f:
                # Convert non-serializable objects
                serializable = {
                    "model": res["model"],
                    "prompt_mode": res["prompt_mode"],
                    "conditions": {
                        k: {
                            "overall_total": v.get("overall_total"),
                            "overall_correct": v.get("overall_correct"),
                            "accuracy": v.get("accuracy"),
                            "instance_results": v.get("instance_results", []),
                        }
                        for k, v in res["conditions"].items()
                    },
                    "elapsed": elapsed,
                }
                json.dump(serializable, f, ensure_ascii=False, indent=2)

            print(f"  Completed in {elapsed:.1f}s → {result_file}")
            all_results.append(res)

    # Analyze
    analyze_results(all_results, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
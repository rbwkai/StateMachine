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
  python3 experiments/rq3_scale_reasoning.py --dry-run         # data/model availability
  python3 experiments/rq3_scale_reasoning.py --model qwen2.5-3b --prompt direct
  python3 experiments/rq3_scale_reasoning.py --model qwen2.5-3b --prompt structured
  python3 experiments/rq3_scale_reasoning.py --all-models --all-prompts
  python3 experiments/rq3_scale_reasoning.py --analyze
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import eval modules
from eval.baselines import query_type_of
from eval.engine import create_engine
from eval.eval_harness import evaluate_predictions, format_prompt
from eval.models import ModelConfig, CORE_MODELS, validate_pinned_revision
from eval.scoring import score_prediction
from experiments._common import write_manifest


# ============================================================
# Evaluation design
# ============================================================

# Models to evaluate (from eval/models.py)
MODELS = ["qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b"]

# Prompt modes
PROMPT_MODES = ["direct", "structured"]  # zero-shot vs CoT

# Representative conditions from RQ1/RQ2 for evaluation.
# (experiment_tag, condition_id, description)
#
# The condition_id must be the id the generation script actually writes into the
# record, and (experiment_tag, condition_id) must be unique per family: RQ1 names
# a cell "{family}_T{T}_D{D}_E{E}" and RQ2 names it by its (D, N) tag. A shared id
# across families made load_dataset() return whichever family happened to match
# first, which is the wrong data for the cell being evaluated.
REPRESENTATIVE_CONDITIONS: List[Tuple[str, str, str]] = [
    # RQ1: mutation structure at T=8
    ("rq1_mutation_depth", "basic_chain_T8_D0_E1", "basic_chain T=8 (ordinary forward)"),
    ("rq1_mutation_depth", "revision_T8_D0_E1", "revision T=8 (revisits)"),
    ("rq1_mutation_depth", "split_chain_T8_D0_E2", "split_chain T=8 (identity mult)"),
    ("rq1_mutation_depth", "merge_chain_T8_D0_E2", "merge_chain T=8 (identity cons)"),
    ("rq1_mutation_depth", "swap_chain_T8_D0_E2", "swap_chain T=8 (bilateral exch)"),
    ("rq1_mutation_depth", "undo_chain_T8_D0_E1", "undo_chain T=8 (rollback)"),
    ("rq1_mutation_depth", "undo_redo_chain_T8_D0_E1", "undo_redo_chain T=8 (edit history)"),
    # RQ1: depth variation (basic_chain)
    ("rq1_mutation_depth", "basic_chain_T2_D0_E1", "basic_chain T=2"),
    ("rq1_mutation_depth", "basic_chain_T4_D0_E1", "basic_chain T=4"),
    ("rq1_mutation_depth", "basic_chain_T16_D0_E1", "basic_chain T=16"),
    # RQ2: interference
    ("rq2_interference", "D4", "D=4 N=0 (state distractors)"),
    ("rq2_interference", "D8", "D=8 N=0"),
    ("rq2_interference", "D16", "D=16 N=0"),
    ("rq2_interference", "D4_N4", "D=4 N=4 (narrative distractors)"),
    ("rq2_interference", "D4_N8", "D=4 N=8"),
    ("rq2_interference", "D4_N16", "D=4 N=16"),
    # RQ2: supersession - revision at T=8 (written into the rq2 dataset by
    # rq2_interference.py under experiment tag rq2_supersession)
    ("rq2_interference", "revision_T8", "Superseded state (revision T=8)"),
]


DEVICE = "cuda"
PRECISION = "bfloat16"
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8

# A filename is built from a model key and a condition id, so it is sanitised
# before it touches the filesystem (AGENTS.md §9).
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    """A filesystem-safe name derived from a model key or a condition id."""
    cleaned = _UNSAFE_NAME.sub("_", value).strip("._")
    return cleaned or "unnamed"


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
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run evaluation for one model×prompt combination.

    ``limit`` caps the records used per condition, so a pilot run can use the
    same code path as a full sweep. It never changes which records come first:
    the JSONL order is the generation order.
    """
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
        if limit is not None:
            records = records[:limit]

        # Format prompts
        prompts = []
        instance_ids = []
        for rec in records:
            chain_of_thought = (prompt_mode == "structured")
            query_type = query_type_of(rec)
            prompt = format_prompt(
                context=rec["context"],
                question=rec["question"],
                system_prompt=model_cfg.system_prompt,
                chain_of_thought=chain_of_thought,
                prompt_version="v2",
                query_type=query_type,
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
        pred_path = output_dir / (
            f"{_safe_name(model_key)}_{_safe_name(prompt_mode)}"
            f"_{_safe_name(exp_tag)}_{_safe_name(cond_id)}_predictions.jsonl"
        )
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
                    etype = (r.get("error_analysis") or {}).get("error_type") or "UNKNOWN"
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


def dry_run_report() -> bool:
    """Report what a real run would evaluate. Writes nothing, loads no model.

    RQ3 has no generator of its own, so its reachability question is "is the
    data for every representative condition present, and are the pinned model
    revisions usable". Returns True when every condition has data.
    """
    print("=" * 75)
    print("RQ3 — Model Scale and Reasoning Strategy")
    print("  MODE          : DRY-RUN (dataset and model availability only)")
    print(f"  Conditions    : {len(REPRESENTATIVE_CONDITIONS)}")
    print(f"  Models        : {MODELS}")
    print(f"  Prompts       : {PROMPT_MODES}")
    print("=" * 75)

    print("\n[Step 1] Representative conditions:")
    missing: List[str] = []
    for exp_tag, cond_id, desc in REPRESENTATIVE_CONDITIONS:
        records = load_dataset(exp_tag, cond_id)
        if records:
            print(f"  [OK] {exp_tag}/{cond_id:26s} {len(records):4d} records — {desc}")
        else:
            print(f"  [MISSING] {exp_tag}/{cond_id} — {desc}")
            missing.append(f"{exp_tag}/{cond_id}")

    print("\n[Step 2] Pinned model revisions:")
    for model_key in MODELS:
        config = CORE_MODELS[model_key]
        try:
            revision = validate_pinned_revision(config)
            print(f"  [OK] {model_key:16s} {config.hf_model_id} @ {revision}")
        except ValueError as exc:
            print(f"  [FAIL] {model_key:16s} {exc}")

    print("\n[DRY-RUN] No model loaded, no results written.")
    if missing:
        print(f"[WARN] {len(missing)} condition(s) have no data: "
              f"{', '.join(missing)}")
    return not missing


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
    parser.add_argument("--dry-run", action="store_true",
                        help="Report dataset and model availability; write nothing")
    parser.add_argument("--instances", type=int, default=None,
                        help="Evaluate at most this many records per condition")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=str,
                        default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_report()
        return

    if not args.analyze and not (args.model or args.all_models):
        parser.error("Must specify --model, --all-models, --dry-run, or --analyze")

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
                result_file = output_dir / f"{_safe_name(model)}_{_safe_name(prompt)}_results.json"
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
    print(f"  Instances/cond: {args.instances if args.instances else 'all'}")
    print(f"  Output dir    : {output_dir}")
    print("=" * 75)

    all_results = []

    for model_key in models_to_eval:
        for prompt_mode in prompts_to_eval:
            print(f"\n{'='*60}")
            print(f"Evaluating {model_key} / {prompt_mode}")
            print(f"{'='*60}")

            t0 = time.perf_counter()
            res = run_evaluation(
                model_key, prompt_mode, REPRESENTATIVE_CONDITIONS, output_dir,
                limit=args.instances,
            )
            elapsed = time.perf_counter() - t0

            # Save results
            result_file = output_dir / (
                f"{_safe_name(model_key)}_{_safe_name(prompt_mode)}_results.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                # Convert non-serializable objects
                serializable = {
                    "model": res["model"],
                    "prompt_mode": res["prompt_mode"],
                    "conditions": {
                        k: {
                            "overall_total": v.get("overall_total"),
                            "overall_correct": v.get("overall_correct"),
                            "accuracy": v.get("overall_accuracy"),
                            "instance_results": v.get("instance_results", []),
                        }
                        for k, v in res["conditions"].items()
                    },
                    "elapsed": elapsed,
                }
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            write_manifest(
                result_file,
                {
                    "experiment": "rq3",
                    "model": model_key,
                    "hf_model_id": CORE_MODELS[model_key].hf_model_id,
                    "revision": CORE_MODELS[model_key].revision,
                    "prompt_mode": prompt_mode,
                    "conditions": sorted(res["conditions"]),
                },
                seed_scheme="evaluation only; decoding from eval.models.ModelConfig, "
                            "inputs are RQ1/RQ2 records (see their manifests)",
            )

            print(f"  Completed in {elapsed:.1f}s → {result_file}")
            all_results.append(res)

    # Analyze
    analyze_results(all_results, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
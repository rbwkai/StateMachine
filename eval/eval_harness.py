"""
eval/eval_harness.py
====================
Standardized Evaluation Harness for DWS-Bench.

Implements §14 of the research plan:
- Standard prompt construction (direct QA vs trajectory tracking)
- Deterministic answer extraction and normalization
- Metric aggregation: final accuracy, trajectory accuracy, first-error breakdown
- Support for offline prediction evaluation and live model pipelines
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from analysis.failure_onset import compute_failure_onset
from analysis.first_error import TrajectoryErrorAnalysis, analyze_first_error


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt(
    context: str,
    question: str,
    system_prompt: Optional[str] = None,
    chain_of_thought: bool = False,
) -> str:
    """Format an instance into a standardized evaluation prompt."""
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f"Instructions: {system_prompt}\n")

    prompt_parts.append(f"Narrative:\n{context}\n")
    prompt_parts.append(f"Question:\n{question}\n")

    if chain_of_thought:
        prompt_parts.append("Let's trace the state changes step by step:")
    else:
        prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


# ============================================================
# Answer Extraction
# ============================================================

def extract_answer(
    raw_response: str,
    candidate_containers: Optional[Sequence[str]] = None,
) -> str:
    """
    Extract and normalize the final predicted answer from a raw model output.
    """
    cleaned = raw_response.strip().lower()

    # Boolean check (e.g. redo-validity)
    if "true" in cleaned and "false" not in cleaned:
        return "True"
    if "false" in cleaned and "true" not in cleaned:
        return "False"

    # Match candidate container names if provided
    if candidate_containers:
        for container in sorted(candidate_containers, key=len, reverse=True):
            if container.lower() in cleaned:
                return container

    # Fallback to looking after 'answer:'
    if "answer:" in cleaned:
        after_answer = cleaned.split("answer:")[-1].strip()
        # Clean trailing punctuation
        return re.sub(r"[^\w\s-]", "", after_answer).strip()

    # First line fallback
    first_line = cleaned.split("\n")[0].strip()
    return re.sub(r"[^\w\s-]", "", first_line).strip()


# ============================================================
# Evaluation Summary Dataclasses
# ============================================================

@dataclass
class InstanceEvalResult:
    instance_id: str
    family: str
    requested_factors: Dict[str, Any]
    gold_answer: str
    pred_answer: str
    is_correct: bool
    gold_trajectory: Optional[List[Any]] = None
    pred_trajectory: Optional[List[Any]] = None
    error_analysis: Optional[Dict[str, Any]] = None


@dataclass
class ConditionEvalSummary:
    condition_key: str
    total_instances: int
    correct_instances: int
    accuracy: float
    error_type_counts: Dict[str, int]


# ============================================================
# Evaluation Runner
# ============================================================

def evaluate_predictions(
    instances: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate a set of predictions against gold instances.

    Each prediction dict should have:
      - "instance_id": matching record in instances
      - "pred_answer": model's predicted final answer
      - "pred_trajectory" (optional): step-by-step state predictions
    """
    pred_map = {p["instance_id"]: p for p in predictions}

    results: List[InstanceEvalResult] = []
    condition_buckets: Dict[str, List[InstanceEvalResult]] = {}

    for inst in instances:
        iid = inst["instance_id"]
        gold_answer = str(inst.get("gold_answer", "")).strip().lower()

        pred_info = pred_map.get(iid, {})
        raw_pred = str(pred_info.get("pred_answer", "")).strip()

        # Extract answer against known container names if available
        containers = inst.get("final_state", {}).get("containers", [])
        extracted_pred = extract_answer(raw_pred, candidate_containers=containers).lower()

        is_correct = (extracted_pred == gold_answer or raw_pred.lower() == gold_answer)

        # Trajectory error analysis if step-wise predictions are present
        gold_traj = inst.get("step_wise_gold")
        pred_traj = pred_info.get("pred_trajectory")
        error_analysis_dict = None

        if gold_traj and pred_traj and len(gold_traj) == len(pred_traj):
            analysis = analyze_first_error(gold_traj, pred_traj)
            error_analysis_dict = analysis.to_dict()

        res = InstanceEvalResult(
            instance_id=iid,
            family=inst.get("family", ""),
            requested_factors=inst.get("requested_factors", {}),
            gold_answer=inst.get("gold_answer", ""),
            pred_answer=raw_pred,
            is_correct=is_correct,
            gold_trajectory=gold_traj,
            pred_trajectory=pred_traj,
            error_analysis=error_analysis_dict,
        )
        results.append(res)

        # Condition grouping: family + T + D
        factors = inst.get("requested_factors", {})
        cond_key = f"{inst.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
        condition_buckets.setdefault(cond_key, []).append(res)

    # Aggregate summaries
    summaries: Dict[str, ConditionEvalSummary] = {}
    for cond_key, bucket in condition_buckets.items():
        total = len(bucket)
        correct = sum(1 for r in bucket if r.is_correct)
        acc = correct / total if total > 0 else 0.0

        error_counts: Dict[str, int] = {}
        for r in bucket:
            if r.error_analysis:
                etype = r.error_analysis.get("error_type", "UNKNOWN")
                error_counts[etype] = error_counts.get(etype, 0) + 1

        summaries[cond_key] = ConditionEvalSummary(
            condition_key=cond_key,
            total_instances=total,
            correct_instances=correct,
            accuracy=acc,
            error_type_counts=error_counts,
        )

    overall_total = len(results)
    overall_correct = sum(1 for r in results if r.is_correct)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    return {
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_acc,
        "condition_summaries": {k: vars(v) for k, v in summaries.items()},
        "instance_results": [vars(r) for r in results],
    }

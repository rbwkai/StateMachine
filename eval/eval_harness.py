"""
eval/eval_harness.py
====================
Standardized Evaluation Harness for DWS-Bench.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from analysis.failure_onset import compute_failure_onset
from analysis.first_error import TrajectoryErrorAnalysis, analyze_first_error
from eval.prompts import build_user_prompt
from eval.scoring import (
    AnswerExtraction,
    candidate_answers,
    extract_instance_answer,
    extract_step_answers as _scoring_extract_step_answers,
    normalize_text,
    score_prediction,
)


# ============================================================
# Prompt formatting
# ============================================================

def format_prompt(
    context: str,
    question: str,
    system_prompt: Optional[str] = None,
    chain_of_thought: bool = False,
    prompt_version: str = "v2",
) -> str:
    """Format an instance into a standardized evaluation prompt."""
    user_prompt = build_user_prompt(
        context=context,
        question=question,
        chain_of_thought=chain_of_thought,
        prompt_version=prompt_version,
    )
    if system_prompt:
        return f"Instructions: {system_prompt}\n\n{user_prompt}"
    return user_prompt


# ============================================================
# Answer Extraction
# ============================================================

def extract_answer(
    raw_response: str,
    candidate_containers: Optional[Sequence[str]] = None,
) -> str:
    """Deprecated: Thin wrapper around eval.scoring for backward compatibility."""
    cands = list(candidate_containers) if candidate_containers else []
    ans_match = re.search(r"(?:final\s+)?answer\s*:\s*(.+?)(?:\n|$)", raw_response, re.IGNORECASE)
    if not cands and ans_match:
        seg = ans_match.group(1).strip().lower()
        if seg in {"true", "false"}:
            cands = ["True", "False"]
    extraction = extract_instance_answer(raw_response, candidates=cands)
    if extraction.answer and (extraction.has_final_answer or ans_match):
        return extraction.answer
    return ""


def normalize_answer(value: Any) -> str:
    """Normalize harmless formatting differences for final-answer comparison."""
    return normalize_text(value)


def extract_step_answers(
    raw_response: str,
    candidate_answers: Sequence[str],
) -> List[Optional[str]]:
    """
    Extract step-wise container predictions from model responses.

    Gold step 1 is the initial Put location (state after op 0).
    """
    return _scoring_extract_step_answers(raw_response, candidate_answers)


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
    raw_response: str = ""
    prompt: str = ""
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
        pred_info = pred_map.get(iid, {})
        raw_pred = str(pred_info.get("pred_answer") or pred_info.get("raw_prediction") or "").strip()

        scored = score_prediction(pred_info, inst, dataset_context=instances)
        is_correct = scored["strict_correct"]

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
            raw_response=str(pred_info.get("raw_response", raw_pred)),
            prompt=str(pred_info.get("prompt", "")),
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

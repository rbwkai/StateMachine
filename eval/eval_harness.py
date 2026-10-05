"""
eval/eval_harness.py
====================
Standardized Evaluation Harness for DWS-Bench.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from analysis.failure_onset import compute_failure_onset
from analysis.first_error import TrajectoryErrorAnalysis, analyze_first_error
from eval.prompts import build_user_prompt
from eval.scoring import (
    AnswerExtraction,
    candidate_answers,
    extract_answer,
    extract_instance_answer,
    extract_step_answers,
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

# ============================================================
# Extraction lives in eval/scoring.py (AGENTS.md §5). Re-exported here
# because harness callers import both from this module; there is no second
# extraction implementation.
# ============================================================


def normalize_answer(value: Any) -> str:
    """Normalize harmless formatting differences for final-answer comparison."""
    return normalize_text(value)


# ============================================================
# Condition keys and trajectory parsing
# ============================================================

def condition_key(instance: Dict[str, Any]) -> str:
    """Bucket key for one condition: family + T + D, plus E and N when set.

    RQ1 forbids pooling two cells that differ only in E or N, so the key has to
    carry them. The minimal cell (E=1, N=0) leaves them implicit, which keeps the
    key equal to the `condition_id` prefix `generate.py` writes
    (`family_T{T}_D{D}_E{E}`) for default cells. The record schema puts N in
    ``measured_factors['N_actual']`` only, so fall back to it rather than pool
    two cells that differ in textual distractors.
    """
    factors = instance.get("requested_factors") or {}
    measured = instance.get("measured_factors") or {}
    key = f"{instance.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
    entity_count = factors.get("E", 1) or 1
    textual_distractors = factors.get("N", measured.get("N_actual", 0)) or 0
    if entity_count != 1:
        key += f"_E{entity_count}"
    if textual_distractors:
        key += f"_N{textual_distractors}"
    return key


def parse_pred_trajectory(
    raw_prediction: str,
    instance: Dict[str, Any],
) -> Optional[List[Optional[str]]]:
    """Step-wise answers parsed out of a chain-of-thought response.

    ``gold_states[0]`` is the initial state, so ``step_wise_gold`` carries one
    more entry than the number of predicted steps; the response is parsed over
    the steps after the initial one. Returns ``None`` when the response has no
    usable Step line, so a markerless reply contributes no trajectory data.
    """
    gold_states = instance.get("step_wise_gold") or []
    if len(gold_states) < 2:
        return None
    parsed = extract_step_answers(
        raw_prediction,
        candidate_answers(instance),
        num_steps=len(gold_states) - 1,
    )
    if not any(step is not None for step in parsed):
        return None
    return parsed


# AGENTS.md §5: eval/scoring.py owns extraction. Re-exported here so harness
# callers keep one import site; there is no second extraction implementation.


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
    
    # Extraction diagnostics (E2: separate extraction vs reasoning failure)
    has_final_answer: bool = False
    extraction_method: str = ""
    protocol_compliant: bool = False
    semantic_correct: bool = False


@dataclass
class ConditionEvalSummary:
    condition_key: str
    total_instances: int
    correct_instances: int
    accuracy: float
    error_type_counts: Dict[str, int]
    
    # Extraction diagnostics (E2)
    format_compliance_rate: float = 0.0
    extraction_rate: float = 0.0
    semantic_accuracy: float = 0.0


# ============================================================
# Evaluation Runner
# ============================================================

def evaluate_predictions(
    instances: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    chain_of_thought: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a set of predictions against gold instances.

    Each prediction dict should have:
      - "instance_id": matching record in instances
      - "pred_answer": model's predicted final answer
      - "pred_trajectory" (optional): step-by-step state predictions
    """
    prediction_ids = [p["instance_id"] for p in predictions]
    duplicate_ids = sorted({
        instance_id
        for instance_id in prediction_ids
        if prediction_ids.count(instance_id) > 1
    })
    if duplicate_ids:
        raise ValueError(
            "duplicate prediction instance_id values: "
            + ", ".join(duplicate_ids)
        )
    pred_map = {p["instance_id"]: p for p in predictions}
    missing_ids = [inst["instance_id"] for inst in instances if inst["instance_id"] not in pred_map]

    results: List[InstanceEvalResult] = []
    condition_buckets: Dict[str, List[InstanceEvalResult]] = {}

    for inst in instances:
        iid = inst["instance_id"]
        pred_info = pred_map.get(iid, {})
        raw_pred = str(pred_info.get("pred_answer") or pred_info.get("raw_prediction") or "").strip()

        scored = score_prediction(
            pred_info,
            inst,
            chain_of_thought=chain_of_thought,
            dataset_context=instances,
        )
        is_correct = scored["strict_correct"]

        # Trajectory error analysis if step-wise predictions are present. A
        # caller-supplied trajectory wins; otherwise parse one out of the
        # response so a chain-of-thought run feeds the taxonomy.
        gold_traj = inst.get("step_wise_gold")
        pred_traj = pred_info.get("pred_trajectory")
        if pred_traj is None:
            pred_traj = parse_pred_trajectory(raw_pred, inst)
        error_analysis_dict = None

        if gold_traj and pred_traj and len(gold_traj) == len(pred_traj) + 1:
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
            # Extraction diagnostics (E2)
            has_final_answer=scored.get("has_final_answer", False),
            extraction_method=scored.get("extraction_method", ""),
            protocol_compliant=scored.get("protocol_compliant", False),
            semantic_correct=scored.get("semantic_correct", False),
        )
        results.append(res)

        # Condition grouping: family + T + D, plus E and N when set (RQ1: no
        # pooling of different E or N into one cell).
        cond_key = condition_key(inst)
        condition_buckets.setdefault(cond_key, []).append(res)

    # Aggregate summaries
    summaries: Dict[str, ConditionEvalSummary] = {}
    for cond_key, bucket in condition_buckets.items():
        total = len(bucket)
        correct = sum(1 for r in bucket if r.is_correct)
        acc = correct / total if total > 0 else 0.0

        # Extraction diagnostics (E2)
        format_compliant = sum(1 for r in bucket if r.has_final_answer)
        extracted = sum(1 for r in bucket if r.extraction_method != "none")
        semantic_correct = sum(1 for r in bucket if r.semantic_correct)
        
        format_compliance_rate = format_compliant / total if total > 0 else 0.0
        extraction_rate = extracted / total if total > 0 else 0.0
        semantic_accuracy = semantic_correct / total if total > 0 else 0.0

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
            format_compliance_rate=format_compliance_rate,
            extraction_rate=extraction_rate,
            semantic_accuracy=semantic_accuracy,
        )

    overall_total = len(results)
    overall_correct = sum(1 for r in results if r.is_correct)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    # Overall extraction diagnostics
    overall_format_compliant = sum(1 for r in results if r.has_final_answer)
    overall_extracted = sum(1 for r in results if r.extraction_method != "none")
    overall_semantic_correct = sum(1 for r in results if r.semantic_correct)
    
    overall_format_compliance = overall_format_compliant / overall_total if overall_total > 0 else 0.0
    overall_extraction_rate = overall_extracted / overall_total if overall_total > 0 else 0.0
    overall_semantic_acc = overall_semantic_correct / overall_total if overall_total > 0 else 0.0

    return {
        "overall_total": overall_total,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_acc,
        # Overall extraction diagnostics (E2)
        "overall_format_compliance_rate": overall_format_compliance,
        "overall_extraction_rate": overall_extraction_rate,
        "overall_semantic_accuracy": overall_semantic_acc,
        "prediction_count": len(predictions),
        "missing_predictions": len(missing_ids),
        "missing_prediction_ids": missing_ids,
        "condition_summaries": {k: vars(v) for k, v in summaries.items()},
        "instance_results": [vars(r) for r in results],
    }

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
    count_step_lines,
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
    query_type: str = "location",
) -> str:
    """Format an instance into a standardized evaluation prompt."""
    user_prompt = build_user_prompt(
        context=context,
        question=question,
        chain_of_thought=chain_of_thought,
        prompt_version=prompt_version,
        query_type=query_type,
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
    """Bucket key for one condition: ``{family}_T{T}_D{D}_E{E}_N{N}``.

    Single owner of the per-condition key for the eval layer (AGENTS.md §5):
    ``eval.baselines.condition_key_of`` re-exports this function, so harness
    summaries, baseline summaries and ``eval.post_run_sanity`` bucket a record
    identically. RQ1 forbids pooling two cells that differ only in E or N, so
    every key carries both, including the minimal cell (``_E1_N0``); a key that
    left them implicit only coincided with the baseline key when E != 1 and
    N != 0. This is not the experiments' ``condition_id``, which is a separate
    label written by ``generate.py``.

    The record schema puts N in ``measured_factors['N_actual']`` only, so fall
    back to it rather than pool two cells that differ in textual distractors.
    """
    factors = instance.get("requested_factors") or {}
    measured = instance.get("measured_factors") or {}
    entity_count = factors.get("E", 1) or 1
    textual_distractors = factors.get("N", measured.get("N_actual", 0)) or 0
    return (
        f"{instance.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
        f"_E{entity_count}_N{textual_distractors}"
    )


def parse_pred_trajectory(
    raw_prediction: str,
    instance: Dict[str, Any],
) -> Optional[List[Optional[str]]]:
    """Step-wise answers parsed out of a chain-of-thought response.

    The prompt asks for one ``Step k`` line per narrated sentence, distractor
    sentences included, so Step k pairs with ``step_wise_gold_answers[k-1]``:
    display names for a location cell, counts for a count cell, one entry per
    sentence. ``step_wise_gold`` is not used here: it holds container ids (and
    locations even for count cells) with an extra initial state, so comparing a
    parsed response against it marks a perfect response as all-wrong.

    The query type and gold answer are passed to the parser so a count cell is
    read in the numeric answer space. Returns ``None`` when the response has
    no usable Step line, so a markerless reply contributes no trajectory data.
    """
    gold_answers = instance.get("step_wise_gold_answers") or []
    if not gold_answers:
        return None
    gold_answer = instance.get("gold_answer")
    parsed = extract_step_answers(
        raw_prediction,
        candidate_answers(instance),
        num_steps=len(gold_answers),
        gold_answer=None if gold_answer is None else str(gold_answer),
        instance=instance,
    )
    if not any(step is not None for step in parsed):
        return None
    return parsed


def _step_alignment(n_steps: int, expected: int) -> str:
    if n_steps == expected:
        return "exact"
    return "under_stepped" if n_steps < expected else "over_stepped"


# analysis.first_error.analyze_first_error expects gold with a leading initial
# state (it compares pred[i] against gold[i + 1] and never reads gold[0]).
# step_wise_gold_answers has no initial entry, so a placeholder is prepended to
# align Step k with gold answer k. It is never compared and never reported.
_INITIAL_STATE_SENTINEL: Optional[str] = None


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

    # "exact" | "under_stepped" | "over_stepped", or None without a trajectory.
    step_alignment: Optional[str] = None
    # Engine stop reason ("length" = hit the token limit), when the prediction
    # row carries one; post_run_sanity derives truncation from it.
    finish_reason: Optional[str] = None


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
      - "finish_reason" (optional): engine stop reason, carried through
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

        # Trajectory error analysis if step-wise predictions are present.
        #
        # Two input contracts:
        # - caller-supplied ``pred_trajectory``: the state space of
        #   ``step_wise_gold`` (container ids, initial state at index 0, so a
        #   full trajectory has len(gold) - 1 steps);
        # - otherwise the response is parsed: one answer per narrated sentence
        #   in the space of ``step_wise_gold_answers``, compared with no offset.
        # A missing step stays None, which first_error reports as MISSING (not
        # as wrong); extra steps are dropped. step_alignment records which.
        # gold_trajectory reports the list the prediction was compared with
        # (step_wise_gold unless a response trajectory was parsed).
        pred_traj = pred_info.get("pred_trajectory")
        gold_traj: Optional[List[Any]] = inst.get("step_wise_gold")
        error_analysis_dict = None
        step_alignment: Optional[str] = None

        if pred_traj is not None:
            if gold_traj and pred_traj:
                expected = len(gold_traj) - 1
                pred_traj = list(pred_traj)
                step_alignment = _step_alignment(len(pred_traj), expected)
                if len(pred_traj) < expected:
                    pred_traj = pred_traj + [None] * (expected - len(pred_traj))
                else:
                    pred_traj = pred_traj[:expected]
                analysis = analyze_first_error(gold_traj, pred_traj)
                error_analysis_dict = analysis.to_dict()
        else:
            pred_traj = parse_pred_trajectory(raw_pred, inst)
            if pred_traj is not None:
                gold_traj = list(inst.get("step_wise_gold_answers") or [])
                # Alignment is judged on the raw Step-line count: the parser
                # always returns exactly len(gold_traj) entries.
                step_alignment = _step_alignment(
                    count_step_lines(raw_pred), len(gold_traj)
                )
                gold_cmp = [_INITIAL_STATE_SENTINEL] + [
                    normalize_text(g) for g in gold_traj
                ]
                pred_cmp = [
                    None if p is None else normalize_text(p) for p in pred_traj
                ]
                analysis = analyze_first_error(gold_cmp, pred_cmp)
                error_analysis_dict = analysis.to_dict()
                # Report the gold/pred finals as the record spells them.
                error_analysis_dict["gold_final"] = gold_traj[-1]
                error_analysis_dict["pred_final"] = pred_traj[-1]
        if error_analysis_dict is not None:
            error_analysis_dict["step_alignment"] = step_alignment

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
            step_alignment=step_alignment,
            finish_reason=pred_info.get("finish_reason"),
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

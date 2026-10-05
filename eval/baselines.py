"""
eval/baselines.py
=================
Baseline predictors for DWS-Bench evaluation.

Implements the two standard baselines from the literature (Rezaee et al., 2025):
1. Stateless Baseline: predicts the initial location of the target entity
2. Most-Frequent-Class (MFC) Baseline: predicts the most common answer in the condition

These baselines establish the floor for tracking performance and detect
shortcuts where models answer correctly without tracking.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from eval.scoring import (
    candidate_answers,
    extract_instance_answer,
    normalize_text,
)


@dataclass
class BaselineResult:
    """Result of a baseline prediction."""
    instance_id: str
    gold_answer: str
    pred_answer: str
    is_correct: bool
    baseline_type: str


def query_type_of(instance: Dict[str, Any]) -> str:
    """Resolve the query type of an instance record.

    ``TrajectorySpec.query_type`` is not serialised into the record's ``spec``
    block (see ``generator.instance.build_validated_instance``), so a plain
    ``spec['query_type']`` lookup silently returns "location" for every count
    instance. Prefer an explicit key when present, otherwise read it off the
    rendered question, which is the only place the answer contract survives in
    the record: ``render.narrative.question_count`` always starts with "How many".
    """
    spec = instance.get("spec") or {}
    declared = instance.get("query_type") or spec.get("query_type")
    if declared in {"location", "count", "redo_validity"}:
        return str(declared)
    return "count" if str(instance.get("question", "")).startswith("How many") else "location"


def compute_stateless_baseline(instances: Sequence[Dict[str, Any]]) -> List[BaselineResult]:
    """
    Stateless baseline: answer based on initial state only.

    For each instance, extract the target's initial container from the
    target's Put operation in the canonical trace.
    For count queries, predict the initial count.
    """
    results: List[BaselineResult] = []
    
    for inst in instances:
        iid = inst["instance_id"]
        gold = str(inst.get("gold_answer", "")).strip()
        
        # Determine query type from instance
        family = inst.get("family", "")
        query_type = query_type_of(inst)
        
        # Find target's initial location from canonical trace
        trace = inst.get("canonical_trace", [])
        target_obj = inst.get("query_entity", "")
        initial_container = None
        target_type = None
        
        for op_dict in trace:
            if op_dict.get("op_type") == "PUT":
                op_obj_id = op_dict.get("obj_id")
                if op_obj_id == target_obj:
                    initial_container = op_dict.get("container")
                    target_type = op_dict.get("obj_type")
                    break
        
        if query_type == "count":
            # For count queries, predict initial count of target_type in the query container
            # The query container is the merge destination for split_chain
            if family == "split_chain":
                # Find the merge destination
                merge_dst = None
                for op_dict in trace:
                    if op_dict.get("op_type") == "MERGE":
                        merge_dst = op_dict.get("dst_container")
                        break
                if merge_dst and target_type:
                    # Count initial objects of target_type in merge_dst
                    # Initially, only the target exists, and it's in its initial container
                    # which is not merge_dst (since merge_dst is different from start)
                    pred = "0"  # Initially 0 in merge destination
                else:
                    pred = "0"
            else:
                pred = "0"
        else:
            # Location query: predict target's initial container
            if initial_container is None:
                pred = ""
            else:
                # Use display name from final_state container_names to match candidates
                final_state = inst.get("final_state", {})
                container_names = final_state.get("container_names") or final_state.get("container_display_names")
                if isinstance(container_names, dict):
                    pred = container_names.get(initial_container, initial_container)
                else:
                    pred = initial_container
        
        # Use scoring extraction for fair comparison
        cands = candidate_answers(inst, dataset_context=list(instances))
        # Format as if model answered with "Final Answer: <pred>"
        formatted_pred = f"Final Answer: {pred}" if pred else ""
        extraction = extract_instance_answer(formatted_pred, cands, chain_of_thought=False, gold_answer=gold)
        extracted = extraction.answer
        is_correct = extraction.strict_correct
        
        results.append(BaselineResult(
            instance_id=iid,
            gold_answer=gold,
            pred_answer=extracted,
            is_correct=is_correct,
            baseline_type="stateless",
        ))
    
    return results


def compute_mfc_baseline(instances: Sequence[Dict[str, Any]]) -> List[BaselineResult]:
    """
    Most-Frequent-Class (MFC) baseline: always predict the most common gold answer.

    Computed per condition (family + T + D) to avoid leakage across conditions.
    Uses normalized gold answers (container IDs) to avoid per-instance display name randomization.
    """
    # Group by condition
    from collections import defaultdict
    cond_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for inst in instances:
        factors = inst.get("requested_factors", {})
        cond_key = f"{inst.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
        cond_instances[cond_key].append(inst)
    
    # Find MFC per condition using normalized gold answers
    cond_mfc: Dict[str, str] = {}
    for cond_key, cond_insts in cond_instances.items():
        answers = [normalize_text(inst.get("gold_answer", "")) for inst in cond_insts if inst.get("gold_answer")]
        if answers:
            cond_mfc[cond_key] = Counter(answers).most_common(1)[0][0]
        else:
            cond_mfc[cond_key] = ""
    
    # Apply MFC prediction
    results: List[BaselineResult] = []
    for inst in instances:
        iid = inst["instance_id"]
        gold = str(inst.get("gold_answer", "")).strip()
        
        factors = inst.get("requested_factors", {})
        cond_key = f"{inst.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
        pred = cond_mfc.get(cond_key, "")
        
        cands = candidate_answers(inst, dataset_context=list(instances))
        formatted_pred = f"Final Answer: {pred}" if pred else ""
        extraction = extract_instance_answer(formatted_pred, cands, chain_of_thought=False, gold_answer=gold)
        extracted = extraction.answer
        is_correct = extraction.strict_correct
        
        results.append(BaselineResult(
            instance_id=iid,
            gold_answer=gold,
            pred_answer=extracted,
            is_correct=is_correct,
            baseline_type="mfc",
        ))
    
    return results


def summarize_baselines(
    results: List[BaselineResult],
    condition_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize baseline results."""
    if condition_key:
        filtered = [r for r in results if r.instance_id.startswith(condition_key)]
    else:
        filtered = results
    
    if not filtered:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    
    total = len(filtered)
    correct = sum(1 for r in filtered if r.is_correct)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
    }


def run_all_baselines(instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Run both baselines and return combined summary."""
    stateless_results = compute_stateless_baseline(instances)
    mfc_results = compute_mfc_baseline(instances)
    
    return {
        "stateless": {
            "overall": summarize_baselines(stateless_results),
            "per_condition": _summarize_per_condition(stateless_results, instances),
        },
        "mfc": {
            "overall": summarize_baselines(mfc_results),
            "per_condition": _summarize_per_condition(mfc_results, instances),
        },
    }


def _summarize_per_condition(
    results: List[BaselineResult],
    instances: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Summarize baseline accuracy per condition."""
    from collections import defaultdict
    
    cond_results: Dict[str, List[BaselineResult]] = defaultdict(list)
    for inst in instances:
        iid = inst["instance_id"]
        factors = inst.get("requested_factors", {})
        cond_key = f"{inst.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
        # Find matching result
        for r in results:
            if r.instance_id == iid:
                cond_results[cond_key].append(r)
                break
    
    return {
        cond: {
            "total": len(rs),
            "correct": sum(1 for r in rs if r.is_correct),
            "accuracy": sum(1 for r in rs if r.is_correct) / len(rs) if rs else 0.0,
        }
        for cond, rs in cond_results.items()
    }
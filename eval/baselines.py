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

from eval.scoring import candidate_answers, extract_instance_answer


@dataclass
class BaselineResult:
    """Result of a baseline prediction."""
    instance_id: str
    gold_answer: str
    pred_answer: str
    is_correct: bool
    baseline_type: str


def compute_stateless_baseline(instances: Sequence[Dict[str, Any]]) -> List[BaselineResult]:
    """
    Stateless baseline: answer based on initial state only.

    For each instance, extract the target's initial container from the
    first Put operation in the canonical trace.
    """
    results: List[BaselineResult] = []
    
    for inst in instances:
        iid = inst["instance_id"]
        gold = str(inst.get("gold_answer", "")).strip()
        
        # Find initial location from canonical trace
        trace = inst.get("canonical_trace", [])
        initial_container = None
        for op_dict in trace:
            if op_dict.get("op_type") == "PUT":
                # This is a Put operation; check if it's the target
                # We need to match by obj_id or assume first Put is target
                initial_container = op_dict.get("container")
                break
        
        if initial_container is None:
            pred = ""
        else:
            # Get display name from final_state container_names if available
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
    """
    # Group by condition
    from collections import defaultdict
    cond_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for inst in instances:
        factors = inst.get("requested_factors", {})
        cond_key = f"{inst.get('family')}_T{factors.get('T')}_D{factors.get('D', 0)}"
        cond_instances[cond_key].append(inst)
    
    # Find MFC per condition
    cond_mfc: Dict[str, str] = {}
    for cond_key, cond_insts in cond_instances.items():
        answers = [str(inst.get("gold_answer", "")).strip() for inst in cond_insts if inst.get("gold_answer")]
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
"""
analysis/solubility.py
======================
Solubility audit for DWS-Bench instances.

Checks whether rendered instances are answerable from surface form alone,
independent of symbolic correctness. A rendering error can make an instance
unanswerable or ambiguous even if the symbolic trace is correct.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from eval.scoring import candidate_answers, extract_instance_answer


@dataclass
class SolubilityResult:
    """Result of solubility check for one instance."""
    instance_id: str
    is_soluble: bool
    ambiguity_score: float  # 0.0 = unambiguous, 1.0 = maximally ambiguous
    issues: List[str]
    gold_answer: str
    candidates: List[str]


def check_answer_uniqueness(
    instance: Dict[str, Any],
    threshold: float = 0.8,
) -> SolubilityResult:
    """
    Check if the gold answer is uniquely identifiable from the narrative.
    
    Uses the candidate answer set and checks if gold answer is the only
    plausible answer given the narrative context.
    """
    iid = instance["instance_id"]
    gold = instance.get("gold_answer", "")
    candidates = candidate_answers(instance)
    sentences = instance.get("sentences", [])
    context = instance.get("context", "")
    
    issues = []
    
    # Issue 1: Gold answer not meaningfully distinguishable from candidates
    # (candidate_answers() always includes gold_answer, so check if it's the ONLY candidate)
    if len(candidates) == 1 and candidates[0] == gold:
        issues.append(f"Gold answer '{gold}' is the only candidate (no alternatives)")
    
    # Issue 2: Too many candidates (ambiguity)
    if len(candidates) > 5:
        issues.append(f"High candidate count ({len(candidates)}) increases ambiguity")
    
    # Issue 3: Narrative doesn't mention gold answer container
    # Only flag if gold is in candidates but not mentioned in context
    if gold and gold in candidates and gold.lower() not in context.lower():
        issues.append(f"Gold answer '{gold}' not explicitly mentioned in context")
    
    # Issue 4: Multiple containers mentioned equally (no clear winner)
    # Only flag if gold is NOT the most mentioned
    container_mentions: Dict[str, int] = {}
    for c in candidates:
        count = context.lower().count(c.lower())
        container_mentions[c] = count
    
    max_mentions = max(container_mentions.values()) if container_mentions else 0
    top_containers = [c for c, cnt in container_mentions.items() if cnt == max_mentions]
    if len(top_containers) > 1 and max_mentions > 0 and gold not in top_containers:
        issues.append(f"Multiple containers tied for mentions: {top_containers}; gold '{gold}' not among them")
    
    # Ambiguity score: based on candidate count and mention distribution
    if not candidates:
        ambiguity = 1.0
    elif len(candidates) == 1:
        ambiguity = 0.0
    else:
        # Normalized entropy-like score
        total_mentions = sum(container_mentions.values())
        if total_mentions == 0:
            ambiguity = 0.5  # No mentions at all - moderate ambiguity
        else:
            # Score based on how concentrated mentions are on gold answer
            gold_mentions = container_mentions.get(gold, 0)
            if gold_mentions == max_mentions and len(top_containers) == 1:
                ambiguity = 0.0
            else:
                ambiguity = 1.0 - (gold_mentions / total_mentions) if total_mentions > 0 else 0.5
    
    is_soluble = len(issues) == 0 and ambiguity < threshold
    
    return SolubilityResult(
        instance_id=iid,
        is_soluble=is_soluble,
        ambiguity_score=ambiguity,
        issues=issues,
        gold_answer=gold,
        candidates=candidates,
    )


def check_solubility_llm(
    instance: Dict[str, Any],
    judge_fn: Optional[callable] = None,
) -> SolubilityResult:
    """
    LLM-based solubility check (requires external judge function).
    
    The judge_fn should take (context, question, gold_answer, candidates)
    and return a dict with keys: is_soluble, confidence, reasoning.
    """
    iid = instance["instance_id"]
    gold = instance.get("gold_answer", "")
    candidates = candidate_answers(instance)
    context = instance.get("context", "")
    question = instance.get("question", "")
    
    if judge_fn is None:
        # Fallback to heuristic
        return check_answer_uniqueness(instance)
    
    try:
        result = judge_fn(context, question, gold, candidates)
        return SolubilityResult(
            instance_id=iid,
            is_soluble=result.get("is_soluble", False),
            ambiguity_score=1.0 - result.get("confidence", 0.0),
            issues=[] if result.get("is_soluble") else [result.get("reasoning", "LLM judged insoluble")],
            gold_answer=gold,
            candidates=candidates,
        )
    except Exception as e:
        return SolubilityResult(
            instance_id=iid,
            is_soluble=False,
            ambiguity_score=1.0,
            issues=[f"LLM judge error: {e}"],
            gold_answer=gold,
            candidates=candidates,
        )


def run_solubility_audit(
    instances: Sequence[Dict[str, Any]],
    judge_fn: Optional[callable] = None,
    sample_size: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Run solubility audit on a sample of instances.
    
    Returns aggregate statistics and per-instance results.
    """
    # AGENTS.md §6.5: sampling flows through an explicit Random; default seed 0.
    rng = rng if rng is not None else random.Random(0)

    if sample_size and len(instances) > sample_size:
        # Stratified sample by family
        from collections import defaultdict
        by_family = defaultdict(list)
        for inst in instances:
            by_family[inst.get("family", "unknown")].append(inst)
        
        sampled = []
        per_family = max(1, sample_size // len(by_family))
        for fam in sorted(by_family):
            fam_insts = by_family[fam]
            sampled.extend(rng.sample(fam_insts, min(per_family, len(fam_insts))))
        
        # Trim if over
        if len(sampled) > sample_size:
            sampled = sampled[:sample_size]
    else:
        sampled = list(instances)
    
    results = []
    for inst in sampled:
        if judge_fn:
            results.append(check_solubility_llm(inst, judge_fn))
        else:
            results.append(check_answer_uniqueness(inst))
    
    # Aggregate
    total = len(results)
    soluble = sum(1 for r in results if r.is_soluble)
    mean_ambiguity = sum(r.ambiguity_score for r in results) / total if total > 0 else 0.0
    
    # Issue breakdown
    issue_counts: Dict[str, int] = {}
    for r in results:
        for issue in r.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    return {
        "total_audited": total,
        "soluble_count": soluble,
        "solubility_rate": soluble / total if total > 0 else 0.0,
        "mean_ambiguity_score": mean_ambiguity,
        "issue_breakdown": issue_counts,
        "per_instance": [vars(r) for r in results],
    }
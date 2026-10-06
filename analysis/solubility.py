"""
analysis/solubility.py
======================
Surface-solubility diagnostic for DWS-Bench instances.

Checks whether rendered instances are answerable from surface form alone,
independent of symbolic correctness. A rendering error can make an instance
unanswerable or ambiguous even if the symbolic trace is correct.

This is a diagnostic, not a generation or release gate (SPEC.md PARTIAL-11):
no acceptance threshold is applied and nothing here rejects an instance.

Checks dispatch on query type and read only record fields written by the
single ``replay_trace`` pass (AGENTS.md §6.2); no state is recomputed here.

- count: the gold is a non-negative integer, equals the final entry of
  ``step_wise_gold_answers``, and the queried container is named in the
  question and in the narrative. Mention counting is meaningless for an
  integer answer and is skipped.
- location: the gold is a known container, agrees with the final step-wise
  answer, and is mentioned at least once. "Gold is the most-mentioned
  container" is NOT required: generators deliberately break that shortcut
  (``generator/trajectories.py::_revision_walk``), so it is reported only as
  the ``ambiguity_score`` mention-share diagnostic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# analysis -> eval imports (AGENTS.md §4 permits eval helpers in analysis only
# for the offline audit script; this module predates that rule). Kept so
# candidate sets and query-type lookup each have a single owner (§5).
from eval.baselines import query_type_of
from eval.scoring import candidate_answers

_NON_NEG_INT = re.compile(r"^\d+$")


@dataclass
class SolubilityResult:
    """Result of solubility check for one instance."""
    instance_id: str
    is_soluble: bool
    # Location: mention-share diagnostic (0.0 = gold is the unique most-mentioned
    # container). It does not enter ``is_soluble``. Count: 0.0, since the answer
    # space is integers and mention share does not apply.
    ambiguity_score: float
    issues: List[str]  # verdict-bearing: any issue makes the instance insoluble
    gold_answer: str
    candidates: List[str]
    query_type: str = "location"
    warnings: List[str] = field(default_factory=list)  # reported, not verdict-bearing


def _container_names(instance: Dict[str, Any]) -> Dict[str, str]:
    final_state = instance.get("final_state") or {}
    if not isinstance(final_state, dict):
        return {}
    names = final_state.get("container_names") or final_state.get("container_display_names") or {}
    return {str(k): str(v) for k, v in names.items() if v} if isinstance(names, dict) else {}


def _final_step_answer(instance: Dict[str, Any]) -> Optional[str]:
    step_wise = instance.get("step_wise_gold_answers")
    if not step_wise:
        return None
    return str(step_wise[-1])


def _check_count(instance: Dict[str, Any], gold: str) -> List[str]:
    issues: List[str] = []
    if not _NON_NEG_INT.match(gold.strip()):
        issues.append(f"Count gold '{gold}' is not a non-negative integer")

    final_step = _final_step_answer(instance)
    if final_step is None:
        issues.append("Count record has no step_wise_gold_answers")
    elif final_step.strip() != gold.strip():
        issues.append(
            f"Count gold '{gold}' disagrees with final step-wise answer '{final_step}'"
        )

    question = str(instance.get("question", "")).lower()
    context = str(instance.get("context", "")).lower()
    names = _container_names(instance)
    gold_container = instance.get("gold_container")
    if gold_container is not None:
        queried = [names.get(str(gold_container), str(gold_container))]
    else:
        # Sorted for deterministic issue text (AGENTS.md §6.5).
        queried = sorted(n for n in names.values() if n.lower() in question)
    queried = [n for n in queried if n.lower() in question]
    if not queried:
        issues.append("Count question names no known container")
    elif not any(n.lower() in context for n in queried):
        issues.append(f"Queried container '{queried[0]}' not mentioned in context")
    return issues


def _check_location(
    instance: Dict[str, Any],
    gold: str,
    candidates: List[str],
    container_mentions: Dict[str, int],
) -> Tuple[List[str], List[str]]:
    issues: List[str] = []
    warnings: List[str] = []
    context = str(instance.get("context", "")).lower()

    # candidate_answers() always includes gold, so a single candidate means
    # there are no alternatives at all.
    if len(candidates) == 1 and candidates[0] == gold:
        issues.append(f"Gold answer '{gold}' is the only candidate (no alternatives)")

    names = _container_names(instance)
    if names and gold.lower() not in {n.lower() for n in names.values()}:
        issues.append(f"Gold answer '{gold}' is not a container of this instance")

    final_step = _final_step_answer(instance)
    if final_step is not None and final_step.strip().lower() != gold.strip().lower():
        issues.append(
            f"Gold answer '{gold}' disagrees with final step-wise answer '{final_step}'"
        )

    if gold and gold.lower() not in context:
        issues.append(f"Gold answer '{gold}' not explicitly mentioned in context")

    # A large container set widens the answer space but does not make the
    # instance insoluble.
    if len(candidates) > 5:
        warnings.append(f"High candidate count ({len(candidates)}) increases ambiguity")

    max_mentions = max(container_mentions.values()) if container_mentions else 0
    top = [c for c, cnt in container_mentions.items() if cnt == max_mentions]
    if max_mentions > 0 and gold not in top:
        # Expected for anti-shortcut families (revision); diagnostic only.
        warnings.append("Gold is not the most-mentioned container")
    return issues, warnings


def _mention_share_ambiguity(
    gold: str, candidates: List[str], container_mentions: Dict[str, int]
) -> float:
    """1 - gold's share of candidate mentions; 0.0 when gold is the unique top."""
    if not candidates:
        return 1.0
    if len(candidates) == 1:
        return 0.0
    total = sum(container_mentions.values())
    if total == 0:
        return 0.5
    max_mentions = max(container_mentions.values())
    top = [c for c, cnt in container_mentions.items() if cnt == max_mentions]
    gold_mentions = container_mentions.get(gold, 0)
    if gold_mentions == max_mentions and len(top) == 1:
        return 0.0
    return 1.0 - gold_mentions / total


def check_answer_uniqueness(
    instance: Dict[str, Any],
    threshold: float = 0.8,
) -> SolubilityResult:
    """
    Check whether the gold answer is recoverable from the rendered record.

    Dispatches on query type (see module docstring). ``threshold`` is retained
    for signature compatibility; the mention-share ``ambiguity_score`` is a
    diagnostic and no longer enters the verdict.
    """
    del threshold  # unused: see docstring
    iid = instance["instance_id"]
    gold = str(instance.get("gold_answer", ""))
    candidates = candidate_answers(instance)
    query_type = query_type_of(instance)

    warnings: List[str] = []
    if query_type == "count":
        issues = _check_count(instance, gold)
        ambiguity = 0.0
    elif query_type != "location":
        # redo_validity is a probe answer space, not a released instance query.
        issues = [f"No solubility check defined for query type '{query_type}'"]
        ambiguity = 1.0
    else:
        context = str(instance.get("context", "")).lower()
        container_mentions = {c: context.count(c.lower()) for c in candidates}
        issues, warnings = _check_location(instance, gold, candidates, container_mentions)
        ambiguity = _mention_share_ambiguity(gold, candidates, container_mentions)

    return SolubilityResult(
        instance_id=iid,
        is_soluble=not issues,
        ambiguity_score=ambiguity,
        issues=issues,
        gold_answer=gold,
        candidates=candidates,
        query_type=query_type,
        warnings=warnings,
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
            query_type=query_type_of(instance),
        )
    except Exception as e:
        return SolubilityResult(
            instance_id=iid,
            is_soluble=False,
            ambiguity_score=1.0,
            issues=[f"LLM judge error: {e}"],
            gold_answer=gold,
            candidates=candidates,
            query_type=query_type_of(instance),
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
    warning_counts: Dict[str, int] = {}
    for r in results:
        for issue in r.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for warning in r.warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    
    return {
        "total_audited": total,
        "soluble_count": soluble,
        "solubility_rate": soluble / total if total > 0 else 0.0,
        "mean_ambiguity_score": mean_ambiguity,
        "issue_breakdown": issue_counts,
        "warning_breakdown": warning_counts,
        "per_instance": [vars(r) for r in results],
    }
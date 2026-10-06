"""
eval/robustness.py
==================
Prompt sensitivity and paraphrase robustness testing for DWS-Bench.

Implements the robustness checks from the audit:
- E4: Prompt template variant testing (3 variants)
- E6: Paraphrase robustness (template-shifted variants)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from eval.baselines import query_type_of
# Prompt text and the variant builder live in eval/prompts.py (AGENTS.md §5);
# re-exported here for existing callers.
from eval.prompts import (  # noqa: F401  (re-export)
    MINIMAL_INSTRUCTIONS,
    PROMPT_TEMPLATES,
    build_prompt_variants,
    build_user_prompt,
)
from eval.scoring import candidate_answers, extract_instance_answer


# ============================================================
# Paraphrase variants (E6)
# ============================================================

# Paraphrase rules for template shifting (Yang et al., 2023)
#
# Sentence shapes are owned by render/narrative.py (render_put .. render_swap).
# Names may contain hyphens, so subjects use _WORD and container phrases use
# _PHRASE; a bare \w / [\w\s] class leaves "the red-bag" sentences unchanged.
_WORD = r"[\w-]+"
_PHRASE = r"[\w\s-]+"

PARAPHRASE_RULES = [
    # (pattern, replacement) - applied to rendered sentences
    (rf"({_WORD}) was placed in ({_PHRASE})\.", r"\1 is put into \2."),
    (rf"({_WORD}) was moved from ({_PHRASE}) to ({_PHRASE})\.", r"\1 goes from \2 to \3."),
    (rf"({_WORD}) was moved to ({_PHRASE})\.", r"\1 is moved to \2."),  # include_source=False
    (rf"({_WORD}) was taken out of ({_PHRASE})\.", r"\1 is removed from \2."),
    (r"That last action was undone\.", r"The previous action is reversed."),
    (r"The undone action was redone\.", r"The reversed action is reapplied."),
    (rf"Everything in ({_PHRASE}) was moved into ({_PHRASE})\.", r"All items in \1 are transferred to \2."),
    (rf"The contents of ({_PHRASE}) and ({_PHRASE}) were swapped\.", r"\1 and \2 exchange their contents."),
]


def apply_paraphrase(sentence: str) -> str:
    """Apply paraphrase rules to a single sentence."""
    import re
    result = sentence
    for pattern, replacement in PARAPHRASE_RULES:
        result = re.sub(pattern, replacement, result)
    return result


def paraphrase_narrative(sentences: List[str]) -> List[str]:
    """Generate a paraphrased version of the narrative."""
    return [apply_paraphrase(s) for s in sentences]


def build_paraphrase_prompt(
    context: str,
    question: str,
    paraphrased_sentences: List[str],
    chain_of_thought: bool = False,
    prompt_version: str = "v2",
    query_type: str = "location",
) -> str:
    """Build prompt with paraphrased narrative."""
    paraphrased_context = " ".join(paraphrased_sentences)
    return build_user_prompt(
        context=paraphrased_context,
        question=question,
        chain_of_thought=chain_of_thought,
        prompt_version=prompt_version,
        query_type=query_type,
    )


# ============================================================
# Robustness evaluation
# ============================================================

@dataclass
class RobustnessResult:
    """Result of a robustness check."""
    instance_id: str
    baseline_accuracy: float
    variant_accuracies: Dict[str, float]
    gap: float  # baseline - min(variant)
    is_robust: bool  # gap < threshold


def evaluate_prompt_sensitivity(
    instance: Dict[str, Any],
    predict_fn: Callable[[str], str],
    chain_of_thought: bool = False,
    threshold: float = 0.1,
) -> RobustnessResult:
    """
    Evaluate sensitivity to prompt template variants.
    
    Args:
        instance: Benchmark instance
        predict_fn: Function that takes a prompt and returns model prediction
        chain_of_thought: Whether to use CoT prompting
        threshold: Maximum allowed accuracy gap for robustness
    """
    context = instance["context"]
    question = instance["question"]
    gold = instance.get("gold_answer", "")
    query_type = query_type_of(instance)

    variants = build_prompt_variants(context, question, chain_of_thought, query_type)
    accuracies = {}
    
    for name, prompt in variants.items():
        pred = predict_fn(prompt)
        cands = candidate_answers(instance)
        extraction = extract_instance_answer(
            pred, cands, chain_of_thought=chain_of_thought, gold_answer=gold,
            instance=instance,
        )
        accuracies[name] = 1.0 if extraction.strict_correct else 0.0
    
    baseline_acc = accuracies.get("v2_standard", 0.0)
    min_acc = min(accuracies.values()) if accuracies else 0.0
    gap = baseline_acc - min_acc
    
    return RobustnessResult(
        instance_id=instance["instance_id"],
        baseline_accuracy=baseline_acc,
        variant_accuracies=accuracies,
        gap=gap,
        is_robust=gap < threshold,
    )


def evaluate_paraphrase_robustness(
    instance: Dict[str, Any],
    predict_fn: Callable[[str], str],
    chain_of_thought: bool = False,
    threshold: float = 0.1,
) -> RobustnessResult:
    """
    Evaluate robustness to paraphrased narratives.
    
    Args:
        instance: Benchmark instance
        predict_fn: Function that takes a prompt and returns model prediction
        chain_of_thought: Whether to use CoT prompting
        threshold: Maximum allowed accuracy gap for robustness
    """
    context = instance["context"]
    question = instance["question"]
    gold = instance.get("gold_answer", "")
    sentences = instance.get("sentences", [])
    query_type = query_type_of(instance)

    # Original
    orig_prompt = build_user_prompt(
        context=context,
        question=question,
        chain_of_thought=chain_of_thought,
        prompt_version="v2",
        query_type=query_type,
    )
    orig_pred = predict_fn(orig_prompt)
    cands = candidate_answers(instance)
    orig_extraction = extract_instance_answer(
        orig_pred, cands, chain_of_thought=chain_of_thought, gold_answer=gold,
        instance=instance,
    )
    orig_acc = 1.0 if orig_extraction.strict_correct else 0.0
    
    # Paraphrased
    paraphrased = paraphrase_narrative(sentences)
    para_prompt = build_paraphrase_prompt(context, question, paraphrased, chain_of_thought, query_type=query_type)
    para_pred = predict_fn(para_prompt)
    para_extraction = extract_instance_answer(
        para_pred, cands, chain_of_thought=chain_of_thought, gold_answer=gold,
        instance=instance,
    )
    para_acc = 1.0 if para_extraction.strict_correct else 0.0
    
    gap = orig_acc - para_acc
    
    return RobustnessResult(
        instance_id=instance["instance_id"],
        baseline_accuracy=orig_acc,
        variant_accuracies={"original": orig_acc, "paraphrased": para_acc},
        gap=gap,
        is_robust=gap < threshold,
    )


def run_robustness_suite(
    instances: Sequence[Dict[str, Any]],
    predict_fn: Callable[[str], str],
    chain_of_thought: bool = False,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Run full robustness suite on a set of instances."""
    prompt_results = []
    paraphrase_results = []
    
    for inst in instances:
        prompt_results.append(evaluate_prompt_sensitivity(inst, predict_fn, chain_of_thought, threshold))
        paraphrase_results.append(evaluate_paraphrase_robustness(inst, predict_fn, chain_of_thought, threshold))
    
    # Aggregate
    def summarize(results: List[RobustnessResult]) -> Dict[str, float]:
        if not results:
            return {"mean_gap": 0.0, "robust_rate": 0.0, "max_gap": 0.0}
        return {
            "mean_gap": sum(r.gap for r in results) / len(results),
            "robust_rate": sum(1 for r in results if r.is_robust) / len(results),
            "max_gap": max(r.gap for r in results),
        }
    
    return {
        "prompt_sensitivity": summarize(prompt_results),
        "paraphrase_robustness": summarize(paraphrase_results),
        "per_instance_prompt": [vars(r) for r in prompt_results],
        "per_instance_paraphrase": [vars(r) for r in paraphrase_results],
    }
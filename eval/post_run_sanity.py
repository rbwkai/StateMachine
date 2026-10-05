"""
eval/post_run_sanity.py
=======================
Post-run screening checks for a scored model run (checklist 12).

Four checks decide whether a run's numbers can be believed:
1. accuracy below the instance's own chance level means a parsing bug;
2. format compliance and truncation rate are reported per cell;
3. the strict/semantic gap is visible, so extraction losses are not read as
   reasoning losses;
4. a model beats a baseline in every cell where that baseline is weak.

Takes the report straight out of :func:`eval.eval_harness.evaluate_predictions`
so a real run can be screened with the same code the checklist tests use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from eval.baselines import chance_level
from eval.eval_harness import condition_key

# A baseline this weak is not something a model should merely match.
WEAK_BASELINE = 0.10


def sanity_checks(
    instances: Sequence[Dict[str, Any]],
    report: Dict[str, Any],
    baseline_accuracy: Optional[Dict[str, float]] = None,
    weak_baseline: float = WEAK_BASELINE,
) -> Dict[str, Dict[str, Any]]:
    """Per-condition screening statistics for a harness report.

    ``baseline_accuracy`` is an optional condition-key -> accuracy map (from
    ``run_all_baselines``); when supplied, the weak-baseline check is evaluated
    for the conditions it covers.
    """
    by_id = {inst["instance_id"]: inst for inst in instances}
    totals: Dict[str, Dict[str, Any]] = {}

    for result in report.get("instance_results", []):
        instance = by_id.get(result["instance_id"])
        if instance is None:
            continue
        entry = totals.setdefault(
            condition_key(instance),
            {"n": 0, "strict": 0, "semantic": 0, "final_line": 0, "chance": 0.0},
        )
        entry["n"] += 1
        entry["strict"] += bool(result["is_correct"])
        entry["semantic"] += bool(result["semantic_correct"])
        entry["final_line"] += bool(result["has_final_answer"])
        entry["chance"] = chance_level(instance)

    stats: Dict[str, Dict[str, Any]] = {}
    for key, entry in totals.items():
        n = entry["n"]
        strict = entry["strict"] / n
        semantic = entry["semantic"] / n
        cell: Dict[str, Any] = {
            "n": n,
            "accuracy": strict,
            "chance": entry["chance"],
            "below_chance": strict < entry["chance"] - 1e-9,
            "format_compliance": entry["final_line"] / n,
            "truncation_rate": 1.0 - entry["final_line"] / n,
            "strict_semantic_gap": semantic - strict,
        }
        if baseline_accuracy is not None and key in baseline_accuracy:
            baseline = baseline_accuracy[key]
            cell["baseline_accuracy"] = baseline
            cell["beats_weak_baseline"] = (
                baseline >= weak_baseline or strict > baseline
            )
        stats[key] = cell
    return stats


def sanity_failures(stats: Dict[str, Dict[str, Any]]) -> List[str]:
    """The screening checks a run did not pass, one message per condition."""
    failures: List[str] = []
    for key in sorted(stats):
        cell = stats[key]
        if cell["below_chance"]:
            failures.append(
                f"{key}: accuracy {cell['accuracy']:.3f} is below chance "
                f"{cell['chance']:.3f}, which signals a parsing bug"
            )
        if cell["truncation_rate"] > 0.0:
            failures.append(
                f"{key}: truncation rate {cell['truncation_rate']:.3f}"
            )
        if cell.get("beats_weak_baseline") is False:
            failures.append(
                f"{key}: accuracy {cell['accuracy']:.3f} does not beat a weak "
                f"baseline ({cell['baseline_accuracy']:.3f})"
            )
    return failures

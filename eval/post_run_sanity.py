"""
eval/post_run_sanity.py
=======================
Post-run screening checks for a scored model run (checklist 12).

Four checks decide whether a run's numbers can be believed:
1. accuracy below the cell's uniform chance (mean 1/|answer space|) flags a
   parsing bug or a model that is not tracking. The heuristic ceiling,
   effective chance and a degenerate-gold flag are reported per cell as
   diagnostics but do not gate: on basic_chain, revision and constant-gold
   cells they reach 1.0 by construction, which flagged nearly every cell;
2. format compliance, missing-final-line rate and truncation rate (generation
   stopped on ``finish_reason == "length"``) are reported per cell;
3. the strict/semantic gap is visible, so extraction losses are not read as
   reasoning losses;
4. a model beats a baseline in every cell where that baseline is weak.

Takes the report straight out of :func:`eval.eval_harness.evaluate_predictions`
so a real run can be screened with the same code the checklist tests use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from eval.baselines import (
    best_heuristic_ceiling,
    degenerate_gold,
    effective_chance,
    uniform_chance,
)
from eval.eval_harness import condition_key

# A baseline this weak is not something a model should merely match.
WEAK_BASELINE = 0.10

# The weak baselines of ``run_all_baselines`` the model is compared against.
# ``heuristic_ceiling`` is excluded on purpose: on several families the
# "shortcut" is the tracking answer itself (``HEURISTIC_NOT_A_SHORTCUT``), so it
# is a diagnostic, not a baseline a model must beat. Order breaks ties.
WEAK_BASELINE_NAMES: Tuple[str, ...] = ("stateless", "mfc")


def baseline_accuracy_by_condition(
    baselines: Mapping[str, Any],
) -> Dict[str, Tuple[Optional[str], float]]:
    """Condition key -> ``(baseline name, accuracy)`` for the weak-baseline check.

    Accepts either a flat ``condition key -> accuracy`` map (name ``None``) or
    the nested output of :func:`eval.baselines.run_all_baselines`
    (``[name]["per_condition"][key]["accuracy"]``). For the nested form each
    cell takes the *strongest* of the :data:`WEAK_BASELINE_NAMES` baselines
    (highest accuracy, first name on a tie), so a model is only required to
    beat the best of them wherever that best one is still weak.
    """
    nested = any(
        isinstance(value, Mapping) and "per_condition" in value
        for value in baselines.values()
    )
    if not nested:
        return {key: (None, float(acc)) for key, acc in baselines.items()}

    best: Dict[str, Tuple[Optional[str], float]] = {}
    for name in WEAK_BASELINE_NAMES:
        per_condition = (baselines.get(name) or {}).get("per_condition") or {}
        for key, summary in per_condition.items():
            accuracy = float(summary["accuracy"])
            if key not in best or accuracy > best[key][1]:
                best[key] = (name, accuracy)
    return best


def sanity_checks(
    instances: Sequence[Dict[str, Any]],
    report: Dict[str, Any],
    baseline_accuracy: Optional[Mapping[str, Any]] = None,
    weak_baseline: float = WEAK_BASELINE,
    predictions: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Per-condition screening statistics for a harness report.

    ``baseline_accuracy`` is optional: either the nested output of
    ``run_all_baselines`` or a flat condition-key -> accuracy map (see
    :func:`baseline_accuracy_by_condition`). When supplied, the weak-baseline
    check sets ``beats_weak_baseline`` for every condition it covers. Keys are
    :func:`eval.eval_harness.condition_key`, the same function the baselines
    bucket by, so they match.

    ``finish_reason`` is read from each instance result (the harness carries it
    through) or, failing that, from the matching row of ``predictions``.
    ``truncation_rate`` is ``None`` for a cell where no row records one.
    """
    by_id = {inst["instance_id"]: inst for inst in instances}
    baselines = (
        baseline_accuracy_by_condition(baseline_accuracy)
        if baseline_accuracy is not None
        else {}
    )
    finish_by_id = {
        row["instance_id"]: row.get("finish_reason")
        for row in (predictions or [])
        if "instance_id" in row
    }
    totals: Dict[str, Dict[str, Any]] = {}

    for result in report.get("instance_results", []):
        instance = by_id.get(result["instance_id"])
        if instance is None:
            continue
        entry = totals.setdefault(
            condition_key(instance),
            {"n": 0, "strict": 0, "semantic": 0, "final_line": 0,
             "length_stops": 0, "finish_known": 0, "records": []},
        )
        entry["n"] += 1
        entry["strict"] += bool(result["is_correct"])
        entry["semantic"] += bool(result["semantic_correct"])
        entry["final_line"] += bool(result["has_final_answer"])
        entry["records"].append(instance)
        # finish_reason is written per row by run_eval.py; a row without it
        # tells us nothing about truncation, so it is not counted as "not
        # truncated".
        finish_reason = result.get("finish_reason")
        if finish_reason is None:
            finish_reason = finish_by_id.get(result["instance_id"])
        if finish_reason is not None:
            entry["finish_known"] += 1
            entry["length_stops"] += finish_reason == "length"

    stats: Dict[str, Dict[str, Any]] = {}
    for key, entry in totals.items():
        n = entry["n"]
        strict = entry["strict"] / n
        semantic = entry["semantic"] / n
        # One chance per cell, over the scored records only (previously the
        # last instance's chance_level silently stood for the whole cell).
        # Gate on the uniform floor only; shortcut ceilings are diagnostics.
        records = entry["records"]
        chance = uniform_chance(records)
        ceiling_name, ceiling_acc = best_heuristic_ceiling(records)
        truncation: Optional[float] = (
            entry["length_stops"] / entry["finish_known"]
            if entry["finish_known"]
            else None
        )
        cell: Dict[str, Any] = {
            "n": n,
            "accuracy": strict,
            "chance": chance,
            "below_chance": strict < chance - 1e-9,
            "heuristic_ceiling_name": ceiling_name,
            "heuristic_ceiling_acc": ceiling_acc,
            "effective_chance": effective_chance(records),
            "degenerate_gold": degenerate_gold(records),
            "format_compliance": entry["final_line"] / n,
            "missing_final_rate": 1.0 - entry["final_line"] / n,
            "truncation_rate": truncation,
            "strict_semantic_gap": semantic - strict,
        }
        if key in baselines:
            baseline_name, baseline = baselines[key]
            cell["baseline_name"] = baseline_name
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
                f"{key}: accuracy {cell['accuracy']:.3f} is below uniform "
                f"chance {cell['chance']:.3f} (1/|answer space|), which "
                f"signals a parsing bug or no tracking"
            )
        # None means no finish_reason was recorded: unknown, not a failure.
        truncation = cell.get("truncation_rate")
        if truncation is not None and truncation > 0.0:
            failures.append(
                f"{key}: truncation rate {truncation:.3f} "
                f"(generation stopped on the token limit)"
            )
        if cell.get("beats_weak_baseline") is False:
            failures.append(
                f"{key}: accuracy {cell['accuracy']:.3f} does not beat a weak "
                f"baseline ({cell['baseline_accuracy']:.3f})"
            )
    return failures

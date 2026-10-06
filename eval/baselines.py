"""
eval/baselines.py
=================
Baseline predictors for DWS-Bench evaluation.

Implements the two standard baselines from the literature (Rezaee et al., 2025;
citation not re-verified here, AGENTS.md hard rule 12):
1. Stateless Baseline: predicts the initial location of the target entity
   (location) or the post-setup count in the query container (count)
2. Most-Frequent-Class (MFC) Baseline: predicts the most common answer in the condition

plus record-level shortcut solvers (``last_move``, ``prev_loc``,
``not_start_and_not_prev``, ``penultimate_move``, ``initial_count``), cell-level
modal solvers, ``best_heuristic_ceiling`` and ``effective_chance``.

The uniform ``chance_level`` is the floor a run is judged against
(``eval.post_run_sanity`` ``below_chance``); the heuristic ceiling and
``effective_chance`` are reported alongside it as shortcut diagnostics, not as
the floor, because on several families a "shortcut" is the tracking answer
itself (see ``HEURISTIC_NOT_A_SHORTCUT``).

These baselines establish the floor for tracking performance and detect
shortcuts where models answer correctly without tracking.

``query_type_of`` is also the shared record -> query_type lookup for the eval
layer (prompt construction, baselines); it is defined once, here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.eval_harness import condition_key
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


# Per-condition bucket key: one definition, owned by eval.eval_harness
# (AGENTS.md §5). Re-exported under the historical name so baseline, harness
# and post-run-sanity keys can never drift apart again.
condition_key_of = condition_key


def query_type_of(instance: Dict[str, Any]) -> str:
    """Resolve the query type of an instance record.

    Single owner of this lookup: ``generator.instance.build_validated_instance``
    serialises ``TrajectorySpec.query_type`` into the record's ``spec`` block, so
    ``spec['query_type']`` is authoritative. An explicit top-level key wins when
    present.

    ``prompt_version``-independent: callers pass the result to
    ``eval.prompts.build_user_prompt`` for "location"/"count", while
    ``chance_level`` also handles "redo_validity".
    """
    spec = instance.get("spec") or {}
    declared = instance.get("query_type") or spec.get("query_type")
    if declared in {"location", "count", "redo_validity"}:
        return str(declared)
    return "count" if str(instance.get("question", "")).startswith("How many") else "location"


def chance_level(instance: Dict[str, Any]) -> float:
    """1/size of the answer space this instance actually offers.

    Per-instance only: this is the uniform-guess floor, not the cell-level
    floor a shortcut solver reaches (see :func:`effective_chance`). A count
    instance answers with one of ``0..E+splits`` (``{1, 2}`` for split_chain,
    D-020 child-container rule); every other query type, including
    ``redo_validity`` which has no dedicated branch here, is treated as
    answering with one of the final containers.
    """
    if query_type_of(instance) == "count":
        if instance.get("family") == "split_chain":
            return 1.0 / 2  # D-020 child-container rule: answer in {1, 2}
        splits = sum(
            1
            for op in (instance.get("canonical_trace") or [])
            if op.get("op_type") == "SPLIT"
        )
        factors = instance.get("requested_factors") or {}
        entity_count = factors.get("E", 1) or 1
        return 1.0 / (entity_count + splits + 1)
    containers = (instance.get("final_state") or {}).get("containers") or []
    return 1.0 / len(containers) if containers else 0.0


# ============================================================
# Heuristic (shortcut) solvers
# ============================================================
# Each solver reads only the serialised record: ``canonical_trace`` and
# ``step_wise_gold`` both come from the one ``replay_trace`` pass in
# ``generator.instance`` (hard rule 2), aligned one entry per op, so nothing is
# re-simulated here. A solver returns a container ID (location) or a decimal
# string (count), or ``None`` when it does not apply to the record. Answers are
# compared on IDs, not display names, because display names are randomised per
# instance.

def _target_put(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    target = record.get("query_entity")
    for op in record.get("canonical_trace") or []:
        if op.get("op_type") == "PUT" and op.get("obj_id") == target:
            return op
    return None


def _target_move_indices(record: Dict[str, Any]) -> List[int]:
    target = record.get("query_entity")
    return [
        index
        for index, op in enumerate(record.get("canonical_trace") or [])
        if op.get("op_type") == "MOVE" and op.get("obj_id") == target
    ]


def heuristic_gold_key(record: Dict[str, Any]) -> str:
    """The record's gold answer in the space heuristics predict in.

    Count: the gold integer as a string. Location: the gold container ID
    (``gold_container``), falling back to reversing the display map when a
    hand-built record omits it.
    """
    gold = str(record.get("gold_answer") or "").strip()
    if query_type_of(record) == "count":
        return gold
    container = record.get("gold_container")
    if container:
        return str(container)
    names = (record.get("final_state") or {}).get("container_names") or {}
    rev = {v: k for k, v in names.items()} if isinstance(names, dict) else {}
    return rev.get(gold, normalize_text(gold))


def last_move(record: Dict[str, Any]) -> Optional[str]:
    """Destination of the last Move of the target (recency shortcut)."""
    if query_type_of(record) != "location":
        return None
    moves = _target_move_indices(record)
    if not moves:
        return None
    return (record["canonical_trace"][moves[-1]]).get("dst")


def penultimate_move(record: Dict[str, Any]) -> Optional[str]:
    """Destination of the second-to-last Move of the target."""
    if query_type_of(record) != "location":
        return None
    moves = _target_move_indices(record)
    if len(moves) < 2:
        return None
    return (record["canonical_trace"][moves[-2]]).get("dst")


def prev_loc(record: Dict[str, Any]) -> Optional[str]:
    """Where the target actually was just before its last Move.

    Read from ``step_wise_gold`` (the replay pass), so an Undo/Swap before the
    last Move is reflected; this is what an "undo the last move" shortcut
    answers.
    """
    if query_type_of(record) != "location":
        return None
    moves = _target_move_indices(record)
    gold_states = record.get("step_wise_gold") or []
    if not moves or moves[-1] == 0 or moves[-1] - 1 >= len(gold_states):
        return None
    return gold_states[moves[-1] - 1]


def _start_loc(record: Dict[str, Any]) -> Optional[str]:
    put = _target_put(record)
    return put.get("container") if put else None


def not_start_and_not_prev(record: Dict[str, Any]) -> Optional[str]:
    """The unique container that is neither the start nor ``prev_loc``.

    Exploits small container sets: with three containers and a start distinct
    from the previous location, elimination leaves one answer. ``None`` unless
    exactly one container remains.
    """
    if query_type_of(record) != "location":
        return None
    containers = (record.get("final_state") or {}).get("containers") or []
    excluded = {_start_loc(record), prev_loc(record)} - {None}
    remaining = sorted(c for c in containers if c not in excluded)
    return remaining[0] if len(remaining) == 1 else None


def initial_count(record: Dict[str, Any]) -> Optional[str]:
    """Count of the query type in the query container after setup.

    Setup is the leading run of Put ops in ``canonical_trace``; the query
    container is the record's ``gold_container`` (the count target chosen by
    ``generator.metadata.count_query_target``) and the query type is the
    target's own type. Counted from the serialised Puts, not re-simulated.
    """
    if query_type_of(record) != "count":
        return None
    container = record.get("gold_container")
    put = _target_put(record)
    if not container or put is None:
        return None
    query_type = put.get("obj_type")
    count = 0
    for op in record.get("canonical_trace") or []:
        if op.get("op_type") != "PUT":
            break
        if op.get("container") == container and op.get("obj_type") == query_type:
            count += 1
    return str(count)


def _modal(cell_records: Sequence[Dict[str, Any]], query_type: str) -> Optional[str]:
    ballots = [
        heuristic_gold_key(r)
        for r in cell_records
        if query_type_of(r) == query_type and r.get("gold_answer") not in (None, "")
    ]
    ballots = [b for b in ballots if b]
    if not ballots:
        return None
    # Ties break on first occurrence (Counter keeps insertion order), which is
    # deterministic for a fixed record order (hard rule 5).
    return Counter(ballots).most_common(1)[0][0]


def modal_location(cell_records: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Most frequent gold container ID over a cell's location records."""
    return _modal(cell_records, "location")


def modal_count(cell_records: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Most frequent gold count over a cell's count records."""
    return _modal(cell_records, "count")


# Solvers that are not shortcuts on a family because they compute the tracking
# answer by construction, so their accuracy measures nothing but the family's
# definition. ``last_move`` equals gold whenever no Undo/Redo/Swap/Merge/Split
# can affect the target after its last Move:
#   basic_chain       - Put then Moves of the target only;
#   interleaved_chain - Moves of several entities, none relocates the target
#                       except its own Moves;
#   revision          - Moves of the target that revisit earlier containers;
#                       the last Move still fixes the final location.
# A model that "uses" last_move on these families is tracking, so including it
# put effective_chance at 1.0 and flagged every cell below chance. Genuine
# shortcuts stay applicable: ``penultimate_move``/``prev_loc`` on
# undo_redo_chain and undo_chain, and ``not_start_and_not_prev`` on revision,
# where a forced elimination answer is the reviewed design flaw (A5) the
# ceiling must expose.
HEURISTIC_NOT_A_SHORTCUT: Dict[str, frozenset] = {
    "basic_chain": frozenset({"last_move"}),
    "interleaved_chain": frozenset({"last_move"}),
    "revision": frozenset({"last_move"}),
}


def heuristic_applies(name: str, record: Dict[str, Any]) -> bool:
    """False when ``name`` is the tracking answer by definition on the
    record's family (:data:`HEURISTIC_NOT_A_SHORTCUT`)."""
    return name not in HEURISTIC_NOT_A_SHORTCUT.get(str(record.get("family")), frozenset())


# Per-record solvers, by name. Cell-level solvers (modal_*, mfc) are handled in
# best_heuristic_ceiling because they need the whole cell.
RECORD_HEURISTICS = {
    "last_move": last_move,
    "prev_loc": prev_loc,
    "not_start_and_not_prev": not_start_and_not_prev,
    "penultimate_move": penultimate_move,
    "initial_count": initial_count,
}


def heuristic_accuracies(cell_records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Accuracy of every heuristic that applies to at least one cell record.

    A heuristic returning ``None`` on a record scores that record wrong (it
    gave no answer), so accuracies share the cell's full denominator. A solver
    that is the tracking answer on a record's family
    (:func:`heuristic_applies`) is treated as giving no answer there.
    """
    records = list(cell_records)
    if not records:
        return {}
    gold = [heuristic_gold_key(r) for r in records]
    out: Dict[str, float] = {}
    for name, solver in RECORD_HEURISTICS.items():
        preds = [solver(r) if heuristic_applies(name, r) else None for r in records]
        if all(p is None for p in preds):
            continue
        out[name] = sum(1 for p, g in zip(preds, gold) if p is not None and p == g) / len(records)
    for name, mode in (
        ("modal_location", modal_location(records)),
        ("modal_count", modal_count(records)),
    ):
        if mode is None:
            continue
        out[name] = sum(1 for g in gold if g == mode) / len(records)
    # MFC is the cell's modal answer whatever the query type.
    modal_accuracies = [
        out[name] for name in ("modal_location", "modal_count") if name in out
    ]
    if modal_accuracies:
        out["mfc"] = max(modal_accuracies)
    return out


def best_heuristic_ceiling(cell_records: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
    """``(name, accuracy)`` of the strongest shortcut solver on this cell.

    Ties resolve to the first name in ``heuristic_accuracies`` order. Returns
    ``("none", 0.0)`` for an empty cell.
    """
    accuracies = heuristic_accuracies(cell_records)
    if not accuracies:
        return ("none", 0.0)
    name = max(accuracies, key=lambda k: accuracies[k])
    return (name, accuracies[name])


def uniform_chance(cell_records: Sequence[Dict[str, Any]]) -> float:
    """Mean per-instance :func:`chance_level` over a cell (0.0 when empty).

    The answer space can vary within a cell (e.g. split count), hence the mean.
    This is the floor ``below_chance`` gates on.
    """
    records = list(cell_records)
    if not records:
        return 0.0
    return sum(chance_level(r) for r in records) / len(records)


def degenerate_gold(cell_records: Sequence[Dict[str, Any]]) -> bool:
    """True when every record in the cell has the same gold answer.

    MFC then scores 1.0 by construction, so the cell cannot separate tracking
    from guessing the constant; that is a dataset property, reported here
    rather than folded into a floor.
    """
    keys = {heuristic_gold_key(r) for r in cell_records}
    return len(keys) == 1


def effective_chance(cell_records: Sequence[Dict[str, Any]]) -> float:
    """Strongest non-tracking answerer on the cell: a diagnostic, not the floor.

    ``max(1/|answer space|, MFC accuracy, best applicable heuristic
    accuracy)``. Heuristics that are the tracking answer on the family are
    excluded (:data:`HEURISTIC_NOT_A_SHORTCUT`). MFC still reaches 1.0 on a
    :func:`degenerate_gold` cell, which is why ``below_chance`` gates on
    :func:`uniform_chance` instead.
    """
    records = list(cell_records)
    if not records:
        return 0.0
    uniform = uniform_chance(records)
    accuracies = heuristic_accuracies(records)
    mfc = accuracies.get("mfc", 0.0)
    best = max(accuracies.values(), default=0.0)
    return max(uniform, mfc, best)


def compute_stateless_baseline(instances: Sequence[Dict[str, Any]]) -> List[BaselineResult]:
    """
    Stateless baseline: answer based on initial state only.

    Location query: the target's initial container (its own Put). Count query:
    :func:`initial_count`, the post-setup count of the query type in the query
    container. (Previously every count query predicted "0": the merge-
    destination lookup computed a container and then discarded it.)
    """
    results: List[BaselineResult] = []

    for inst in instances:
        iid = inst["instance_id"]
        gold = str(inst.get("gold_answer", "")).strip()

        if query_type_of(inst) == "count":
            pred = initial_count(inst) or ""
        else:
            initial_container = _start_loc(inst)
            if initial_container is None:
                pred = ""
            else:
                # Use display name so extraction matches the candidate list.
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

    Computed per condition (family + T + D + E + N) to avoid leakage across
    conditions. Groups on display-mapped answers so extraction accepts them.
    """
    # Group by condition
    from collections import defaultdict
    cond_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def _cond_key(inst: Dict[str, Any]) -> str:
        return condition_key_of(inst)

    for inst in instances:
        cond_instances[_cond_key(inst)].append(inst)
    
    # Find MFC per condition on container IDs (display names are randomized
    # per instance, so voting on raw gold would never agree). Reverse each
    # instance's ID->display map to ballot the ID; digits pass through.
    def _ballot(inst: Dict[str, Any]) -> str:
        gold = str(inst.get("gold_answer", "")).strip()
        if gold.isdigit():
            return gold
        names = (inst.get("final_state", {}) or {}).get(
            "container_display_names"
        ) or (inst.get("final_state", {}) or {}).get("container_names") or {}
        rev = {v: k for k, v in names.items()} if isinstance(names, dict) else {}
        return rev.get(gold, normalize_text(gold))

    cond_mfc: Dict[str, str] = {}
    for cond_key, cond_insts in cond_instances.items():
        answers = [_ballot(i) for i in cond_insts if i.get("gold_answer")]
        answers = [a for a in answers if a]
        if answers:
            cond_mfc[cond_key] = Counter(answers).most_common(1)[0][0]
        else:
            cond_mfc[cond_key] = ""
    
    # Apply MFC prediction, mapped to display names like stateless does so
    # extraction accepts the prediction against the candidate list.
    results: List[BaselineResult] = []
    for inst in instances:
        iid = inst["instance_id"]
        gold = str(inst.get("gold_answer", "")).strip()

        cond_key = _cond_key(inst)
        pred = cond_mfc.get(cond_key, "")
        if pred and not pred.isdigit():
            final_state = inst.get("final_state", {})
            names = final_state.get("container_names") or final_state.get(
                "container_display_names"
            )
            if isinstance(names, dict):
                pred = names.get(pred, pred)
        
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
    """Run both baselines plus the per-cell heuristic ceiling and
    effective chance, and return the combined summary."""
    stateless_results = compute_stateless_baseline(instances)
    mfc_results = compute_mfc_baseline(instances)
    
    cells: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instances:
        cells.setdefault(condition_key_of(inst), []).append(inst)
    ceilings: Dict[str, Dict[str, Any]] = {}
    for cond, cell in cells.items():
        name, accuracy = best_heuristic_ceiling(cell)
        ceilings[cond] = {
            "name": name,
            "accuracy": accuracy,
            "heuristic_ceiling_name": name,
            "heuristic_ceiling_acc": accuracy,
            "chance_level": uniform_chance(cell),
            "effective_chance": effective_chance(cell),
            "degenerate_gold": degenerate_gold(cell),
        }

    return {
        "heuristic_ceiling": {"per_condition": ceilings},
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

    by_id = {r.instance_id: r for r in reversed(results)}  # first result wins
    cond_results: Dict[str, List[BaselineResult]] = defaultdict(list)
    for inst in instances:
        result = by_id.get(inst["instance_id"])
        if result is not None:
            cond_results[condition_key_of(inst)].append(result)

    return {
        cond: {
            "total": len(rs),
            "correct": sum(1 for r in rs if r.is_correct),
            "accuracy": sum(1 for r in rs if r.is_correct) / len(rs) if rs else 0.0,
        }
        for cond, rs in cond_results.items()
    }
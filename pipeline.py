
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, Optional, Sequence

from generator import (
    CountQuery, LocationQuery, build_counterfactual_probes,
    build_redo_validity_example, select_query, step_wise_gold,
)
from analysis import QuerySpec, analyze_trajectory
from render import (
    NameRegistry, make_distractor_sentences, question_count,
    question_counterfactual, question_location, question_redo_validity,
    render_narrative, splice_distractors,
)
from sampler import sample_sequence
from world import Operation, Redo, Undo


def _op_to_dict(op: Operation) -> Dict:
    d = asdict(op)
    d["type"] = type(op).__name__.upper()
    return d


def generate_example(
    rng: random.Random,
    example_id: str,
    entity_count: int,
    update_count: int,
    distractor_count: int,
    operations_enabled: Sequence[type],
    query_spec: Optional[QuerySpec] = None,
    include_counterfactual: bool = True,
    force_redo_probe: bool = False,
    max_trajectory_attempts: int = 200,
) -> Dict:
    """Constraint-first instance generation.

    The trajectory is repeatedly sampled until a candidate query satisfies
    QuerySpec. Difficulty is therefore encoded in verified structural
    properties rather than operation weights.
    """
    if query_spec is None:
        query_spec = QuerySpec(
            query_type="location",
            must_change_from_initial=True,
            min_relevant_steps=1,
            min_state_changes=1,
        )

    if force_redo_probe:
        ops, final_state, history, containers, redo_info = build_redo_validity_example(
            rng, entity_count, update_count, operations_enabled
        )
        names = NameRegistry(rng, containers)
        op_sentences, replay_final = render_narrative(ops, containers, names)

        if replay_final.location != final_state.location:
            raise AssertionError("canonical replay disagrees with sampled state")

        distractors = make_distractor_sentences(
            rng, distractor_count, names, sorted(set(final_state.object_type.values()))
        )
        sentences = splice_distractors(rng, op_sentences, distractors)

        return {
            "id": example_id,
            "factors": {
                "E": entity_count,
                "U": len(ops),
                "N": distractor_count,
                "R": sum(
                    1 for op in ops
                    if type(op).__name__ in ("Undo", "Redo")
                ),
            },
            "operations": [_op_to_dict(op) for op in ops],
            "sentences": sentences,
            "query": {"type": "redo_validity"},
            "question": question_redo_validity(),
            "gold_answer": redo_info["would_be_valid"],
        }

    last_error = None

    for attempt in range(max_trajectory_attempts):
        try:
            ops, final_state, history, containers = sample_sequence(
                rng, entity_count, update_count, operations_enabled
            )

            query, analysis = select_query(
                rng, ops, final_state, containers, query_spec
            )

            names = NameRegistry(rng, containers)
            op_sentences, replay_final_state = render_narrative(
                ops, containers, names
            )

            if replay_final_state.location != final_state.location:
                raise AssertionError(
                    "renderer replay and sampled final state disagree"
                )

            used_types = sorted(set(final_state.object_type.values()))
            distractors = make_distractor_sentences(
                rng, distractor_count, names, used_types
            )
            sentences = splice_distractors(rng, op_sentences, distractors)

            record: Dict = {
                "id": example_id,
                "attempt": attempt,
                "factors": {
                    "E": entity_count,
                    "D": update_count,
                    "N": distractor_count,
                    "R": sum(isinstance(op, (Undo, Redo)) for op in ops),
                    "query_spec": asdict(query_spec),
                },
                "operations": [_op_to_dict(op) for op in ops],
                "sentences": sentences,
                "query": {},
                "question": "",
                "gold_answer": query.read(final_state),
                "step_wise_gold": step_wise_gold(ops, containers, query),
                "analysis": analysis.to_dict(),
            }

            if isinstance(query, LocationQuery):
                record["query"] = {
                    "type": "location",
                    "target": query.obj_id,
                    "canonical": {
                        "type": "FINAL_LOCATION",
                        "entity": query.obj_id,
                    },
                }
                record["question"] = question_location(
                    query.obj_id, final_state, names
                )
            else:
                record["query"] = {
                    "type": "count",
                    "target": {
                        "container": query.container,
                        "type": query.obj_type,
                    },
                    "canonical": {
                        "type": "COUNT",
                        "container": query.container,
                        "object_type": query.obj_type,
                    },
                }
                record["question"] = question_count(
                    query.container, query.obj_type, names
                )

            if include_counterfactual:
                raw = build_counterfactual_probes(
                    rng, ops, containers, query
                )
                cf = []
                for probe in raw:
                    idx = probe["remove_step"]
                    removed_sentence = op_sentences[idx]
                    cf.append({
                        "remove_step": idx,
                        "removed_sentence": removed_sentence,
                        "gold_answer": probe["gold_answer"],
                    })
                record["counterfactual_probes"] = cf

            return record

        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"could not generate a trajectory satisfying {query_spec!r} "
        f"after {max_trajectory_attempts} attempts; last error={last_error!r}"
    )

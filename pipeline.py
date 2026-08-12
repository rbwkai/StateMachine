from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Sequence

from generator import (
    CountQuery, LocationQuery, build_counterfactual_probes,
    build_redo_validity_example, select_query, sample_sequence, step_wise_gold,
)
from render import (
    NameRegistry, make_distractor_sentences, question_count,
    question_counterfactual, question_location, question_redo_validity,
    render_narrative, splice_distractors,
)
from world import Operation


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
    include_counterfactual: bool = True,
    force_redo_probe: bool = False,
) -> Dict:
    """Produces one fully-assembled example record: narrative sentences,
    question, gold answer, step-wise gold trajectory, and (optionally)
    counterfactual probes or a redo-validity probe.
    """
    redo_probe_info = None
    if force_redo_probe:
        ops, final_state, history, containers, redo_probe_info = build_redo_validity_example(
            rng, entity_count, update_count, operations_enabled
        )
    else:
        ops, final_state, history, containers = sample_sequence(
            rng, entity_count, update_count, operations_enabled
        )

    names = NameRegistry(rng, containers)
    op_sentences, replay_final_state = render_narrative(ops, containers, names)
    assert replay_final_state.location == final_state.location, (
        "renderer's replay and generator's own final state disagree -- "
        "this should be impossible since both go through apply_op/replay_trace"
    )

    used_types = sorted(set(final_state.object_type.values()))
    distractors = make_distractor_sentences(rng, distractor_count, names, used_types)
    sentences = splice_distractors(rng, op_sentences, distractors)

    record: Dict = {
        "id": example_id,
        "factors": {
            "E": entity_count,
            "U": update_count,
            "N": distractor_count,
            "R": sum(1 for op in ops if type(op).__name__ in ("Undo", "Redo")),
        },
        "operations": [_op_to_dict(op) for op in ops],
        "sentences": sentences,
    }

    if force_redo_probe:
        record["query"] = {"type": "redo_validity"}
        record["question"] = question_redo_validity()
        record["gold_answer"] = redo_probe_info["would_be_valid"]
        return record

    query = select_query(rng, ops, final_state)
    if isinstance(query, LocationQuery):
        record["query"] = {"type": "location", "target": query.obj_id}
        record["question"] = question_location(query.obj_id, final_state, names)
        record["gold_answer"] = query.read(final_state)
    else:
        record["query"] = {
            "type": "count",
            "target": {"container": query.container, "type": query.obj_type},
        }
        record["question"] = question_count(query.container, query.obj_type, names)
        record["gold_answer"] = query.read(final_state)

    record["step_wise_gold"] = step_wise_gold(ops, containers, query)

    if include_counterfactual:
        raw_probes = build_counterfactual_probes(rng, ops, containers, query)
        cf_probes = []
        for probe in raw_probes:
            removed_op_index = probe["remove_step"]
            removed_sentence = op_sentences[removed_op_index]
            cf_probes.append({
                "remove_step": removed_op_index,
                "removed_sentence": removed_sentence,
                "gold_answer": probe["gold_answer"],
                "question": (
                    question_counterfactual(removed_sentence, query.obj_id, final_state)
                    if isinstance(query, LocationQuery)
                    else (
                        f'Suppose this had not happened: "{removed_sentence}" '
                        f"How many {query.obj_type}s would be in "
                        f"{names.container(query.container)} now?"
                    )
                ),
            })
        record["counterfactual_probes"] = cf_probes

    return record
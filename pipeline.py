from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict, Optional, Sequence

from generator import (
    CountQuery,
    LocationQuery,
    build_counterfactual_probes,
    build_redo_validity_example,
    select_query,
)
from analysis import QuerySpec
from render import (
    NameRegistry,
    make_distractor_sentences,
    question_count,
    question_redo_validity,
    question_location,
    render_narrative,
    splice_distractors,
)
from sampler import sample_sequence
from trajectory import (
    build_trajectory_and_gold,
    trajectory_summary,
)
from world import Operation, Redo, Undo, replay_trace


def _op_to_dict(op: Operation) -> Dict:
    """Serialize an Operation into a JSON-safe dictionary."""
    d = asdict(op)
    d["type"] = type(op).__name__.upper()
    return d


def _state_to_dict(state) -> Dict:
    """Serialize a WorldState into a JSON-safe dictionary."""
    return {
        "object_type": dict(
            sorted(state.object_type.items())
        ),
        "location": dict(
            sorted(state.location.items())
        ),
        "containers": sorted(state.containers),
        "step_index": state.step_index,
    }


def _build_redo_trajectory(
    ops: Sequence[Operation],
    containers,
    op_sentences: Sequence[str],
) -> list[Dict]:
    """
    Build the symbolic trajectory for a redo-validity probe.

    Redo-validity is a meta-query about whether a redo operation
    would be valid, so it does not have the normal query projection
    used by LocationQuery or CountQuery.
    """
    trace, _, _ = replay_trace(
        ops,
        containers,
    )

    if len(trace) != len(op_sentences):
        raise AssertionError(
            "redo trajectory and operation sentences "
            "have different lengths"
        )

    trajectory: list[Dict] = []

    for step, (op, before, after) in enumerate(trace):
        trajectory.append(
            {
                "step": step,

                "operation": _op_to_dict(op),

                "sentence": op_sentences[step],

                "state_before": _state_to_dict(
                    before
                ),

                "state_after": _state_to_dict(
                    after
                ),

                "answer_before": None,

                "answer_after": None,

                "state_changed": (
                    before != after
                ),

                "query_changed": None,
            }
        )

    return trajectory


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
    """
    Generate one DWS-Bench instance.

    Generation is constraint-first:

        sample operations
            ↓
        determine symbolic final state
            ↓
        select query satisfying QuerySpec
            ↓
        replay canonical trajectory
            ↓
        render operations into natural language
            ↓
        inject distractors
            ↓
        export benchmark record

    The symbolic WorldState/replay trace is the source of truth.
    """

    if query_spec is None:
        query_spec = QuerySpec(
            query_type="location",
            must_change_from_initial=True,
            min_relevant_steps=1,
            min_state_changes=1,
        )

    # ============================================================
    # SPECIAL REDO-VALIDITY PROBE
    # ============================================================
    if force_redo_probe:
        (
            ops,
            final_state,
            history,
            containers,
            redo_info,
        ) = build_redo_validity_example(
            rng,
            entity_count,
            update_count,
            operations_enabled,
        )

        names = NameRegistry(
            rng,
            containers,
        )

        op_sentences, replay_final = render_narrative(
            ops,
            containers,
            names,
        )

        if replay_final.location != final_state.location:
            raise AssertionError(
                "canonical replay disagrees with sampled state"
            )

        trajectory = _build_redo_trajectory(
            ops,
            containers,
            op_sentences,
        )

        distractors = make_distractor_sentences(
            rng,
            distractor_count,
            names,
            sorted(
                set(final_state.object_type.values())
            ),
        )

        sentences = splice_distractors(
            rng,
            op_sentences,
            distractors,
        )

        return {
            "id": example_id,

            "attempt": 0,

            "factors": {
                "E": entity_count,
                "U": len(ops),
                "N": distractor_count,
                "R": sum(
                    isinstance(op, (Undo, Redo))
                    for op in ops
                ),
                "query_spec": None,
            },

            "operations": [
                _op_to_dict(op)
                for op in ops
            ],

            "sentences": sentences,

            "query": {
                "type": "redo_validity",
                "canonical": {
                    "type": "REDO_VALIDITY",
                },
            },

            "question": question_redo_validity(),

            "gold_answer": redo_info[
                "would_be_valid"
            ],

            "trajectory": trajectory,

            "trajectory_summary": {
                "length": len(trajectory),

                "state_change_count": sum(
                    step["state_changed"]
                    for step in trajectory
                ),

                "query_change_count": None,

                "first_query_change": None,

                "last_query_change": None,
            },

            "step_wise_gold": None,

            "counterfactual_probes": [],
        }

    # ============================================================
    # NORMAL TRAJECTORY GENERATION
    # ============================================================
    last_error = None

    for attempt in range(
        max_trajectory_attempts
    ):
        try:

            # ----------------------------------------------------
            # 1. Generate symbolic operation sequence
            # ----------------------------------------------------
            (
                ops,
                final_state,
                history,
                containers,
            ) = sample_sequence(
                rng,
                entity_count,
                update_count,
                operations_enabled,
            )

            # ----------------------------------------------------
            # 2. Select query satisfying QuerySpec
            # ----------------------------------------------------
            query, analysis = select_query(
                rng,
                ops,
                final_state,
                containers,
                query_spec,
            )

            # ----------------------------------------------------
            # 3. Render actual operations
            # ----------------------------------------------------
            names = NameRegistry(
                rng,
                containers,
            )

            (
                op_sentences,
                replay_final_state,
            ) = render_narrative(
                ops,
                containers,
                names,
            )

            # ----------------------------------------------------
            # 4. Verify renderer replay
            # ----------------------------------------------------
            if (
                replay_final_state.location
                != final_state.location
            ):
                raise AssertionError(
                    "renderer replay and sampled "
                    "final state disagree"
                )

            # ----------------------------------------------------
            # 5. Build canonical trajectory AND gold answers
            # ----------------------------------------------------
            (
                trajectory,
                step_wise,
            ) = build_trajectory_and_gold(
                ops=ops,
                containers=containers,
                query=query,
                op_sentences=op_sentences,
            )

            # ----------------------------------------------------
            # 6. Verify final trajectory answer
            # ----------------------------------------------------
            if not step_wise:
                raise AssertionError(
                    "trajectory produced no steps"
                )

            trajectory_final_answer = step_wise[-1]

            direct_final_answer = query.read(
                final_state
            )

            if (
                trajectory_final_answer
                != direct_final_answer
            ):
                raise AssertionError(
                    "trajectory final answer disagrees "
                    "with final state"
                )

            # ----------------------------------------------------
            # 7. Add natural-language distractors
            #
            # IMPORTANT:
            # Distractors are NOT added to trajectory.
            # ----------------------------------------------------
            used_types = sorted(
                set(
                    final_state.object_type.values()
                )
            )

            distractors = make_distractor_sentences(
                rng,
                distractor_count,
                names,
                used_types,
            )

            sentences = splice_distractors(
                rng,
                op_sentences,
                distractors,
            )

            # ----------------------------------------------------
            # 8. Build base record
            # ----------------------------------------------------
            record: Dict = {
                "id": example_id,

                "attempt": attempt,

                "factors": {
                    "E": entity_count,
                    "U": update_count,
                    "N": distractor_count,
                    "R": sum(
                        isinstance(
                            op,
                            (Undo, Redo),
                        )
                        for op in ops
                    ),
                    "query_spec": asdict(
                        query_spec
                    ),
                },

                "operations": [
                    _op_to_dict(op)
                    for op in ops
                ],

                "sentences": sentences,

                "query": {},

                "question": "",

                "gold_answer": query.read(
                    final_state
                ),

                "trajectory": trajectory,

                "step_wise_gold": step_wise,

                "trajectory_summary": (
                    trajectory_summary(
                        trajectory
                    )
                ),

                "analysis": analysis.to_dict(),
            }

            # ----------------------------------------------------
            # 9. Serialize query
            # ----------------------------------------------------
            if isinstance(
                query,
                LocationQuery,
            ):

                record["query"] = {
                    "type": "location",

                    "target": query.obj_id,

                    "canonical": {
                        "type": "FINAL_LOCATION",
                        "entity": query.obj_id,
                    },
                }

                record["question"] = (
                    question_location(
                        query.obj_id,
                        final_state,
                        names,
                    )
                )

            elif isinstance(
                query,
                CountQuery,
            ):

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

                record["question"] = (
                    question_count(
                        query.container,
                        query.obj_type,
                        names,
                    )
                )

            else:
                raise TypeError(
                    "Unsupported query type: "
                    f"{type(query).__name__}"
                )

            # ----------------------------------------------------
            # 10. Counterfactual probes
            # ----------------------------------------------------
            if include_counterfactual:

                raw = build_counterfactual_probes(
                    rng,
                    ops,
                    containers,
                    query,
                )

                counterfactuals = []

                for probe in raw:

                    idx = probe[
                        "remove_step"
                    ]

                    if not (
                        0
                        <= idx
                        < len(op_sentences)
                    ):
                        raise IndexError(
                            "counterfactual "
                            f"remove_step={idx} "
                            "is outside operation "
                            f"range 0..{len(op_sentences) - 1}"
                        )

                    counterfactuals.append(
                        {
                            "remove_step": idx,

                            "removed_sentence": (
                                op_sentences[idx]
                            ),

                            "gold_answer": (
                                probe[
                                    "gold_answer"
                                ]
                            ),
                        }
                    )

                record[
                    "counterfactual_probes"
                ] = counterfactuals

            else:
                record[
                    "counterfactual_probes"
                ] = []

            # ----------------------------------------------------
            # 11. Final consistency checks
            # ----------------------------------------------------
            if (
                record["gold_answer"]
                != step_wise[-1]
            ):
                raise AssertionError(
                    "gold_answer and final "
                    "step_wise_gold disagree"
                )

            if len(trajectory) != len(ops):
                raise AssertionError(
                    "trajectory length must equal "
                    "operation count"
                )

            if len(step_wise) != len(ops):
                raise AssertionError(
                    "step_wise_gold length must equal "
                    "operation count"
                )

            return record

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"could not generate a trajectory "
        f"satisfying {query_spec!r} after "
        f"{max_trajectory_attempts} attempts; "
        f"last error={last_error!r}"
    )

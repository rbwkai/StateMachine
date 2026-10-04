"""Small package-level smoke test for the current generator contract."""

from __future__ import annotations

import random

from generator import (
    LocationQuery,
    TrajectorySpec,
    build_counterfactual_probes,
    build_redo_validity_example,
    build_trajectory,
    select_query,
    step_wise_gold,
)
from analysis.query_analysis import QuerySpec
from generator.sampler import sample_sequence
from render import NameRegistry, render_narrative
from world import InvalidOperation, Move, Put, Redo, Undo, WorldState, apply_op, can_redo, replay_trace


def main() -> None:
    containers = {"c0", "c1", "c2"}
    state = WorldState(object_type={}, location={}, containers=set(containers))
    from world import History

    history = History()
    state = apply_op(Put("o0", "phone", "c0"), state, history)
    state = apply_op(Move("o0", "c1"), state, history)
    state = apply_op(Undo(), state, history)
    assert state.location["o0"] == "c0"
    assert can_redo(history)
    state = apply_op(Redo(), state, history)
    assert state.location["o0"] == "c1"

    try:
        apply_op(Move("missing", "c1"), WorldState({}, {}, set(containers)), History())
    except InvalidOperation:
        pass
    else:
        raise AssertionError("invalid Move was accepted")

    ops, final_state, _history, sampled_containers = sample_sequence(
        random.Random(12345),
        entity_count=4,
        update_count=8,
        operations_enabled=[Put, Move],
    )
    _, replay_final, _ = replay_trace(ops, sampled_containers)
    assert final_state.location == replay_final.location

    spec = TrajectorySpec(
        family="basic_chain",
        entity_count=1,
        total_updates=4,
        target_updates=4,
    )
    built = build_trajectory(random.Random(7), spec)
    query = LocationQuery(built.target_obj)
    assert step_wise_gold(built.ops, built.containers, query)
    names = NameRegistry(random.Random(7), built.containers)
    sentences, rendered_state = render_narrative(
        built.ops,
        built.containers,
        names,
        include_move_sources=False,
    )
    assert rendered_state.location == built.final_state.location
    assert all("moved from" not in sentence.lower() for sentence in sentences)

    probes = build_counterfactual_probes(
        random.Random(9), built.ops, built.containers, query, max_probes=2
    )
    assert probes
    redo_example = build_redo_validity_example(
        random.Random(7),
        entity_count=3,
        update_count=6,
        operations_enabled=[Put, Move, Undo],
        num_containers=3,
        redo_class="valid",
    )
    assert redo_example

    selected_query, analysis = select_query(
        random.Random(8),
        built.ops,
        built.final_state,
        built.containers,
        query_spec=QuerySpec(query_type="location"),
    )
    assert selected_query and analysis
    print("generator package smoke test: PASS")


if __name__ == "__main__":
    main()

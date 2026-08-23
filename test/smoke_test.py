from __future__ import annotations

import inspect
from pathlib import Path
import random
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generator
from analysis import QuerySpec
from generator import (
    CountQuery,
    LocationQuery,
    build_counterfactual_probes,
    build_redo_validity_example,
    select_query,
)
from pipeline import generate_example
from sampler import sample_sequence
from trajectory import build_trajectory_and_gold, trajectory_summary
from world import (
    History,
    InvalidOperation,
    Move,
    Put,
    Redo,
    Remove,
    Split,
    Swap,
    Undo,
    WorldState,
    apply_op,
    can_redo,
    can_undo,
    gold_location,
    replay_trace,
)


# ============================================================
# Helpers
# ============================================================

def banner(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


# ============================================================
# 1. IMPORT / API CHECK
# ============================================================

banner("1. IMPORT / API CHECK")

print("generator:", generator.__file__)

check(
    hasattr(generator, "select_query"),
    "generator.select_query missing",
)

check(
    hasattr(generator, "build_counterfactual_probes"),
    "generator.build_counterfactual_probes missing",
)

check(
    hasattr(generator, "build_redo_validity_example"),
    "generator.build_redo_validity_example missing",
)

print(
    "select_query:",
    inspect.signature(generator.select_query),
)

print(
    "build_counterfactual_probes:",
    inspect.signature(
        generator.build_counterfactual_probes
    ),
)

print("PASS")


# ============================================================
# 2. WORLD STATE BASIC OPERATION CHECK
# ============================================================

banner("2. WORLD STATE BASIC OPERATIONS")

containers = {"c0", "c1"}

state = WorldState(
    object_type={},
    location={},
    containers=set(containers),
)

history = History()

put = Put(
    obj_id="o0",
    obj_type="phone",
    container="c0",
)

state = apply_op(
    put,
    state,
    history,
)

check(
    state.location["o0"] == "c0",
    "Put failed",
)

check(
    state.object_type["o0"] == "phone",
    "Put did not preserve object type",
)

check(
    can_undo(history),
    "Undo stack should contain Put",
)

move = Move(
    obj_id="o0",
    dst="c1",
)

state = apply_op(
    move,
    state,
    history,
)

check(
    state.location["o0"] == "c1",
    "Move failed",
)

check(
    can_undo(history),
    "Undo stack should contain Move",
)

undo = Undo()

state = apply_op(
    undo,
    state,
    history,
)

check(
    state.location["o0"] == "c0",
    "Undo did not restore previous state",
)

check(
    can_redo(history),
    "Redo should be available after Undo",
)

redo = Redo()

state = apply_op(
    redo,
    state,
    history,
)

check(
    state.location["o0"] == "c1",
    "Redo did not restore moved state",
)

check(
    not can_redo(history),
    "Redo stack should be empty after Redo",
)

print("Put -> Move -> Undo -> Redo: PASS")


# ============================================================
# 3. NEW ACTION INVALIDATES REDO
# ============================================================

banner("3. REDO INVALIDATION")

state = WorldState(
    object_type={},
    location={},
    containers=set(containers),
)

history = History()

state = apply_op(
    Put("o0", "phone", "c0"),
    state,
    history,
)

state = apply_op(
    Move("o0", "c1"),
    state,
    history,
)

state = apply_op(
    Undo(),
    state,
    history,
)

check(
    can_redo(history),
    "Redo should exist after Undo",
)

# A new normal operation must clear redo.
state = apply_op(
    Put("o1", "book", "c0"),
    state,
    history,
)

check(
    not can_redo(history),
    "New operation failed to invalidate redo history",
)

print("Undo -> new action -> redo invalidation: PASS")


# ============================================================
# 4. REPLAY TRACE CHECK
# ============================================================

banner("4. REPLAY TRACE")

ops = [
    Put("o0", "phone", "c0"),
    Move("o0", "c1"),
]

trace, final_state, final_history = replay_trace(
    ops,
    containers,
)

check(
    len(trace) == len(ops),
    "Trace length != operation length",
)

check(
    trace[0][0] == ops[0],
    "Trace operation 0 mismatch",
)

check(
    trace[1][0] == ops[1],
    "Trace operation 1 mismatch",
)

check(
    trace[0][1].location == {},
    "Initial trace state incorrect",
)

check(
    trace[0][2].location["o0"] == "c0",
    "State after Put incorrect",
)

check(
    trace[1][1].location["o0"] == "c0",
    "Move before-state incorrect",
)

check(
    trace[1][2].location["o0"] == "c1",
    "Move after-state incorrect",
)

check(
    final_state.location["o0"] == "c1",
    "Replay final state incorrect",
)

print("Replay trace: PASS")


# ============================================================
# 5. INVALID OPERATION CHECK
# ============================================================

banner("5. INVALID OPERATION HANDLING")

state = WorldState(
    object_type={},
    location={},
    containers=set(containers),
)

history = History()

try:
    apply_op(
        Move("o404", "c1"),
        state,
        history,
    )
except InvalidOperation:
    print("Invalid Move correctly rejected: PASS")
else:
    raise AssertionError(
        "Invalid Move was accepted"
    )

try:
    apply_op(
        Undo(),
        state,
        history,
    )
except InvalidOperation:
    print("Invalid Undo correctly rejected: PASS")
else:
    raise AssertionError(
        "Invalid Undo was accepted"
    )


# ============================================================
# 6. SAMPLER CHECK
# ============================================================

banner("6. SAMPLER")

rng = random.Random(12345)

ops, final_state, history, containers = sample_sequence(
    rng=rng,
    entity_count=4,
    update_count=8,
    operations_enabled=[
        Put,
        Move,
        Remove,
    ],
)

check(
    len(ops) == 8,
    f"Sampler produced {len(ops)} operations instead of 8",
)

trace, replay_final, _ = replay_trace(
    ops,
    containers,
)

check(
    len(trace) == len(ops),
    "Sampled trajectory cannot be replayed",
)

check(
    final_state.location == replay_final.location,
    "Sampler final state disagrees with replay",
)

check(
    final_state.object_type == replay_final.object_type,
    "Sampler object types disagree with replay",
)

print("Sample sequence: PASS")


# ============================================================
# 7. QUERY SELECTION
# ============================================================

banner("7. QUERY SELECTION")

query_spec = QuerySpec(
    query_type="location",
    must_change_from_initial=True,
    min_relevant_steps=1,
    min_state_changes=1,
)

rng = random.Random(777)

ops, final_state, history, containers = sample_sequence(
    rng=rng,
    entity_count=4,
    update_count=8,
    operations_enabled=[
        Put,
        Move,
        Remove,
    ],
)

query, analysis = select_query(
    rng,
    ops,
    final_state,
    containers,
    query_spec,
)

check(
    isinstance(query, LocationQuery),
    "Location QuerySpec produced non-location query",
)

check(
    query_spec.matches(analysis),
    "Returned query does not satisfy QuerySpec",
)

print("Selected query:", query)
print("Analysis:", analysis.to_dict())
print("Query selection: PASS")


# ============================================================
# 8. TRAJECTORY + STEP-WISE GOLD
# ============================================================

banner("8. TRAJECTORY / STEP-WISE GOLD")

trajectory, step_wise = build_trajectory_and_gold(
    ops=ops,
    containers=containers,
    query=query,
    op_sentences=[
        f"operation {i}"
        for i in range(len(ops))
    ],
)

check(
    len(trajectory) == len(ops),
    "Trajectory length != operation count",
)

check(
    len(step_wise) == len(ops),
    "Step-wise gold length != operation count",
)

direct_answer = query.read(
    final_state
)

check(
    step_wise[-1] == direct_answer,
    "Final step-wise gold != direct final answer",
)

check(
    trajectory[-1]["answer_after"]
    == direct_answer,
    "Trajectory final answer != direct answer",
)

summary = trajectory_summary(
    trajectory
)

print("SUMMARY")
print(summary)

check(
    "length" in summary,
    "trajectory summary missing length",
)

check(
    "world_state_change_count" in summary,
    "trajectory summary missing world_state_change_count",
)

check(
    "query_change_count" in summary,
    "trajectory summary missing query_change_count",
)

check(
    "state_change_count" not in summary,
    "Old state_change_count field still present",
)

print("Trajectory + gold: PASS")


# ============================================================
# 9. COUNTERFACTUAL PROBES
# ============================================================

banner("9. COUNTERFACTUAL PROBES")

probes = build_counterfactual_probes(
    rng=random.Random(42),
    ops_applied=ops,
    containers=containers,
    query=query,
    max_probes=2,
)

check(
    len(probes) <= 2,
    "Too many counterfactual probes returned",
)

for probe in probes:

    required = {
        "remove_step",
        "removed_operation",
        "original_answer",
        "counterfactual_answer",
        "answer_changed",
    }

    missing = required - set(probe)

    check(
        not missing,
        f"Counterfactual missing fields: {missing}",
    )

    idx = probe["remove_step"]

    check(
        0 <= idx < len(ops),
        f"Invalid counterfactual index: {idx}",
    )

    check(
        probe["answer_changed"]
        == (
            probe["original_answer"]
            != probe["counterfactual_answer"]
        ),
        "answer_changed is inconsistent with answers",
    )

    check(
        probe["original_answer"]
        == query.read(final_state),
        "Counterfactual original answer disagrees with final gold",
    )

print("Counterfactual probes: PASS")


# ============================================================
# 10. FULL PIPELINE GENERATION
# ============================================================

banner("10. FULL PIPELINE")

record = generate_example(
    rng=random.Random(2026),
    example_id="smoke-basic",
    entity_count=4,
    update_count=8,
    distractor_count=3,
    operations_enabled=[
        Put,
        Move,
        Remove,
    ],
    query_spec=QuerySpec(
        query_type="location",
        must_change_from_initial=True,
        min_relevant_steps=1,
        min_state_changes=1,
    ),
    include_counterfactual=True,
)

check(
    record["id"] == "smoke-basic",
    "Record ID incorrect",
)

check(
    len(record["operations"])
    == 8,
    "Pipeline operation count incorrect",
)

check(
    len(record["trajectory"])
    == len(record["operations"]),
    "Pipeline trajectory length mismatch",
)

check(
    len(record["step_wise_gold"])
    == len(record["operations"]),
    "Pipeline step-wise gold length mismatch",
)

check(
    record["gold_answer"]
    == record["step_wise_gold"][-1],
    "Pipeline gold answer mismatch",
)

check(
    "trajectory_summary" in record,
    "Missing trajectory_summary",
)

summary = record["trajectory_summary"]

check(
    "world_state_change_count" in summary,
    "Missing world_state_change_count",
)

check(
    "query_change_count" in summary,
    "Missing query_change_count",
)

check(
    "state_change_count" not in summary,
    "Deprecated state_change_count remains",
)

check(
    "counterfactual_probes" in record,
    "Missing counterfactual_probes",
)

for probe in record["counterfactual_probes"]:

    check(
        "original_answer" in probe,
        "Missing original_answer",
    )

    check(
        "counterfactual_answer" in probe,
        "Missing counterfactual_answer",
    )

    check(
        "answer_changed" in probe,
        "Missing answer_changed",
    )

    check(
        probe["answer_changed"]
        == (
            probe["original_answer"]
            != probe["counterfactual_answer"]
        ),
        "Counterfactual answer_changed inconsistency",
    )

print("Full pipeline: PASS")


# ============================================================
# 11. REDO-VALIDITY PROBE
# ============================================================

banner("11. REDO-VALIDITY PROBE")

redo_result = generate_example(
    rng=random.Random(9090),
    example_id="smoke-redo",
    entity_count=4,
    update_count=8,
    distractor_count=2,
    operations_enabled=[
        Put,
        Move,
        Undo,
        Redo,
    ],
    force_redo_probe=True,
)

check(
    redo_result["query"]["type"]
    == "redo_validity",
    "Redo probe has wrong query type",
)

check(
    redo_result["gold_answer"] is False,
    "Redo invalidation probe should have gold=False",
)

check(
    redo_result["counterfactual_probes"] == [],
    "Redo probe should not have normal counterfactual probes",
)

check(
    len(redo_result["trajectory"])
    == len(redo_result["operations"]),
    "Redo trajectory length mismatch",
)

print("Redo-validity probe: PASS")


# ============================================================
# 12. DETERMINISM CHECK
# ============================================================

banner("12. DETERMINISM")

kwargs = dict(
    example_id="determinism-test",
    entity_count=4,
    update_count=8,
    distractor_count=2,
    operations_enabled=[
        Put,
        Move,
        Remove,
    ],
    query_spec=QuerySpec(
        query_type="location",
        must_change_from_initial=True,
        min_relevant_steps=1,
        min_state_changes=1,
    ),
    include_counterfactual=True,
)

record_a = generate_example(
    random.Random(123456),
    **kwargs,
)

record_b = generate_example(
    random.Random(123456),
    **kwargs,
)

check(
    record_a == record_b,
    "Same seed produced different benchmark records",
)

print("Same seed -> identical record: PASS")


# ============================================================
# FINAL
# ============================================================

banner("ALL SMOKE TESTS PASSED")

print("World model ............... PASS")
print("Replay engine ............. PASS")
print("Sampler ................... PASS")
print("Query selection ........... PASS")
print("QuerySpec filtering ....... PASS")
print("Trajectory/gold ........... PASS")
print("Counterfactual probes ..... PASS")
print("Redo validity ............. PASS")
print("Pipeline integration ...... PASS")
print("Determinism ............... PASS")

print()
print("DWS-Bench generator is internally consistent for the tested paths.")

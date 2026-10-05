"""Make sibling packages (world, generator, render, analysis, eval) importable
from the repo root regardless of where pytest is launched."""
import contextlib
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Shared helpers for the checklist test files (test/test_1_*.py .. test_12_*.py)
#
# These are helpers, not fixtures: every checklist file imports them by name so
# the suite has one definition of "a generated instance" instead of twelve
# copies that drift. All of them are offline and deterministic.
# ---------------------------------------------------------------------------

from world import (  # noqa: E402  (import after sys.path fix)
    Merge,
    Move,
    Put,
    Redo,
    Remove,
    Split,
    Swap,
    Undo,
)
from generator import (  # noqa: E402  (import after sys.path fix)
    TrajectorySpec,
    build_trajectory,
    build_validated_instance,
    reset_deduplication_registry,
)

# Default seed sweeps. Property-style tests loop over these instead of relying
# on hypothesis (not installed, and not needed: the pipeline is seeded).
SEEDS_SMALL = range(10)
SEEDS_MEDIUM = range(50)
SEEDS_LARGE = range(200)

# query_type per family, mirroring experiments/rq1_mutation_depth.py
# FAMILY_QUERY_TYPES. Section 2/3 tests use it so a "grid cell" means the same
# thing in tests as in the experiment scripts.
FAMILY_QUERY_TYPES = {
    "basic_chain": "location",
    "interleaved_chain": "location",
    "revision": "location",
    "split_chain": "count",
    "merge_chain": "count",
    "swap_chain": "count",
    "undo_chain": "count",
    "undo_redo_chain": "location",
}


@contextlib.contextmanager
def quiet():
    """Silence generator/experiment chatter so pytest output stays readable."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def make_spec(
    family,
    entity_count=1,
    target_updates=4,
    distractor_updates=0,
    num_containers=3,
    query_type=None,
):
    """A TrajectorySpec with total_updates consistent (SPEC §2)."""
    return TrajectorySpec(
        family=family,
        entity_count=entity_count,
        num_containers=num_containers,
        total_updates=target_updates + distractor_updates,
        target_updates=target_updates,
        distractor_updates=distractor_updates,
        query_type=query_type or FAMILY_QUERY_TYPES.get(family, "location"),
    )


def build_record(
    family,
    seed,
    entity_count=1,
    target_updates=4,
    distractor_updates=0,
    num_containers=3,
    textual_distractors=0,
    query_type=None,
    instance_id=None,
    max_attempts=10,
):
    """
    One validated instance record, or None if the gate rejected every attempt.

    The deduplication registry is cleared per call: a seed sweep wants
    independent instances, not cross-seed duplicate rejections.
    """
    reset_deduplication_registry()
    spec = make_spec(
        family,
        entity_count=entity_count,
        target_updates=target_updates,
        distractor_updates=distractor_updates,
        num_containers=num_containers,
        query_type=query_type,
    )
    iid = instance_id or f"{family}_s{seed}"
    for attempt in range(max_attempts):
        result = build_validated_instance(
            random.Random(seed * 1000 + attempt),
            spec,
            random.Random(seed * 2000 + attempt + 7),
            textual_distractors,
            instance_id=iid,
            experiment="checklist",
            condition_id=f"{family}_T{target_updates}_D{distractor_updates}",
            seed=seed,
            attempt=attempt,
        )
        if result.ok:
            return result.record
    return None


def build_trajectory_for(family, seed, **kwargs):
    """build_trajectory() with a quiet spec factory; raises like the builder."""
    return build_trajectory(random.Random(seed), make_spec(family, **kwargs))


def gold_container_ids(record):
    """Container ids visited by the target, in order, from the canonical trace.

    Uses symbolic ids, not display names: display names are randomised per
    instance, so any balance/shortcut statistic computed on them is noise.
    """
    target = record["query_entity"]
    ids = []
    loc = {}
    for step in record["canonical_trace"]:
        op = step["op_type"]
        if op == "PUT" and step.get("obj_id") == target:
            loc[target] = step["container"]
            ids.append(step["container"])
        elif op == "MOVE" and step.get("obj_id") == target:
            loc[target] = step["dst"]
            ids.append(step["dst"])
        elif op == "REMOVE" and step.get("obj_id") == target:
            loc.pop(target, None)
            ids.append(None)
    return ids


def target_operations(record):
    """Target-affecting op signatures (op_type + endpoint), in order."""
    target = record["query_entity"]
    out = []
    for step in record["canonical_trace"]:
        op = step["op_type"]
        if op == "MOVE" and step.get("obj_id") == target:
            out.append((op, step.get("dst")))
        elif op == "PUT" and step.get("obj_id") == target:
            out.append((op, step.get("container")))
    return out


def container_count(record):
    return len(record["final_state"]["containers"])


def containers_of(record):
    """Container set of a record, from the stored final state."""
    return set(record["final_state"]["containers"])


# Serialised trace -> Operation objects. Mirrors generator.instance._canonical_trace
# (the frozen attribute order documented there); tests use it to replay a stored
# record instead of trusting the stored final state.
_OP_FIELDS = {
    "PUT": ("obj_id", "obj_type", "container"),
    "MOVE": ("obj_id", "dst"),
    "REMOVE": ("obj_id",),
    "UNDO": (),
    "REDO": (),
    "SWAP": ("container_a", "container_b"),
    "MERGE": ("src_container", "dst_container"),
    "SPLIT": ("source_obj_id", "new_obj_id"),
}
_OP_CLASSES = {
    "PUT": Put,
    "MOVE": Move,
    "REMOVE": Remove,
    "UNDO": Undo,
    "REDO": Redo,
    "SWAP": Swap,
    "MERGE": Merge,
    "SPLIT": Split,
}


def ops_from_record(record):
    """Rebuild Operation objects from ``record['canonical_trace']``."""
    out = []
    for step in record["canonical_trace"]:
        op_type = step["op_type"]
        assert op_type in _OP_CLASSES, f"unknown op in canonical trace: {op_type}"
        fields = _OP_FIELDS[op_type]
        out.append(_OP_CLASSES[op_type](*(step[f] for f in fields)))
    return out


def replay_record(record):
    """The single shared replay pass: returns (trace, final_state, history)."""
    from world import replay_trace

    return replay_trace(ops_from_record(record), containers_of(record))

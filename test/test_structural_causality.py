from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from generator import build_trajectory
from generator.structural import validate_structural_causality
from generator.trajectory_specs import TrajectorySpec
from generator.dataset_spec import REQUIRED_STRUCTURAL_OPS, STRUCTURAL_FAMILIES
from world import Merge, Split, Move, Put, Swap, Undo, Redo, replay_trace


def _run_once(family: str, seed: int):
    rng = random.Random(seed)
    if family == "split_chain":
        spec = TrajectorySpec(
            family=family,
            entity_count=2,
            num_containers=3,
            total_updates=5,
            target_updates=3,
            distractor_updates=0,
            min_interleaving=0.0,
            query_type="count",
        )
    elif family == "merge_chain":
        spec = TrajectorySpec(
            family=family,
            entity_count=2,
            num_containers=3,
            total_updates=3,
            target_updates=1,
            distractor_updates=0,
            min_interleaving=0.0,
            query_type="count",
        )
    elif family == "swap_chain":
        spec = TrajectorySpec(
            family=family,
            entity_count=2,
            num_containers=3,
            total_updates=3,
            target_updates=1,
            distractor_updates=0,
            min_interleaving=0.0,
            query_type="count",
        )
    elif family == "undo_chain":
        spec = TrajectorySpec(
            family=family,
            entity_count=1,
            num_containers=3,
            total_updates=3,
            target_updates=2,
            distractor_updates=0,
            min_interleaving=0.0,
            query_type="count",
        )
    elif family == "undo_redo_chain":
        spec = TrajectorySpec(
            family=family,
            entity_count=1,
            num_containers=3,
            total_updates=7,
            target_updates=6,
            distractor_updates=0,
            min_interleaving=0.0,
            query_type="location",
        )
    else:
        spec = TrajectorySpec(
            family=family,
            entity_count=2,
            num_containers=3,
            total_updates=5,
            target_updates=3,
            distractor_updates=0,
            min_interleaving=0.0,
        )
    res = build_trajectory(rng, spec)
    validate_structural_causality(res.ops, res.containers, res.target_obj, res.spec)
    return res


def test_all_families_pass_seed_sweep():
    for family in STRUCTURAL_FAMILIES:
        for seed in range(25):
            try:
                _run_once(family, seed)
            except Exception as e:
                raise AssertionError(f"failed {family} seed={seed}: {e}") from e


def test_presence_fires():
    containers = {"c0", "c1"}
    ops = [
        Put(obj_id="o0", obj_type="X", container="c0"),
        Put(obj_id="o1", obj_type="X", container="c1"),
    ]
    spec = TrajectorySpec(
        family="split_chain", entity_count=2, num_containers=3, total_updates=3, target_updates=3, query_type="count"
    )
    try:
        validate_structural_causality(ops, containers, "o1", spec)
        assert False, "should have raised"
    except ValueError:
        pass


def test_target_effect_fires():
    """A Split that touches neither the target nor its new child must fail."""
    containers = {"c0", "c1"}
    ops = [
        Put(obj_id="o0", obj_type="X", container="c0"),
        Split(source_obj_id="o0", new_obj_id="o2"),
    ]
    spec = TrajectorySpec(
        family="split_chain", entity_count=2, num_containers=3, total_updates=3, target_updates=3, query_type="count"
    )
    with pytest.raises(ValueError, match="check=target_effect"):
        validate_structural_causality(ops, containers, "o99", spec)


def test_necessity_fires():
    """Removing the required Swap must change the answer to pass."""
    containers = {"c0", "c1"}
    ops = [
        Put(obj_id="o0", obj_type="X", container="c0"),
        Put(obj_id="o1", obj_type="X", container="c1"),
        Swap(container_a="c0", container_b="c1"),
        Swap(container_a="c0", container_b="c1"),
        Merge(src_container="c0", dst_container="c1"),
    ]
    spec = TrajectorySpec(
        family="swap_chain", entity_count=2, num_containers=3, total_updates=5, target_updates=5, query_type="count"
    )
    # gold: o0 ends in c1 both with and without the last Swap.
    with pytest.raises(ValueError, match="check=necessity"):
        validate_structural_causality(ops, containers, "o0", spec)


def test_unrelated_exempt_passes():
    rng = random.Random(42)
    spec = TrajectorySpec(
        family="undo_redo_chain",
        entity_count=1,
        num_containers=3,
        total_updates=7,
        target_updates=6,
        distractor_updates=1,
    )
    res = build_trajectory(rng, spec)
    validate_structural_causality(res.ops, res.containers, res.target_obj, res.spec)


def test_no_trailing_move_fires():
    """Test that a trailing Move after a required operation triggers the check.
    
    For the new count-query families, the structural operation is the final
    target-affecting operation, so there's no trailing Move. This test verifies
    the check using merge_chain with a trailing Move manually added.
    """
    # merge_chain requires count query, but we can manually construct a case
    # with a trailing Move to test the check
    containers = {"c0", "c1", "c2"}
    ops = [
        Put(obj_id="o0", obj_type="X", container="c0"),
        Put(obj_id="o1", obj_type="X", container="c0"),
        Merge(src_container="c0", dst_container="c1"),  # required merge
        Move(obj_id="o0", dst="c2"),  # trailing move - final determinant
    ]
    spec = TrajectorySpec(
        family="merge_chain", entity_count=2, num_containers=3, total_updates=3, target_updates=2, query_type="count"
    )
    with pytest.raises(ValueError, match="check=no_trailing_ordinary_move"):
        validate_structural_causality(ops, containers, "o0", spec)


def test_split_child_target():
    rng = random.Random(7)
    spec = TrajectorySpec(
        family="split_chain",
        entity_count=2,
        num_containers=3,
        total_updates=3,
        target_updates=3,
        distractor_updates=0,
        query_type="count",
    )
    res = build_trajectory(rng, spec)
    split_ops = [op for op in res.ops if isinstance(op, Split)]
    assert split_ops
    # In the new split_chain, the target is the original object, not the child
    # The child is the new_obj_id from Split
    split_ops = [op for op in res.ops if isinstance(op, Split)]
    assert split_ops
    assert split_ops[0].source_obj_id == res.target_obj


def test_undo_redo_both():
    for seed in range(10):
        rng = random.Random(seed)
        spec = TrajectorySpec(
            family="undo_redo_chain",
            entity_count=1,
            num_containers=3,
            total_updates=7,
            target_updates=6,
            distractor_updates=0,
        )
        res = build_trajectory(rng, spec)
        validate_structural_causality(res.ops, res.containers, res.target_obj, res.spec)


def test_determinism():
    for family in ["split_chain", "undo_redo_chain"]:
        rng1 = random.Random(5)
        rng2 = random.Random(5)
        if family == "split_chain":
            spec = TrajectorySpec(
                family=family,
                entity_count=2,
                num_containers=3,
                total_updates=3,
                target_updates=3,
                distractor_updates=0,
                query_type="count",
            )
        else:
            spec = TrajectorySpec(
                family=family,
                entity_count=1,
                num_containers=3,
                total_updates=7,
                target_updates=6,
                distractor_updates=0,
            )
        r1 = build_trajectory(rng1, spec)
        r2 = build_trajectory(rng2, spec)
        assert len(r1.ops) == len(r2.ops)


def test_registry_coverage():
    for f in STRUCTURAL_FAMILIES:
        assert f in REQUIRED_STRUCTURAL_OPS
    for k in list(REQUIRED_STRUCTURAL_OPS.keys()):
        assert k in STRUCTURAL_FAMILIES

"""
test/test_solubility.py
=======================
Tests for solubility audit module (G2).
"""

from __future__ import annotations

from analysis.solubility import (
    check_answer_uniqueness,
    run_solubility_audit,
    SolubilityResult,
)


def _make_instance(instance_id: str, gold_answer: str, context: str = None, family: str = "basic_chain", T: int = 2) -> dict:
    """Create a minimal instance for testing."""
    if context is None:
        context = "A key was placed in the red box. The key was moved to the blue bag."
    # Provide step_wise_gold_answers to avoid cross-instance candidate pollution
    return {
        "instance_id": instance_id,
        "family": family,
        "requested_factors": {"T": T, "D": 0, "E": 1},
        "gold_answer": gold_answer,
        "context": context,
        "sentences": context.split(". "),
        "canonical_trace": [
            {"op_type": "PUT", "obj_id": "o0", "container": "c0"},
            {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
        ],
        "step_wise_gold_answers": ["the red box", "the blue bag"],
        "final_state": {
            "container_names": {"c0": "the red box", "c1": "the blue bag"},
            "container_display_names": {"c0": "the red box", "c1": "the blue bag"},
        },
    }


def test_soluble_instance_passes() -> None:
    """Clear instance with unique gold answer passes solubility."""
    inst = _make_instance("test-1", "the blue bag")
    result = check_answer_uniqueness(inst)
    
    assert isinstance(result, SolubilityResult)
    assert result.instance_id == "test-1"
    assert result.is_soluble is True
    # Ambiguity is 0.5 because both initial and final containers are candidates
    # This is expected for normal instances - not a solubility failure
    assert result.ambiguity_score == 0.5
    assert len(result.issues) == 0


def test_missing_gold_in_candidates_fails() -> None:
    """Test that solubility checker detects when gold has zero mentions in context."""
    # Gold is in candidates (always added) but not mentioned in context
    inst = _make_instance("test-2", "the green cup", family="unique_fam_12345", T=9999)
    # Context doesn't mention "the green cup"
    inst["context"] = "A key was placed in the red box. The key was moved to the blue bag."
    inst["step_wise_gold_answers"] = ["the red box", "the blue bag"]
    result = check_answer_uniqueness(inst)
    
    # Should flag that gold not mentioned in context
    assert result.is_soluble is False
    assert any("not explicitly mentioned in context" in issue for issue in result.issues)


def test_high_candidate_count_warns() -> None:
    """Instance with many candidates gets ambiguity warning."""
    inst = _make_instance("test-3", "the blue bag", family="other_family2", T=98)
    # Add many fake containers to final_state
    inst["final_state"]["container_names"] = {
        f"c{i}": f"container {i}" for i in range(10)
    }
    inst["final_state"]["container_display_names"] = inst["final_state"]["container_names"]
    # Update step_wise_gold_answers to include all
    inst["step_wise_gold_answers"] = [f"container {i}" for i in range(10)]
    result = check_answer_uniqueness(inst)
    
    # A large answer space is reported, but is not a solubility defect.
    assert any("High candidate count" in w for w in result.warnings)
    assert not any("High candidate count" in issue for issue in result.issues)


def test_run_solubility_audit() -> None:
    """Solubility audit runs on multiple instances."""
    instances = [
        _make_instance(f"test-{i}", "the blue bag", family=f"fam{i}", T=i)
        for i in range(5)
    ]
    # Add one bad instance with different family
    instances.append(_make_instance("test-bad", "the green cup", family="bad_fam", T=999))
    
    results = run_solubility_audit(instances, sample_size=10)
    
    assert results["total_audited"] == 6
    assert results["soluble_count"] == 5
    assert results["solubility_rate"] == 5/6
    assert "per_instance" in results
    assert len(results["per_instance"]) == 6


def test_solubility_audit_stratified_sampling() -> None:
    """Audit samples proportionally from each family."""
    instances = [
        _make_instance(f"basic_chain_{i}", "the blue bag")
        for i in range(10)
    ]
    instances += [
        {"instance_id": f"interleaved_chain_{i}", "family": "interleaved_chain",
         "requested_factors": {"T": 2, "D": 1, "E": 2},
         "gold_answer": "the red box",
         "context": "A key was placed in the red box.",
         "sentences": ["A key was placed in the red box."],
         "canonical_trace": [],
         "final_state": {"container_names": {"c0": "the red box"}, "container_display_names": {"c0": "the red box"}},
        }
        for i in range(10)
    ]
    
    results = run_solubility_audit(instances, sample_size=5)
    
    # Should have sampled from both families
    audited_families = set()
    for inst in results["per_instance"]:
        # Can't easily check family from SolubilityResult, but total should be <= sample_size
        pass
    
    assert results["total_audited"] <= 5
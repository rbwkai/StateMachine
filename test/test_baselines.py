"""
test/test_baselines.py
======================
Tests for baseline predictors.
"""

from __future__ import annotations

from eval.baselines import (
    compute_stateless_baseline,
    compute_mfc_baseline,
    run_all_baselines,
)


def _make_instance(
    instance_id: str,
    family: str,
    T: int,
    D: int,
    gold_answer: str,
    canonical_trace: list = None,
) -> dict:
    """Create a minimal instance dict for testing."""
    if canonical_trace is None:
        canonical_trace = [
            {"op_type": "PUT", "obj_id": "o0", "container": "c0"},
            {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
        ]
    return {
        "instance_id": instance_id,
        "family": family,
        "requested_factors": {"T": T, "D": D, "E": 1},
        "gold_answer": gold_answer,
        "canonical_trace": canonical_trace,
        "final_state": {
            "container_names": {"c0": "the red box", "c1": "the blue bag"},
            "container_display_names": {"c0": "the red box", "c1": "the blue bag"},
        },
    }


def test_stateless_baseline_uses_initial_location() -> None:
    """Stateless baseline predicts initial location from first Put."""
    instances = [
        _make_instance("test-1", "basic_chain", 2, 0, "the blue bag"),
        _make_instance("test-2", "basic_chain", 2, 0, "the red box"),
    ]
    
    results = compute_stateless_baseline(instances)
    
    assert len(results) == 2
    # First instance: initial location is c0 -> "the red box"
    # But gold is "the blue bag" -> incorrect
    # Second instance: same initial -> "the red box" -> correct
    assert results[0].baseline_type == "stateless"
    assert results[1].baseline_type == "stateless"


def test_mfc_baseline_predicts_most_common_per_condition() -> None:
    """MFC baseline predicts most frequent answer per condition."""
    # Condition 1: two instances, gold answers "A" and "A" -> MFC = "A"
    # Condition 2: two instances, gold answers "B" and "A" -> MFC = "B" (first most common)
    instances = [
        _make_instance("c1_0", "basic_chain", 2, 0, "the red box"),
        _make_instance("c1_1", "basic_chain", 2, 0, "the red box"),
        _make_instance("c2_0", "interleaved_chain", 3, 1, "the blue bag"),
        _make_instance("c2_1", "interleaved_chain", 3, 1, "the red box"),
    ]
    
    results = compute_mfc_baseline(instances)
    
    assert len(results) == 4
    # Condition 1 instances should predict "the red box" (correct for both)
    assert results[0].is_correct is True
    assert results[1].is_correct is True
    # Condition 2 instances should predict "the blue bag" (correct for first, wrong for second)
    assert results[2].is_correct is True
    assert results[3].is_correct is False


def test_run_all_baselines_returns_both() -> None:
    """run_all_baselines returns stateless and MFC summaries."""
    instances = [
        _make_instance("test-1", "basic_chain", 2, 0, "the red box"),
        _make_instance("test-2", "basic_chain", 2, 0, "the blue bag"),
    ]
    
    summaries = run_all_baselines(instances)
    
    assert "stateless" in summaries
    assert "mfc" in summaries
    assert "overall" in summaries["stateless"]
    assert "overall" in summaries["mfc"]
    assert "per_condition" in summaries["stateless"]
    assert "per_condition" in summaries["mfc"]


def test_baseline_summaries_have_correct_structure() -> None:
    """Baseline summary has expected keys."""
    instances = [
        _make_instance("test-1", "basic_chain", 2, 0, "the red box"),
    ]
    
    from eval.baselines import summarize_baselines
    
    results = compute_stateless_baseline(instances)
    summary = summarize_baselines(results)
    
    assert "total" in summary
    assert "correct" in summary
    assert "accuracy" in summary
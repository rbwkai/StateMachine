"""
test/test_robustness.py
=======================
Tests for prompt sensitivity and paraphrase robustness (E4, E6).
"""

from __future__ import annotations

from eval.robustness import (
    build_prompt_variants,
    paraphrase_narrative,
    evaluate_prompt_sensitivity,
    evaluate_paraphrase_robustness,
    run_robustness_suite,
    RobustnessResult,
)


def _make_instance() -> dict:
    """Create a minimal instance for testing."""
    return {
        "instance_id": "test-1",
        "family": "basic_chain",
        "requested_factors": {"T": 2, "D": 0, "E": 1},
        "gold_answer": "the blue bag",
        "context": "A key was placed in the red box. The key was moved to the blue bag.",
        "question": "Where is the key now?",
        "sentences": [
            "A key was placed in the red box.",
            "The key was moved to the blue bag.",
        ],
        "canonical_trace": [
            {"op_type": "PUT", "obj_id": "o0", "container": "c0"},
            {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
        ],
        "final_state": {
            "container_names": {"c0": "the red box", "c1": "the blue bag"},
            "container_display_names": {"c0": "the red box", "c1": "the blue bag"},
        },
    }


def test_build_prompt_variants() -> None:
    """Prompt variants are generated for all three templates."""
    inst = _make_instance()
    variants = build_prompt_variants(inst["context"], inst["question"])
    
    assert len(variants) == 3
    assert "v1_original" in variants
    assert "v2_standard" in variants
    assert "v3_minimal" in variants
    
    # Each variant should contain the context and question
    for v in variants.values():
        assert "key" in v.lower()
        assert "where is" in v.lower() or "Where is" in v


def test_paraphrase_narrative() -> None:
    """Paraphrase transforms narrative sentences."""
    inst = _make_instance()
    paraphrased = paraphrase_narrative(inst["sentences"])
    
    assert len(paraphrased) == 2
    # Check that paraphrasing changed something
    assert paraphrased != inst["sentences"]
    # Check specific transformations
    assert "put into" in paraphrased[0].lower() or "placed in" in paraphrased[0].lower()
    assert "goes from" in paraphrased[1].lower() or "moved" in paraphrased[1].lower()


def test_evaluate_prompt_sensitivity() -> None:
    """Prompt sensitivity evaluation runs without error."""
    inst = _make_instance()
    
    # Mock predict_fn that always returns correct answer
    def mock_predict(prompt: str) -> str:
        return "Final Answer: the blue bag"
    
    result = evaluate_prompt_sensitivity(inst, mock_predict)
    
    assert isinstance(result, RobustnessResult)
    assert result.instance_id == "test-1"
    assert result.baseline_accuracy == 1.0
    assert len(result.variant_accuracies) == 3
    assert all(v == 1.0 for v in result.variant_accuracies.values())
    assert result.gap == 0.0
    assert result.is_robust is True


def test_evaluate_paraphrase_robustness() -> None:
    """Paraphrase robustness evaluation runs without error."""
    inst = _make_instance()
    
    def mock_predict(prompt: str) -> str:
        return "Final Answer: the blue bag"
    
    result = evaluate_paraphrase_robustness(inst, mock_predict)
    
    assert isinstance(result, RobustnessResult)
    assert result.instance_id == "test-1"
    assert result.baseline_accuracy == 1.0
    assert "original" in result.variant_accuracies
    assert "paraphrased" in result.variant_accuracies
    assert result.gap == 0.0
    assert result.is_robust is True


def test_run_robustness_suite() -> None:
    """Full robustness suite runs and aggregates correctly."""
    instances = [_make_instance(), _make_instance()]
    instances[1]["instance_id"] = "test-2"
    
    def mock_predict(prompt: str) -> str:
        return "Final Answer: the blue bag"
    
    results = run_robustness_suite(instances, mock_predict)
    
    assert "prompt_sensitivity" in results
    assert "paraphrase_robustness" in results
    assert "per_instance_prompt" in results
    assert "per_instance_paraphrase" in results
    
    # Check aggregated metrics
    for key in ["prompt_sensitivity", "paraphrase_robustness"]:
        agg = results[key]
        assert "mean_gap" in agg
        assert "robust_rate" in agg
        assert "max_gap" in agg
        assert agg["mean_gap"] == 0.0
        assert agg["robust_rate"] == 1.0


def test_robustness_detects_sensitivity() -> None:
    """Robustness detects when model is sensitive to prompt changes."""
    inst = _make_instance()
    
    # Mock predict_fn that fails on v3_minimal
    def sensitive_predict(prompt: str) -> str:
        if "v3_minimal" in prompt or "Answer the question based on the narrative" in prompt:
            return "Final Answer: the red box"  # Wrong
        return "Final Answer: the blue bag"
    
    result = evaluate_prompt_sensitivity(inst, sensitive_predict)
    
    assert result.gap > 0.0
    assert result.is_robust is False
    assert result.variant_accuracies["v3_minimal"] == 0.0
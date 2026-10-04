"""
test/test_extraction_diagnostics.py
===================================
Tests for extraction vs reasoning failure separation (E2).
"""

from __future__ import annotations

from eval.eval_harness import evaluate_predictions


def _make_instance(instance_id: str, gold_answer: str) -> dict:
    """Create a minimal instance dict for testing."""
    return {
        "instance_id": instance_id,
        "family": "basic_chain",
        "requested_factors": {"T": 2, "D": 0, "E": 1},
        "gold_answer": gold_answer,
        "canonical_trace": [
            {"op_type": "PUT", "obj_id": "o0", "container": "c0"},
            {"op_type": "MOVE", "obj_id": "o0", "dst": "c1"},
        ],
        "step_wise_gold": ["c0", "c1"],
        "step_wise_gold_answers": ["the red box", "the blue bag"],
        "final_state": {
            "container_names": {"c0": "the red box", "c1": "the blue bag"},
            "container_display_names": {"c0": "the red box", "c1": "the blue bag"},
        },
    }


def test_extraction_diagnostics_in_results() -> None:
    """Evaluation results include extraction diagnostics."""
    instances = [
        _make_instance("test-1", "the blue bag"),
        _make_instance("test-2", "the blue bag"),
    ]
    
    # Prediction with proper Final Answer format
    predictions = [
        {"instance_id": "test-1", "raw_prediction": "Step 1: the red box\nFinal Answer: the blue bag"},
        {"instance_id": "test-2", "raw_prediction": "Final Answer: the blue bag"},
    ]
    
    result = evaluate_predictions(instances, predictions, chain_of_thought=True)
    
    # Check overall extraction diagnostics
    assert "overall_format_compliance_rate" in result
    assert "overall_extraction_rate" in result
    assert "overall_semantic_accuracy" in result
    
    # Both predictions have Final Answer
    assert result["overall_format_compliance_rate"] == 1.0
    # Both should extract correctly
    assert result["overall_extraction_rate"] == 1.0
    # Both semantically correct
    assert result["overall_semantic_accuracy"] == 1.0
    
    # Check per-condition summaries
    for cond_summary in result["condition_summaries"].values():
        assert "format_compliance_rate" in cond_summary
        assert "extraction_rate" in cond_summary
        assert "semantic_accuracy" in cond_summary


def test_extraction_diagnostics_separates_format_from_reasoning() -> None:
    """Extraction diagnostics separate format failure from reasoning failure."""
    instances = [
        _make_instance("test-1", "the blue bag"),
        _make_instance("test-2", "the blue bag"),
    ]
    
    # test-1: has Final Answer but wrong answer (reasoning failure)
    # test-2: no Final Answer, no extraction (format failure)
    predictions = [
        {"instance_id": "test-1", "raw_prediction": "Final Answer: the red box"},
        {"instance_id": "test-2", "raw_prediction": "I think it's in the box."},
    ]
    
    result = evaluate_predictions(instances, predictions, chain_of_thought=False)
    
    # test-1: format compliant, extracted, but semantically wrong
    # test-2: not format compliant, not extracted, semantically wrong
    assert result["overall_format_compliance_rate"] == 0.5  # 1/2
    assert result["overall_extraction_rate"] == 0.5  # 1/2
    assert result["overall_semantic_accuracy"] == 0.0  # 0/2
    assert result["overall_accuracy"] == 0.0  # 0/2 strict correct


def test_condition_summaries_include_extraction_rates() -> None:
    """Condition summaries include extraction diagnostics."""
    instances = [
        _make_instance("basic_chain_T2_D0_E1_i000", "the blue bag"),
    ]
    
    predictions = [
        {"instance_id": "basic_chain_T2_D0_E1_i000", "raw_prediction": "Final Answer: the blue bag"},
    ]
    
    result = evaluate_predictions(instances, predictions)
    
    cond_key = "basic_chain_T2_D0"
    assert cond_key in result["condition_summaries"]
    cond = result["condition_summaries"][cond_key]
    
    assert cond["format_compliance_rate"] == 1.0
    assert cond["extraction_rate"] == 1.0
    assert cond["semantic_accuracy"] == 1.0
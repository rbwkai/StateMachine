"""
test/test_eval_pipeline.py
==========================
Unit and integration tests for the SLM evaluation pipeline.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.engine import MockInferenceEngine
from eval.eval_harness import extract_answer, format_prompt
from eval.models import CORE_MODELS, OPTIONAL_MODELS
from run_eval import run_evaluation


def test_model_registry() -> None:
    print("Testing Model Registry...")
    expected_core = {"qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b", "llama-3.2-3b", "olmo-2-1b"}
    assert set(CORE_MODELS.keys()) == expected_core, f"Expected {expected_core}, got {set(CORE_MODELS.keys())}"

    assert "phi-4-mini" in OPTIONAL_MODELS
    print("  [PASS] All 5 core models and optional models properly configured.")


def test_prompt_formatting_and_extraction() -> None:
    print("Testing Prompt Formatting and Answer Extraction...")
    prompt = format_prompt(
        context="A key was placed in the box. The key was moved to the basket.",
        question="Where is the key now?",
        system_prompt="Be concise.",
        chain_of_thought=False,
    )
    assert "Instructions: Be concise." in prompt
    assert "Narrative:" in prompt
    assert "Question:" in prompt

    # Extraction tests
    containers = ["the box", "the basket", "the drawer"]
    assert extract_answer("Answer: the basket", candidate_containers=containers) == "the basket"
    assert extract_answer("Based on the story, the object is in the basket.", candidate_containers=containers) == "the basket"
    assert extract_answer("True") == "True"
    assert extract_answer("False") == "False"
    print("  [PASS] Prompt formatting and answer extraction verified.")


def test_end_to_end_mock_eval() -> None:
    print("Testing End-to-End Mock Evaluation Flow...")
    temp_dir = Path(tempfile.mkdtemp(prefix="dws_eval_test_"))
    try:
        sample_records = [
            {
                "instance_id": "test_001",
                "family": "basic_chain",
                "experiment": "rq1_depth",
                "requested_factors": {"T": 2, "E": 1, "D": 0},
                "measured_factors": {"T_actual": 2, "E_actual": 1, "D_actual": 0, "V_actual": 0, "L_actual": 20},
                "context": "A key was put in the green box. The key was moved to the large cabinet.",
                "question": "Where is the key now?",
                "gold_answer": "the large cabinet",
                "gold_container": "c2",
                "final_state": {"containers": ["the green box", "the large cabinet"]},
            },
            {
                "instance_id": "test_002",
                "family": "basic_chain",
                "experiment": "rq1_depth",
                "requested_factors": {"T": 4, "E": 1, "D": 0},
                "measured_factors": {"T_actual": 4, "E_actual": 1, "D_actual": 0, "V_actual": 0, "L_actual": 35},
                "context": "A ball was placed in the green box.",
                "question": "Where is the ball now?",
                "gold_answer": "the green box",
                "gold_container": "c1",
                "final_state": {"containers": ["the green box", "the blue bin"]},
            },
        ]

        metrics = run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"],
            dataset_records=sample_records,
            dataset_name="test_dataset",
            output_dir=temp_dir,
            mock=True,
        )

        assert metrics["total_instances"] == 2
        assert "overall_accuracy" in metrics
        assert "family_accuracies" in metrics

        # Verify output files exist
        model_out = temp_dir / "qwen2.5-0.5b"
        assert (model_out / "test_dataset_predictions.jsonl").exists()
        assert (model_out / "test_dataset_metrics.json").exists()
        assert (model_out / "test_dataset_report.md").exists()

        print("  [PASS] End-to-end evaluation, metrics, and report generation verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    print("=" * 70)
    print("SLM EVALUATION PIPELINE TEST SUITE")
    print("=" * 70)
    test_model_registry()
    test_prompt_formatting_and_extraction()
    test_end_to_end_mock_eval()
    print("=" * 70)
    print("ALL EVAL PIPELINE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

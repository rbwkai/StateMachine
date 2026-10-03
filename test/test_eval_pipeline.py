"""
test/test_eval_pipeline.py
==========================
Unit and integration tests for the SLM evaluation pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.engine import MockInferenceEngine
from eval.eval_harness import extract_answer, format_prompt
from eval.models import CORE_MODELS, OPTIONAL_MODELS
from eval.prompts import build_user_prompt
from run_eval import run_evaluation


def test_model_registry() -> None:
    print("Testing Model Registry...")
    expected_core = {"qwen2.5-0.5b", "qwen2.5-3b", "qwen2.5-7b", "llama-3.2-3b", "olmo-2-1b"}
    assert set(CORE_MODELS.keys()) == expected_core, f"Expected {expected_core}, got {set(CORE_MODELS.keys())}"

    assert "phi-4-mini" in OPTIONAL_MODELS
    print("  [PASS] All 5 core models and optional models properly configured.")


def test_prompt_formatting_and_unification() -> None:
    print("Testing Prompt Formatting, Unification, and Versioning...")

    # v2 CoT prompt check
    prompt_v2_cot = build_user_prompt(
        context="A key was placed in the box.",
        question="Where is the key now?",
        chain_of_thought=True,
        prompt_version="v2",
    )
    assert "Step 3" not in prompt_v2_cot
    assert prompt_v2_cot.count("<container>") == 0
    assert "Step k:" in prompt_v2_cot
    assert "Final Answer: <name of the container>" in prompt_v2_cot

    # v1 CoT prompt check (backward compatibility)
    prompt_v1_cot = build_user_prompt(
        context="A key was placed in the box.",
        question="Where is the key now?",
        chain_of_thought=True,
        prompt_version="v1",
    )
    assert "Step 3: <container>" in prompt_v1_cot

    # Unification check: build_user_prompt output is identical from engine and harness
    harness_prompt = format_prompt(
        context="A key was placed in the box.",
        question="Where is the key now?",
        chain_of_thought=True,
        prompt_version="v2",
    )
    engine = MockInferenceEngine()
    engine_prompt = engine.format_input(
        context="A key was placed in the box.",
        question="Where is the key now?",
        chain_of_thought=True,
        prompt_version="v2",
    )
    assert harness_prompt == prompt_v2_cot
    assert engine_prompt == prompt_v2_cot

    # Answer extraction tests
    containers = ["the box", "the basket", "the drawer"]
    assert extract_answer("Final Answer: the basket", candidate_containers=containers) == "the basket"
    assert extract_answer("The basket is mentioned. Final Answer: the drawer", candidate_containers=containers) == "the drawer"
    assert extract_answer("The answer is the basket.", candidate_containers=containers) == ""
    assert extract_answer("Final Answer: True") == "True"
    assert extract_answer("Final Answer: False") == "False"
    print("  [PASS] Prompt formatting, unification, and extraction verified.")


def test_end_to_end_mock_eval_and_layout() -> None:
    print("Testing End-to-End Mock Evaluation Flow and Directory Layout...")
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

        # 1. CoT run @ 128
        m_cot_128 = run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"],
            dataset_records=sample_records,
            dataset_name="test_dataset",
            output_dir=temp_dir,
            chain_of_thought=True,
            max_new_tokens=128,
            mock=True,
        )

        # 2. Non-CoT run @ 32
        m_nocot_32 = run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"],
            dataset_records=sample_records,
            dataset_name="test_dataset",
            output_dir=temp_dir,
            chain_of_thought=False,
            max_new_tokens=32,
            mock=True,
        )

        # 3. CoT run @ 256
        m_cot_256 = run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"],
            dataset_records=sample_records,
            dataset_name="test_dataset",
            output_dir=temp_dir,
            chain_of_thought=True,
            max_new_tokens=256,
            mock=True,
        )

        path_cot_128 = temp_dir / "qwen2.5-0.5b" / "cot_128" / "test_dataset_predictions.jsonl"
        path_nocot_32 = temp_dir / "qwen2.5-0.5b" / "no_cot_32" / "test_dataset_predictions.jsonl"
        path_cot_256 = temp_dir / "qwen2.5-0.5b" / "cot_256" / "test_dataset_predictions.jsonl"

        assert path_cot_128.exists()
        assert path_nocot_32.exists()
        assert path_cot_256.exists()

        assert path_cot_128 != path_nocot_32 != path_cot_256
        print("  [PASS] Output paths differ for cot/no_cot/budget layout.")

        # Overwrite protection check
        try:
            run_evaluation(
                model_config=CORE_MODELS["qwen2.5-0.5b"],
                dataset_records=sample_records,
                dataset_name="test_dataset",
                output_dir=temp_dir,
                chain_of_thought=True,
                max_new_tokens=128,
                overwrite=False,
                mock=True,
            )
            assert False, "Should have raised FileExistsError"
        except FileExistsError:
            pass

        print("  [PASS] Overwrite protection verified.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cli_mock_run_eval() -> None:
    print("Testing CLI run_eval with --mock --limit 20...")
    temp_dir = Path(tempfile.mkdtemp(prefix="dws_cli_test_"))
    try:
        cmd = [
            sys.executable,
            "run_eval.py",
            "--model", "qwen2.5-0.5b",
            "--dataset", "rq1",
            "--mock",
            "--limit", "20",
            "--cot",
            "--max-new-tokens", "256",
            "--output-dir", str(temp_dir),
        ]
        res = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
        assert res.returncode == 0, f"run_eval failed: {res.stderr}"

        target_file = temp_dir / "qwen2.5-0.5b" / "cot_256" / "rq1_depth_predictions.jsonl"
        assert target_file.exists(), f"Target predictions file {target_file} does not exist"

        lines = [line for line in target_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 20, f"Expected 20 records, got {len(lines)}"

        print("  [PASS] CLI mock run_eval writes 20 records to the new layout.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    print("=" * 70)
    print("SLM EVALUATION PIPELINE TEST SUITE")
    print("=" * 70)
    test_model_registry()
    test_prompt_formatting_and_unification()
    test_end_to_end_mock_eval_and_layout()
    test_cli_mock_run_eval()
    print("=" * 70)
    print("ALL EVAL PIPELINE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

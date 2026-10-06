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

from eval.baselines import query_type_of
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

    # v2 CoT prompt check (location query type - default)
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
    assert "name of the container the object is in after that event" in prompt_v2_cot

    # v1 CoT prompt check (backward compatibility, location query type)
    prompt_v1_cot = build_user_prompt(
        context="A key was placed in the box.",
        question="Where is the key now?",
        chain_of_thought=True,
        prompt_version="v1",
    )
    assert "Step 3: name of the container the object is in after that event" in prompt_v1_cot
    assert "Final Answer: <name of the container>" in prompt_v1_cot

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


def test_prompt_query_type_aware() -> None:
    """Test that prompts are query-type-aware for both location and count."""
    print("Testing Query-Type-Aware Prompts...")

    context = "A ball was placed in the red box. The ball was moved to the blue bin."
    question_location = "Where is the ball now?"
    question_count = "How many balls are in the blue bin?"

    # Location query type (default)
    loc_v2_nocot = build_user_prompt(context, question_location, chain_of_thought=False, prompt_version="v2")
    loc_v2_cot = build_user_prompt(context, question_location, chain_of_thought=True, prompt_version="v2")
    loc_v1_nocot = build_user_prompt(context, question_location, chain_of_thought=False, prompt_version="v1")
    loc_v1_cot = build_user_prompt(context, question_location, chain_of_thought=True, prompt_version="v1")

    for prompt in [loc_v2_nocot, loc_v2_cot, loc_v1_nocot, loc_v1_cot]:
        assert "Final Answer: <name of the container>" in prompt
        assert "Final Answer: <a single integer>" not in prompt

    # Only CoT prompts have step slots
    for prompt in [loc_v2_cot, loc_v1_cot]:
        assert "name of the container the object is in after that event" in prompt
        assert "number of" not in prompt

    # Count query type
    cnt_v2_nocot = build_user_prompt(context, question_count, chain_of_thought=False, prompt_version="v2", query_type="count")
    cnt_v2_cot = build_user_prompt(context, question_count, chain_of_thought=True, prompt_version="v2", query_type="count")
    cnt_v1_nocot = build_user_prompt(context, question_count, chain_of_thought=False, prompt_version="v1", query_type="count")
    cnt_v1_cot = build_user_prompt(context, question_count, chain_of_thought=True, prompt_version="v1", query_type="count")

    for prompt in [cnt_v2_nocot, cnt_v2_cot, cnt_v1_nocot, cnt_v1_cot]:
        assert "Final Answer: <a single integer>" in prompt
        assert "Final Answer: <name of the container>" not in prompt

    # Count CoT step slot must be fully rendered (no unfilled braces)
    for prompt in [cnt_v2_cot, cnt_v1_cot]:
        assert "number of objects of the asked type in the asked container after that event" in prompt
        assert "name of the container the object is in after that event" not in prompt

    # v1 CoT has fixed 3-step shape
    assert cnt_v1_cot.count("Step 1:") == 1
    assert cnt_v1_cot.count("Step 2:") == 1
    assert cnt_v1_cot.count("Step 3:") == 1
    assert "Step 4:" not in cnt_v1_cot

    # v2 CoT uses Step k: with variable steps
    assert "Step k:" in cnt_v2_cot
    assert "Step 1:" not in cnt_v2_cot  # v2 uses generic Step k:

    print("  [PASS] Query-type-aware prompts verified.")


def test_prompt_invalid_query_type_raises() -> None:
    """Test that invalid query_type raises ValueError."""
    print("Testing Invalid Query Type Raises...")

    try:
        build_user_prompt("context", "question", query_type="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown query_type" in str(e)

    try:
        build_user_prompt("context", "question", query_type="")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown query_type" in str(e)

    print("  [PASS] Invalid query_type raises ValueError.")


def test_format_prompt_and_engine_thread_query_type() -> None:
    """Test that format_prompt and engine.format_input thread query_type from record."""
    print("Testing Query Type Threading Through format_prompt and Engine...")

    # Simulate a record with count query type
    record_count = {
        "instance_id": "test_count_001",
        "family": "split_chain",
        "spec": {"query_type": "count"},
        "context": "A ball was placed in the red box. The ball split into two.",
        "question": "How many balls are in the red box?",
        "gold_answer": "2",
    }

    record_location = {
        "instance_id": "test_loc_001",
        "family": "basic_chain",
        "spec": {"query_type": "location"},
        "context": "A key was placed in the green box. The key was moved to the large cabinet.",
        "question": "Where is the key now?",
        "gold_answer": "the large cabinet",
    }

    # Test format_prompt with explicit query_type
    prompt_count = format_prompt(
        context=record_count["context"],
        question=record_count["question"],
        chain_of_thought=True,
        prompt_version="v2",
        query_type="count",
    )
    assert "Final Answer: <a single integer>" in prompt_count
    assert "number of objects of the asked type in the asked container after that event" in prompt_count

    prompt_location = format_prompt(
        context=record_location["context"],
        question=record_location["question"],
        chain_of_thought=True,
        prompt_version="v2",
        query_type="location",
    )
    assert "Final Answer: <name of the container>" in prompt_location
    assert "name of the container the object is in after that event" in prompt_location

    # Test engine.format_input with explicit query_type
    engine = MockInferenceEngine()
    engine_prompt_count = engine.format_input(
        context=record_count["context"],
        question=record_count["question"],
        chain_of_thought=True,
        prompt_version="v2",
        query_type="count",
    )
    assert "Final Answer: <a single integer>" in engine_prompt_count
    assert "number of objects of the asked type in the asked container after that event" in engine_prompt_count

    engine_prompt_location = engine.format_input(
        context=record_location["context"],
        question=record_location["question"],
        chain_of_thought=True,
        prompt_version="v2",
        query_type="location",
    )
    assert "Final Answer: <name of the container>" in engine_prompt_location
    assert "name of the container the object is in after that event" in engine_prompt_location

    # Test query_type_of helper extracts correctly from record
    assert query_type_of(record_count) == "count"
    assert query_type_of(record_location) == "location"

    # Test format_prompt using query_type_of
    prompt_from_record = format_prompt(
        context=record_count["context"],
        question=record_count["question"],
        chain_of_thought=True,
        prompt_version="v2",
        query_type=query_type_of(record_count),
    )
    assert "Final Answer: <a single integer>" in prompt_from_record

    print("  [PASS] Query type threading verified.")


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
                "measured_factors": {"T_actual": 2, "E_actual": 1, "D_actual": 0, "V_actual": 0, "L_word": 20},
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
                "measured_factors": {"T_actual": 4, "E_actual": 1, "D_actual": 0, "V_actual": 0, "L_word": 35},
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
        dataset_path = temp_dir / "rq1_depth.jsonl"
        # Build the fixture with the real generator. D-010: a test that needs a
        # dataset must generate one, never hand-write JSON that can drift from
        # the record schema. These 20 records come from the same validated path
        # as the sweeps, so this test exercises the true schema end to end.
        from experiments._common import generate_instance

        with dataset_path.open("w", encoding="utf-8") as handle:
            for index in range(20):
                rec = generate_instance(
                    seed=1000 + index,
                    instance_id=f"cli_fixture_{index}",
                    family="basic_chain",
                    entity_count=1,
                    target_updates=2 + (index % 4),
                    distractor_updates=0,
                    num_containers=4,
                    experiment_tag="rq1_depth",
                    condition_id=f"cli_fixture_T{2 + (index % 4)}",
                )
                assert rec is not None
                handle.write(json.dumps(rec) + "\n")
        cmd = [
            sys.executable,
            "run_eval.py",
            "--model", "qwen2.5-0.5b",
            "--dataset", str(dataset_path),
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


# ============================================================
# run_eval regressions: L3 (tau constant), L5 (fail fast), L6 (step metrics)
# ============================================================

def _scripted_engine(responses):
    """MockInferenceEngine that returns ``responses`` in prompt order."""
    queue = list(responses)

    class _Scripted(MockInferenceEngine):
        def generate_batch(self, prompts, max_new_tokens=32, enforce_greedy=True, **_):
            out = queue[: len(prompts)]
            del queue[: len(prompts)]
            return out

    return _Scripted


def _step_records():
    base = {
        "family": "basic_chain",
        "experiment": "rq1_depth",
        "requested_factors": {"T": 2, "E": 1, "D": 0},
        "measured_factors": {"T_actual": 2},
        "context": "A key was put in the green box. The key was moved to the large cabinet.",
        "question": "Where is the key now?",
        "gold_answer": "the large cabinet",
        "step_wise_gold_answers": ["the green box", "the large cabinet"],
        "spec": {"query_type": "location"},
    }
    return [dict(base, instance_id="s_parsed"), dict(base, instance_id="s_unparsed")]


def test_run_eval_stepwise_metrics(monkeypatch) -> None:
    """L6: stepwise_parseable_instances counts instances with >= 1 parsed step
    (not every CoT instance), and a non-CoT run reports coverage None."""
    import run_eval

    responses = [
        "Step 1: the green box\nStep 2: the large cabinet\nFinal Answer: the large cabinet",
        "I think it moved.\nFinal Answer: the large cabinet",
    ]
    monkeypatch.setattr(run_eval, "MockInferenceEngine", _scripted_engine(responses))
    with tempfile.TemporaryDirectory() as tmp:
        cot = run_eval.run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"], dataset_records=_step_records(),
            dataset_name="steps", output_dir=Path(tmp), chain_of_thought=True,
            max_new_tokens=128, mock=True,
        )
    assert cot["stepwise_parseable_instances"] == 1
    assert cot["stepwise_coverage"] == 0.5

    monkeypatch.setattr(run_eval, "MockInferenceEngine",
                        _scripted_engine(["Final Answer: the large cabinet"] * 2))
    with tempfile.TemporaryDirectory() as tmp:
        no_cot = run_eval.run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"], dataset_records=_step_records(),
            dataset_name="steps", output_dir=Path(tmp), chain_of_thought=False,
            max_new_tokens=16, mock=True,
        )
    assert no_cot["stepwise_coverage"] is None
    assert no_cot["stepwise_parseable_instances"] == 0


def test_run_eval_fails_fast_on_unpromptable_query_type(monkeypatch) -> None:
    """L5: a redo_validity record is refused before any engine is built."""
    import pytest
    import run_eval

    def _no_engine(*_a, **_k):
        raise AssertionError("engine must not be initialised")

    monkeypatch.setattr(run_eval, "MockInferenceEngine", _no_engine)
    records = _step_records()
    records[1] = dict(records[1], instance_id="redo_1", spec={"query_type": "redo_validity"})
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match=r"redo_1 \(redo_validity\)"):
            run_eval.run_evaluation(
                model_config=CORE_MODELS["qwen2.5-0.5b"], dataset_records=records,
                dataset_name="redo", output_dir=Path(tmp), mock=True,
            )
        assert not list(Path(tmp).rglob("*"))
    with pytest.raises(ValueError, match="redo_1"):
        run_eval.build_prompts(MockInferenceEngine(), records, False, "v2")


def test_run_eval_uses_tau_constant() -> None:
    """L3: no hard-coded tau literal in run_eval.py."""
    import re

    from generator.constants import FAILURE_THRESHOLD_TAU

    src = (_REPO_ROOT / "run_eval.py").read_text(encoding="utf-8")
    assert "FAILURE_THRESHOLD_TAU" in src
    assert not re.search(r"tau\s*=\s*0\.7", src)
    assert "0.70" not in src
    assert FAILURE_THRESHOLD_TAU == 0.70


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

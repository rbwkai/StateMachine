"""Checklist 8: harness and inference (mock first, then the cached real model).

Covers the end-to-end path `format_prompt` -> engine -> `evaluate_predictions`
over the whole grid, prediction-id handling, condition pooling, first-error
analysis, and the decoding guarantees that the greedy path claims. Real-model
tests use the locally cached Qwen2.5-0.5B-Instruct snapshot at the pinned
revision and skip when that snapshot is absent. Defects are documented as plain
failing `test_failsnow_*` tests.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from analysis.first_error import ErrorType, analyze_first_error
from eval.engine import MockInferenceEngine
from eval.eval_harness import evaluate_predictions, format_prompt
from eval.models import CORE_MODELS, OPTIONAL_MODELS
from test.conftest import build_record, quiet


# RQ1 grid: every family at two target depths. Families that cannot generate a
# cell are exactly what this checklist section is meant to catch.
RQ1_GRID = [
    ("basic_chain", 1, 4, 0, 0),
    ("basic_chain", 1, 8, 0, 0),
    ("interleaved_chain", 2, 4, 1, 0),
    ("revision", 1, 4, 0, 0),
    ("revision", 1, 16, 0, 0),
    ("undo_chain", 1, 4, 0, 0),
    ("undo_chain", 1, 8, 0, 0),
    ("undo_redo_chain", 1, 8, 0, 0),
    ("split_chain", 2, 6, 0, 0),
    ("merge_chain", 1, 8, 0, 0),
    ("swap_chain", 1, 8, 0, 0),
]


def _cell_records(cell, seeds=(0, 1)):
    family, e, t, d, n = cell
    with quiet():
        records = [
            build_record(family, seed, entity_count=e, target_updates=t,
                         distractor_updates=d, textual_distractors=n,
                         instance_id=f"{family}_T{t}_D{d}_N{n}_s{seed}",
                         max_attempts=20)
            for seed in seeds
        ]
    return [record for record in records if record]


@pytest.fixture(scope="module")
def grid():
    """Every cell that generates, plus the cells that do not."""
    cells = {}
    for cell in RQ1_GRID:
        cells[cell] = _cell_records(cell)
    return cells


@pytest.fixture(scope="module")
def records(grid):
    flat = [record for cell_records in grid.values() for record in cell_records]
    assert flat, "at least one grid cell must generate"
    return flat


def _mock_predictions(instances, engine=None):
    engine = engine or MockInferenceEngine()
    prompts = [format_prompt(i["context"], i["question"]) for i in instances]
    responses = engine.generate_batch(prompts)
    return [
        {"instance_id": i["instance_id"], "raw_prediction": response, "pred_answer": response}
        for i, response in zip(instances, responses)
    ]


# ---------------------------------------------------------------------------
# Whole grid through the mock engine and the harness.
# ---------------------------------------------------------------------------

def test_every_instance_gets_a_prompt_and_a_response(records):
    engine = MockInferenceEngine()
    prompts = [format_prompt(record["context"], record["question"]) for record in records]
    assert all(record["question"] in prompt for record, prompt in zip(records, prompts))
    assert all(record["context"] in prompt for record, prompt in zip(records, prompts))
    responses = engine.generate_batch(prompts)
    assert len(responses) == len(records)
    assert engine.last_generation_metadata
    assert all(meta["finish_reason"] for meta in engine.last_generation_metadata)


def test_whole_grid_evaluates_end_to_end(records):
    predictions = _mock_predictions(records)
    report = evaluate_predictions(records, predictions)
    assert report["overall_total"] == len(records)
    assert report["prediction_count"] == len(records)
    assert report["missing_predictions"] == 0
    assert 0.0 <= report["overall_accuracy"] <= 1.0
    scored_ids = {result["instance_id"] for result in report["instance_results"]}
    assert scored_ids == {record["instance_id"] for record in records}
    assert report["condition_summaries"]


def test_condition_summaries_cover_every_family_in_the_grid(grid):
    reachable = {cell[0]: records for cell, records in grid.items() if records}
    flat = [record for records in reachable.values() for record in records]
    report = evaluate_predictions(flat, _mock_predictions(flat))
    for family in reachable:
        assert any(key.startswith(family) for key in report["condition_summaries"]), family


def test_failsnow_every_rq1_grid_cell_reaches_the_harness(grid):
    """[checklist 8] 'Run the whole grid through the mock engine and
    `evaluate_predictions`. This one test would have caught the RQ1 breakage.'

    [fails now] expected: every RQ1 cell produces records, so the grid can be
    scored end to end. currently the two count-only families return nothing at
    any depth, so a grid sweep silently scores an empty dataset.
    """
    empty = [cell for cell, records in grid.items() if not records]
    assert not empty, f"RQ1 cells with no records: {empty}"


# ---------------------------------------------------------------------------
# Prediction handling.
# ---------------------------------------------------------------------------

def test_duplicate_prediction_ids_are_rejected(records):
    predictions = _mock_predictions(records[:2])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_predictions(records[:2], predictions + [predictions[0]])


def test_missing_predictions_are_scored_as_incorrect_and_reported(records):
    subset = records[:3]
    predictions = _mock_predictions(subset)[:2]
    report = evaluate_predictions(subset, predictions)
    assert report["missing_predictions"] == 1
    assert report["missing_prediction_ids"] == [subset[2]["instance_id"]]
    result = next(r for r in report["instance_results"]
                  if r["instance_id"] == subset[2]["instance_id"])
    assert result["is_correct"] is False
    assert result["has_final_answer"] is False


def test_extra_prediction_ids_are_ignored(records):
    predictions = _mock_predictions(records[:2])
    predictions.append({"instance_id": "not-in-the-dataset", "raw_prediction": "Final Answer: x"})
    report = evaluate_predictions(records[:2], predictions)
    assert report["overall_total"] == 2
    assert all(r["instance_id"] != "not-in-the-dataset" for r in report["instance_results"])


def test_default_chain_of_thought_still_enforces_the_protocol(records):
    """`chain_of_thought=False` must not hand out credit for a markerless reply."""
    record = records[0]
    bare = {"instance_id": record["instance_id"],
            "raw_prediction": "The key is in the wooden cabinet."}
    report = evaluate_predictions([record], [bare])
    result = report["instance_results"][0]
    assert result["has_final_answer"] is False
    assert result["protocol_compliant"] is False
    assert result["is_correct"] is False

    with_cot = evaluate_predictions([record], [bare], chain_of_thought=True)
    cot_result = with_cot["instance_results"][0]
    assert cot_result["protocol_compliant"] is False
    assert cot_result["semantic_correct"] in {True, False}


def test_failsnow_different_entity_counts_and_distractor_counts_are_not_pooled():
    """[checklist 8] 'no pooling of different E or N into one cell.'

    [fails now] expected: the condition key carries E and N as well as family,
    T and D. currently it is `family_T{T}_D{D}`, so two cells that differ only
    in E and N are merged into one bucket and one accuracy.
    """
    cell_a = _cell_records(("basic_chain", 1, 4, 0, 0))
    cell_b = _cell_records(("basic_chain", 1, 4, 0, 4))
    assert cell_a and cell_b
    assert cell_a[0]["requested_factors"]["T"] == cell_b[0]["requested_factors"]["T"]
    assert cell_a[0]["requested_factors"]["N"] != cell_b[0]["requested_factors"]["N"]
    report = evaluate_predictions(cell_a + cell_b,
                                  _mock_predictions(cell_a + cell_b))
    assert len(report["condition_summaries"]) == 2


# ---------------------------------------------------------------------------
# First-error analysis.
# ---------------------------------------------------------------------------

# gold_states[0] is the initial state, so a 4-prediction trajectory pairs with
# gold_states[1:4]; gold[0] is never compared.
GOLD = ["A", "B", "C", "D", "A"]


@pytest.mark.parametrize("pred,expected", [
    (["B", "X", "D", "Z"], ErrorType.LOCAL_ERROR),
    (["B", "X", "Y", "Z"], ErrorType.PROPAGATING_ERROR),
    (["B", "C", "D", "Z"], ErrorType.FINAL_ONLY_ERROR),
    (["B", "X", "D", "A"], ErrorType.CANCELLATION_ERROR),
    (["B", "C", "D", "A"], ErrorType.NO_ERROR),
])
def test_all_five_error_types_are_reachable(pred, expected):
    gold = GOLD
    analysis = analyze_first_error(gold, pred)
    assert analysis.error_type is expected
    assert analysis.final_is_correct is (analysis.pred_final == analysis.gold_final)


def test_first_error_step_is_the_earliest_divergence():
    """Step numbers are 1-indexed over the predicted steps."""
    analysis = analyze_first_error(GOLD, ["B", "X", "D", "Z"])
    assert analysis.step_errors == [2, 4]
    assert analysis.first_error_step == 2


def test_harness_runs_first_error_analysis_when_a_trajectory_is_supplied(records):
    """A caller that supplies `pred_trajectory` with one entry fewer than
    `step_wise_gold` gets the full taxonomy back, aligned by step number."""
    record = records[0]
    golds = record["step_wise_gold_answers"]
    report = evaluate_predictions([record], [{
        "instance_id": record["instance_id"],
        "raw_prediction": f"Final Answer: {record['gold_answer']}",
        "pred_trajectory": golds[1:],
    }])
    result = report["instance_results"][0]
    assert result["error_analysis"] is not None
    assert result["error_analysis"]["error_type"] in {t.name for t in ErrorType}


def test_failsnow_harness_fills_pred_trajectory_from_the_response(records):
    """[checklist 8] 'nothing fills `pred_trajectory`.'

    [fails now] expected: a CoT response with parseable Step lines populates
    `pred_trajectory` so the trajectory analysis has input. currently the
    harness forwards whatever the caller passed and never parses the response,
    so a chain-of-thought run contributes no trajectory data at all.
    """
    record = records[0]
    golds = record["step_wise_gold_answers"]
    steps = "\n".join(f"Step {i + 1}. {value}" for i, value in enumerate(golds[1:]))
    report = evaluate_predictions([record], [{
        "instance_id": record["instance_id"],
        "raw_prediction": f"{steps}\nFinal Answer: {record['gold_answer']}",
    }], chain_of_thought=True)
    result = report["instance_results"][0]
    assert result["pred_trajectory"] == golds[1:]
    assert result["error_analysis"] is not None


# ---------------------------------------------------------------------------
# Registry invariants.
# ---------------------------------------------------------------------------

def test_every_registered_model_pins_a_commit_hash():
    registry = {**CORE_MODELS, **OPTIONAL_MODELS}
    assert registry
    for key, config in registry.items():
        assert len(config.revision) == 40, key
        assert all(char in "0123456789abcdef" for char in config.revision), key
        assert config.revision != "main", key


def test_greedy_decoding_flags_are_uniform_across_the_registry():
    for key, config in {**CORE_MODELS, **OPTIONAL_MODELS}.items():
        assert config.do_sample is False, key
        assert config.temperature == 0.0, key
        assert config.max_new_tokens > 0, key


def test_engine_never_enables_remote_code():
    import inspect

    from eval.engine import HuggingFaceEngine

    source = inspect.getsource(HuggingFaceEngine.__init__)
    assert "trust_remote_code=False" in source


def test_failsnow_placeholder_or_floating_revisions_fail_loudly():
    """[checklist 8] 'the revision hash resolves, and a placeholder detector
    fails loudly.'

    [fails now] expected: the registry refuses a placeholder revision such as
    `main` or a TODO string before any weight download starts. currently no
    validation function exists, so such a config is accepted and only fails
    later inside `from_pretrained`.
    """
    import eval.models as models

    validator = getattr(models, "validate_pinned_revision", None)
    assert validator is not None, (
        "eval.models has no revision validator, so a placeholder revision is "
        "accepted silently"
    )
    bad = models.ModelConfig(name="x", hf_model_id="x", family="x",
                             parameter_count_b=0.1, revision="main")
    with pytest.raises(ValueError):
        validator(bad)


# ---------------------------------------------------------------------------
# Real-model decoding guarantees (locally cached snapshot only).
# ---------------------------------------------------------------------------

MODEL_KEY = "qwen2.5-0.5b"


def _snapshot_dir(config):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    folder = hub / ("models--" + config.hf_model_id.replace("/", "--")) / "snapshots"
    return folder / config.revision if folder.exists() else None


@pytest.fixture(scope="module")
def hf_engine():
    transformers = pytest.importorskip("transformers")
    del transformers
    config = CORE_MODELS[MODEL_KEY]
    snapshot = _snapshot_dir(config)
    if snapshot is None or not snapshot.is_dir():
        pytest.skip(f"no cached snapshot for {config.hf_model_id}@{config.revision[:8]}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from eval.engine import HuggingFaceEngine

    return HuggingFaceEngine(config, device="cpu", precision="float32")


@pytest.fixture(scope="module")
def hf_prompt(hf_engine):
    context = ("The key was in the blue box. The key was moved to the red trunk. "
               "The blue box was moved to the wooden cabinet.")
    return hf_engine.format_input(context, "Where is the key now?", chain_of_thought=False)


def test_greedy_output_is_identical_at_batch_size_one_and_eight(hf_engine, hf_prompt):
    single = hf_engine.generate_batch([hf_prompt], max_new_tokens=24)
    batched = hf_engine.generate_batch([hf_prompt] * 8, max_new_tokens=24)
    assert len(batched) == 8
    assert len(set(batched)) == 1
    assert single[0] == batched[0]


def test_greedy_output_is_identical_across_two_runs(hf_engine, hf_prompt):
    first = hf_engine.generate_batch([hf_prompt], max_new_tokens=24)
    second = hf_engine.generate_batch([hf_prompt], max_new_tokens=24)
    assert first == second


def test_failsnow_greedy_path_leaves_repetition_penalty_at_one(hf_engine):
    """[checklist 8] 'the greedy path leaves `repetition_penalty` at 1.0 (please
    verify the model's `generation_config`).'

    [fails now] expected: pure greedy decoding, i.e. repetition_penalty == 1.0
    after `_configure_generation_defaults`. the shipped Qwen2.5 config carries
    repetition_penalty=1.1 and the engine only clears temperature, top_p and
    top_k, so every "greedy" run is penalised.
    """
    penalty = getattr(hf_engine.model.generation_config, "repetition_penalty", None)
    assert penalty == 1.0, f"repetition_penalty={penalty} on the greedy path"


def test_max_new_tokens_covers_the_gold_answer_and_the_finish_reason_is_reported(
    hf_engine, hf_prompt, records
):
    config = CORE_MODELS[MODEL_KEY]
    needed = hf_engine.tokenizer.encode(
        "Step 1. The key moves containers.\nFinal Answer: " + records[0]["gold_answer"],
        add_special_tokens=False,
    )
    assert config.max_new_tokens >= len(needed)

    hf_engine.generate_batch([hf_prompt], max_new_tokens=1)
    metadata = hf_engine.last_generation_metadata[0]
    assert metadata["finish_reason"] == "length"
    assert metadata["prompt_tokens"] > 0

    hf_engine.generate_batch([hf_prompt], max_new_tokens=24)
    assert hf_engine.last_generation_metadata[0]["finish_reason"] in {"length", "eos_token"}


def test_prompt_tokens_stay_below_the_model_max_length(hf_engine, hf_prompt):
    hf_engine.generate_batch([hf_prompt], max_new_tokens=8)
    prompt_tokens = hf_engine.last_generation_metadata[0]["prompt_tokens"]
    limit = hf_engine.tokenizer.model_max_length
    assert prompt_tokens < limit

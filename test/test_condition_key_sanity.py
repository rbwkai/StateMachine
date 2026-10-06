"""One condition key across harness and baselines, and the post-run sanity
screen that run_eval writes next to every run.

Regression for audit item 4: the harness key dropped ``_E1`` and ``_N0`` while
the baseline key always carried them, so the weak-baseline check never matched
and ``beats_weak_baseline`` was never set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.baselines import condition_key_of, run_all_baselines
from eval.eval_harness import condition_key, evaluate_predictions
from eval.models import CORE_MODELS
from eval.post_run_sanity import baseline_accuracy_by_condition, sanity_checks
from test.conftest import build_record, quiet


def _record(family, seed, **kwargs):
    with quiet():
        record = build_record(family, seed, max_attempts=20, **kwargs)
    assert record, (family, seed, kwargs)
    return record


@pytest.mark.parametrize("family,kwargs,expected", [
    ("basic_chain", {"entity_count": 1, "target_updates": 4}, "basic_chain_T4_D0_E1_N0"),
    ("split_chain", {"entity_count": 2, "target_updates": 6, "query_type": "count"},
     "split_chain_T6_D0_E2_N0"),
    ("basic_chain", {"entity_count": 1, "target_updates": 4, "textual_distractors": 2},
     "basic_chain_T4_D0_E1_N2"),
])
def test_harness_and_baseline_keys_are_identical(family, kwargs, expected):
    record = _record(family, 0, **kwargs)
    assert condition_key(record) == condition_key_of(record) == expected


def test_baseline_key_is_the_harness_function():
    """One definition (AGENTS.md §5), not two that happen to agree."""
    assert condition_key_of is condition_key


def _cell():
    return [_record("basic_chain", seed, entity_count=1, target_updates=4,
                    instance_id=f"b{seed}") for seed in range(4)]


def _report(records, answer):
    predictions = [{"instance_id": r["instance_id"],
                    "raw_prediction": f"Final Answer: {answer(r)}"} for r in records]
    return evaluate_predictions(records, predictions)


def test_nested_baselines_set_beats_weak_baseline():
    records = _cell()
    with quiet():
        baselines = run_all_baselines(records)
    key = condition_key(records[0])
    flat = baseline_accuracy_by_condition(baselines)
    assert key in flat and flat[key][0] in ("stateless", "mfc")

    right = sanity_checks(records, _report(records, lambda r: r["gold_answer"]),
                          baseline_accuracy=baselines)
    assert right[key]["beats_weak_baseline"] is True


def test_tying_a_weak_nested_baseline_is_flagged():
    records = _cell()
    key = condition_key(records[0])
    nested = {
        "stateless": {"per_condition": {key: {"accuracy": 0.0}}},
        "mfc": {"per_condition": {key: {"accuracy": 0.05}}},
    }
    wrong = sanity_checks(records, _report(records, lambda r: "nowhere"),
                          baseline_accuracy=nested)
    assert wrong[key]["baseline_name"] == "mfc"
    assert wrong[key]["baseline_accuracy"] == pytest.approx(0.05)
    assert wrong[key]["beats_weak_baseline"] is False


def _mock_run(tmp_path: Path, records, overwrite=False):
    import run_eval

    with quiet():
        return run_eval.run_evaluation(
            model_config=CORE_MODELS["qwen2.5-0.5b"],
            dataset_records=records,
            dataset_name="tiny",
            output_dir=tmp_path,
            mock=True,
            overwrite=overwrite,
        )


def test_run_eval_writes_sanity_json_and_protects_it(tmp_path):
    records = _cell()
    _mock_run(tmp_path, records)
    written = list(tmp_path.rglob("tiny.sanity.json"))
    assert len(written) == 1
    sanity = json.loads(written[0].read_text(encoding="utf-8"))
    key = condition_key(records[0])
    assert key in sanity["per_condition"]
    assert "beats_weak_baseline" in sanity["per_condition"][key]
    assert isinstance(sanity["failures"], list)

    # Only the sanity file remains: it alone must still block a rerun.
    for path in written[0].parent.iterdir():
        if path != written[0]:
            path.unlink()
    with pytest.raises(FileExistsError):
        _mock_run(tmp_path, records)
    _mock_run(tmp_path, records, overwrite=True)

"""
test/test_rescoring_layout.py
=============================
Regression tests for re-scoring saved predictions on the real run_eval layout.

- run_eval.py writes <out>/<model>/<cot|no_cot>_<tokens>/<dataset>_predictions.jsonl
  and stamps model / chain_of_thought / max_new_tokens on every row.
- analysis/evaluate_existing_predictions.py::infer_condition resolves the run
  condition from the row, then the real layout, then the legacy layout, and
  raises instead of defaulting to model="unknown", cot=False (which re-scored
  CoT runs without Step-protocol enforcement).
- Provenance: a missing scoring_version is drift; a missing instance_id is a
  clean ValueError, not a KeyError.
- condition_summary "accuracy" is strict, as everywhere else.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.evaluate_existing_predictions import (  # noqa: E402
    infer_condition,
    make_condition_summary,
    validate_provenance,
)
from eval.engine import MockInferenceEngine  # noqa: E402
from eval.models import CORE_MODELS  # noqa: E402
from generator.constants import SCORING_VERSION  # noqa: E402
from test.conftest import build_record, quiet  # noqa: E402

RESCORE_SCRIPT = _REPO_ROOT / "analysis" / "evaluate_existing_predictions.py"
MODEL = "qwen2.5-0.5b"
TOKENS = 256


# ============================================================
# infer_condition
# ============================================================

def test_infer_condition_real_run_eval_layout(tmp_path: Path) -> None:
    cot = tmp_path / MODEL / "cot_512" / "rq1_predictions.jsonl"
    no_cot = tmp_path / MODEL / "no_cot_32" / "rq1_predictions.jsonl"
    assert infer_condition(cot) == {
        "model": MODEL, "cot": True, "max_new_tokens": 512,
        "condition": f"{MODEL}_cot_512",
    }
    assert infer_condition(no_cot) == {
        "model": MODEL, "cot": False, "max_new_tokens": 32,
        "condition": f"{MODEL}_no_cot_32",
    }


def test_infer_condition_prefers_row_fields(tmp_path: Path) -> None:
    unresolvable = tmp_path / "anything" / "else" / "x_predictions.jsonl"
    row = {"model": "olmo-2-1b-instruct", "chain_of_thought": True, "max_new_tokens": 128}
    assert infer_condition(unresolvable, row)["condition"] == "olmo-2-1b-instruct_cot_128"

    # A row that agrees with a resolvable path resolves to the same condition.
    path = tmp_path / MODEL / "no_cot_32" / "x_predictions.jsonl"
    agreeing = {"model": MODEL, "chain_of_thought": False, "max_new_tokens": 32}
    resolved = infer_condition(path, agreeing)
    assert (resolved["model"], resolved["cot"], resolved["max_new_tokens"]) == (
        MODEL, False, 32,
    )

    # CSV rows carry strings.
    csv_row = {"model": "olmo-2-1b-instruct", "chain_of_thought": "False", "max_new_tokens": "64"}
    assert infer_condition(unresolvable, csv_row)["condition"] == "olmo-2-1b-instruct_no_cot_64"

    # Partial rows are completed from the path, field by field.
    partial = infer_condition(path, {"chain_of_thought": False})
    assert (partial["model"], partial["cot"], partial["max_new_tokens"]) == (MODEL, False, 32)


def test_infer_condition_legacy_layout(tmp_path: Path) -> None:
    path = tmp_path / "llama-3.2-3b_no_cot_32" / "sub" / "rq1_predictions.jsonl"
    assert infer_condition(path) == {
        "model": "llama-3.2-3b", "cot": False, "max_new_tokens": 32,
        "condition": "llama-3.2-3b_no_cot_32",
    }


def test_infer_condition_unresolvable_raises(tmp_path: Path) -> None:
    path = tmp_path / "foo" / "bar" / "x_predictions.jsonl"
    with pytest.raises(ValueError, match="x_predictions.jsonl"):
        infer_condition(path)
    # A row that resolves only some fields still fails loudly.
    with pytest.raises(ValueError, match="max_new_tokens"):
        infer_condition(path, {"model": MODEL, "chain_of_thought": True})


# ============================================================
# End-to-end: run_eval --cot (mock) -> re-scoring CLI
# ============================================================

def _records():
    records = []
    with quiet():
        for seed in range(6):
            rec = build_record("basic_chain", seed, target_updates=2, max_attempts=20)
            if rec is not None and rec.get("step_wise_gold_answers"):
                records.append(rec)
            if len(records) == 2:
                break
    assert len(records) == 2, "generator produced too few basic_chain records"
    assert records[0]["instance_id"] != records[1]["instance_id"]
    return records


def _scripted_engine(responses):
    queue = list(responses)

    class _Scripted(MockInferenceEngine):
        def generate_batch(self, prompts, max_new_tokens=None, *, enforce_greedy=True, **_):
            out = queue[: len(prompts)]
            del queue[: len(prompts)]
            return out

    return _Scripted


@pytest.fixture(scope="module")
def cot_run(tmp_path_factory):
    """One mock CoT run on the real layout: record 0 answers without Step
    lines, record 1 follows the Step protocol. Both final answers are gold."""
    import run_eval

    root = tmp_path_factory.mktemp("rescore")
    records = _records()
    no_steps = f"Final Answer: {records[0]['gold_answer']}"
    steps = "\n".join(
        f"Step {i}: {answer}"
        for i, answer in enumerate(records[1]["step_wise_gold_answers"], start=1)
    )
    with_steps = f"{steps}\nFinal Answer: {records[1]['gold_answer']}"

    dataset = root / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records),
        encoding="utf-8",
    )
    out = root / "results"
    with pytest.MonkeyPatch.context() as mp, quiet():
        mp.setattr(run_eval, "MockInferenceEngine", _scripted_engine([no_steps, with_steps]))
        run_eval.run_evaluation(
            model_config=CORE_MODELS[MODEL],
            dataset_records=records,
            dataset_name="tiny",
            output_dir=out,
            chain_of_thought=True,
            max_new_tokens=TOKENS,
            mock=True,
        )
    pred_file = out / MODEL / f"cot_{TOKENS}" / "tiny_predictions.jsonl"
    assert pred_file.exists()
    return {"root": root, "dataset": dataset, "out": out, "pred_file": pred_file,
            "records": records}


def _rescore(dataset: Path, predictions_root: Path, output_dir: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(RESCORE_SCRIPT), "--dataset", str(dataset),
         "--predictions-root", str(predictions_root), "--output-dir", str(output_dir),
         *extra],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=300,
    )


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_run_eval_rows_carry_run_condition(cot_run) -> None:
    for row in _load(cot_run["pred_file"]):
        assert row["model"] == MODEL
        assert row["chain_of_thought"] is True
        assert row["max_new_tokens"] == TOKENS
        assert row["prompt_version"] == "v2"
        assert row["scoring_version"] == SCORING_VERSION


def _assert_cot_rescored(rows, condition_summary, records) -> None:
    by_id = {row["instance_id"]: row for row in rows}
    assert set(by_id) == {rec["instance_id"] for rec in records}
    for row in rows:
        assert row["model"] == MODEL
        assert row["cot"] == "True"
        assert row["max_new_tokens"] == str(TOKENS)
        assert row["condition"] == f"{MODEL}_cot_{TOKENS}"
    no_steps = by_id[records[0]["instance_id"]]
    with_steps = by_id[records[1]["instance_id"]]
    # Re-scored with chain_of_thought=True: right answer, no Step lines -> not strict.
    assert no_steps["semantic_correct"] == "True"
    assert no_steps["strict_correct"] == "False"
    assert no_steps["protocol_compliant"] == "False"
    assert with_steps["strict_correct"] == "True"

    (summary,) = condition_summary
    assert summary["model"] == MODEL
    assert float(summary["accuracy"]) == pytest.approx(0.5)
    assert float(summary["semantic_accuracy"]) == pytest.approx(1.0)


def test_rescore_recovers_model_and_cot_from_real_layout(cot_run) -> None:
    res_dir = cot_run["root"] / "rescored"
    res = _rescore(cot_run["dataset"], cot_run["out"], res_dir)
    assert res.returncode == 0, res.stderr
    _assert_cot_rescored(
        _read_csv(res_dir / "predictions.csv"),
        _read_csv(res_dir / "condition_summary.csv"),
        cot_run["records"],
    )


def test_rescore_path_only_rows_still_resolve(cot_run) -> None:
    """Older run_eval rows lack the condition fields; the layout alone suffices."""
    root = cot_run["root"] / "path_only"
    stripped = [
        {k: v for k, v in row.items() if k not in {"model", "chain_of_thought", "max_new_tokens"}}
        for row in _load(cot_run["pred_file"])
    ]
    _write(root / "results" / MODEL / f"cot_{TOKENS}" / "tiny_predictions.jsonl", stripped)
    res = _rescore(cot_run["dataset"], root / "results", root / "rescored")
    assert res.returncode == 0, res.stderr
    _assert_cot_rescored(
        _read_csv(root / "rescored" / "predictions.csv"),
        _read_csv(root / "rescored" / "condition_summary.csv"),
        cot_run["records"],
    )


def test_rescore_unresolvable_layout_fails_loudly(cot_run) -> None:
    root = cot_run["root"] / "unresolvable"
    stripped = [
        {k: v for k, v in row.items() if k not in {"model", "chain_of_thought", "max_new_tokens"}}
        for row in _load(cot_run["pred_file"])
    ]
    _write(root / "results" / "flat" / "dir" / "tiny_predictions.jsonl", stripped)
    res = _rescore(cot_run["dataset"], root / "results", root / "rescored")
    assert res.returncode != 0
    assert "Cannot resolve run condition" in res.stderr
    assert not (root / "rescored" / "predictions.csv").exists()


def test_plot_results_load_rows_uses_resolver(cot_run) -> None:
    from analysis.plot_results import load_rows

    rows = load_rows(cot_run["out"])
    assert {row["model"] for row in rows} == {MODEL}
    assert {row["condition"] for row in rows} == {f"{MODEL}_cot_{TOKENS}"}
    assert all(row["cot"] is True for row in rows)


# ============================================================
# Provenance
# ============================================================

def test_missing_scoring_version_is_drift(cot_run) -> None:
    instances = {rec["instance_id"]: rec for rec in cot_run["records"]}
    preds = _load(cot_run["pred_file"])
    assert validate_provenance(preds, instances, False)[1] == 0

    preds[0].pop("scoring_version")
    fallback, drift = validate_provenance(preds, instances, False)
    assert drift == 1

    # CLI: aborts by default, proceeds with the explicit flag.
    root = cot_run["root"] / "no_scoring_version"
    _write(root / "results" / MODEL / f"cot_{TOKENS}" / "tiny_predictions.jsonl", preds)
    res = _rescore(cot_run["dataset"], root / "results", root / "rescored")
    assert res.returncode == 1
    assert "ABORT: 1 predictions" in res.stderr
    assert not (root / "rescored" / "predictions.csv").exists()

    res = _rescore(cot_run["dataset"], root / "results", root / "allowed",
                   "--allow-scoring-version-drift")
    assert res.returncode == 0, res.stderr
    provenance = json.loads((root / "allowed" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["mismatched_scoring_version"] == 1


def test_missing_instance_id_reported_cleanly(cot_run) -> None:
    instances = {rec["instance_id"]: rec for rec in cot_run["records"]}
    preds = _load(cot_run["pred_file"])
    preds[0]["instance_id"] = "no_such_id"
    with pytest.raises(ValueError, match="no_such_id"):
        validate_provenance(preds, instances, False)

    root = cot_run["root"] / "missing_id"
    _write(root / "results" / MODEL / f"cot_{TOKENS}" / "tiny_predictions.jsonl", preds)
    res = _rescore(cot_run["dataset"], root / "results", root / "rescored")
    assert res.returncode != 0
    assert "not found in dataset" in res.stderr
    assert "KeyError" not in res.stderr


# ============================================================
# Condition summary
# ============================================================

def test_condition_summary_accuracy_is_strict() -> None:
    rows = [
        {"model": MODEL, "cot": True, "max_new_tokens": 128,
         "semantic_correct": True, "strict_correct": False},
        {"model": MODEL, "cot": True, "max_new_tokens": 128,
         "semantic_correct": True, "strict_correct": True},
    ]
    (summary,) = make_condition_summary(rows)
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["semantic_accuracy"] == pytest.approx(1.0)


def test_row_contradicting_the_layout_raises(tmp_path: Path) -> None:
    path = tmp_path / MODEL / "cot_512" / "x_predictions.jsonl"
    with pytest.raises(ValueError, match="contradicts"):
        infer_condition(path, {"chain_of_thought": False})


@pytest.mark.parametrize("model", ['=HYPERLINK("http://x")', "a$b$", "../../etc", "-x"])
def test_unsafe_model_names_are_rejected(tmp_path: Path, model: str) -> None:
    path = tmp_path / "anything" / "else" / "x_predictions.jsonl"
    with pytest.raises(ValueError, match="Unsafe model name"):
        infer_condition(path, {"model": model, "chain_of_thought": True,
                               "max_new_tokens": 64})

"""Checklist 12: post-run sanity checks.

Once a model run exists, four checks decide whether the numbers can be believed:
accuracy below chance means a parsing bug, format compliance and truncation rate
are reported per cell, the strict/semantic gap is visible, and models beat the
baselines in cells where the baselines are weak. The checks run here against real
`evaluate_predictions` output from the mock engine. The absence of a reusable
helper in the analysis package is documented as a failing `test_failsnow_*` test.
"""
from __future__ import annotations

import pytest

from eval.baselines import compute_mfc_baseline, compute_stateless_baseline
from eval.engine import MockInferenceEngine
from eval.eval_harness import evaluate_predictions, format_prompt
from test.conftest import build_record, quiet
from test.test_baseline_solvers import chance_level


WEAK_BASELINE = 0.10  # a baseline this weak should not be matched by a model


@pytest.fixture(scope="module")
def cells():
    """Two small cells with enough instances for a per-cell rate."""
    built = {}
    with quiet():
        for family, entity_count, target_updates in (("basic_chain", 1, 4),
                                                    ("undo_redo_chain", 1, 8)):
            records = [
                build_record(family, seed, entity_count=entity_count,
                             target_updates=target_updates,
                             instance_id=f"{family}_T{target_updates}_s{seed}",
                             max_attempts=20)
                for seed in range(4)
            ]
            built[(family, target_updates)] = [r for r in records if r]
    assert all(built.values()), built
    return built


def _run(cells, builder):
    """Mock-engine predictions for every cell; `builder` shapes the response."""
    engine = MockInferenceEngine()
    instances = [record for records in cells.values() for record in records]
    prompts = [format_prompt(i["context"], i["question"]) for i in instances]
    responses = engine.generate_batch(prompts)
    shaped = []
    for record, response in zip(instances, responses):
        text = builder(record, response)
        shaped.append({"instance_id": record["instance_id"],
                       "raw_prediction": text, "pred_answer": text})
    return instances, evaluate_predictions(instances, shaped)


def _condition_key(record):
    factors = record["requested_factors"]
    return f"{record['family']}_T{factors['T']}_D{factors.get('D', 0)}"


def _per_cell(instances, report):
    by_id = {record["instance_id"]: record for record in instances}
    cells = {}
    for result in report["instance_results"]:
        record = by_id[result["instance_id"]]
        entry = cells.setdefault(_condition_key(record), {"n": 0, "strict": 0,
                                                         "semantic": 0,
                                                         "final_line": 0,
                                                         "chance": 0.0})
        entry["n"] += 1
        entry["strict"] += bool(result["is_correct"])
        entry["semantic"] += bool(result["semantic_correct"])
        entry["final_line"] += bool(result["has_final_answer"])
        entry["chance"] = chance_level(record)
    return cells


def sanity_checks(instances, report):
    """The four checklist-12 checks, over harness output."""
    per_cell = _per_cell(instances, report)
    stats = {}
    for key, entry in per_cell.items():
        n = entry["n"]
        strict = entry["strict"] / n
        semantic = entry["semantic"] / n
        stats[key] = {
            "accuracy": strict,
            "chance": entry["chance"],
            "below_chance": strict < entry["chance"] - 1e-9,
            "format_compliance": entry["final_line"] / n,
            "truncation_rate": 1.0 - entry["final_line"] / n,
            "strict_semantic_gap": semantic - strict,
        }
    return stats


# ---------------------------------------------------------------------------
# Accuracy below chance.
# ---------------------------------------------------------------------------

def test_below_chance_accuracy_is_flagged_as_a_parsing_bug(cells):
    instances, report = _run(cells, lambda record, response: "Final Answer: nowhere at all")
    stats = sanity_checks(instances, report)
    assert stats
    for key, cell in stats.items():
        assert cell["accuracy"] <= cell["chance"], key
        assert cell["below_chance"] is True, key


def test_accuracy_at_or_above_chance_is_not_flagged(cells):
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    _, report = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    stats = sanity_checks(instances, report)
    assert all(cell["accuracy"] == 1.0 for cell in stats.values())
    assert all(cell["below_chance"] is False for cell in stats.values())


def test_chance_level_used_here_matches_the_baseline_module():
    with quiet():
        record = build_record("basic_chain", 0, entity_count=1, target_updates=4)
    assert chance_level(record) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Format compliance and truncation per cell.
# ---------------------------------------------------------------------------

def test_format_compliance_and_truncation_rate_are_reported_per_cell(cells):
    """A response with no Final Answer line counts as truncated."""
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    marked = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    unmarked = _run(cells, lambda record, response: "I am not sure about this one.")

    compliant = sanity_checks(*marked)
    truncated = sanity_checks(*unmarked)
    for key in compliant:
        assert compliant[key]["format_compliance"] == 1.0, key
        assert compliant[key]["truncation_rate"] == 0.0, key
        assert truncated[key]["format_compliance"] == 0.0, key
        assert truncated[key]["truncation_rate"] == 1.0, key


def test_a_partially_compliant_run_shows_an_intermediate_rate(cells):
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    with_final = {"basic_chain"}
    _, report = _run(
        cells,
        lambda record, response: (f"Final Answer: {gold_answers[record['instance_id']]}"
                                  if record["family"] in with_final
                                  else "no marker in this reply"),
    )
    stats = sanity_checks(instances, report)
    assert stats["basic_chain_T4_D0"]["format_compliance"] == 1.0
    assert stats["undo_redo_chain_T8_D0"]["format_compliance"] == 0.0


# ---------------------------------------------------------------------------
# Strict versus semantic gap.
# ---------------------------------------------------------------------------

def test_strict_and_semantic_accuracy_gap_is_visible(cells):
    """Semantic credit without the marker produces a positive gap."""
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    _, report = _run(
        cells,
        lambda record, response: f"I believe the object is in {gold_answers[record['instance_id']]}.",
    )
    stats = sanity_checks(instances, report)
    for key, cell in stats.items():
        assert cell["accuracy"] == 0.0, key
        assert cell["strict_semantic_gap"] > 0.0, key
        assert cell["format_compliance"] == 0.0, key


def test_a_fully_compliant_run_has_no_gap(cells):
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    _, report = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    stats = sanity_checks(instances, report)
    assert all(cell["strict_semantic_gap"] == 0.0 for cell in stats.values())


# ---------------------------------------------------------------------------
# Models against baselines.
# ---------------------------------------------------------------------------

def test_a_model_that_only_matches_a_weak_baseline_is_flagged(cells):
    per_cell_model = {}
    for (family, target_updates), records in cells.items():
        baseline = compute_mfc_baseline(records) + compute_stateless_baseline(records)
        per_cell_model[f"{family}_T{target_updates}_D0"] = (
            sum(1 for r in baseline if r.is_correct) / len(baseline)
        )

    def beats_baseline(cell_key, model_accuracy, baseline_accuracy):
        return baseline_accuracy >= WEAK_BASELINE or model_accuracy > baseline_accuracy

    for key, baseline_accuracy in per_cell_model.items():
        # A model that merely ties the baseline is flagged wherever the baseline
        # is weak, and never flagged where the baseline is already strong.
        assert beats_baseline(key, baseline_accuracy, baseline_accuracy) is (
            baseline_accuracy >= WEAK_BASELINE
        ), key
        assert beats_baseline(key, 0.99, baseline_accuracy) is True, key


def test_model_accuracy_from_the_harness_is_comparable_to_baseline_accuracy(cells):
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    _, report = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    stats = sanity_checks(instances, report)
    for key, records in ((f"{family}_T{t}_D0", rs)
                         for (family, t), rs in cells.items()):
        baseline = compute_mfc_baseline(records)
        baseline_accuracy = sum(1 for r in baseline if r.is_correct) / len(baseline)
        assert stats[key]["accuracy"] >= baseline_accuracy


# ---------------------------------------------------------------------------
# Reusability.
# ---------------------------------------------------------------------------

def test_failsnow_the_sanity_checks_are_available_as_a_helper():
    """[checklist 12] post-run sanity checks.

    [fails now] expected: the checks ship with the analysis layer so a real run
    can be screened, not only a test. currently they exist nowhere in `analysis`
    or `eval`; the logic below lives in this test module alone.
    """
    import analysis
    import eval

    helpers = [
        name for module in (analysis, eval)
        for name in dir(module)
        if "sanity" in name.lower() or "below_chance" in name.lower()
    ]
    assert helpers, "no sanity-check helper is exported by analysis or eval"

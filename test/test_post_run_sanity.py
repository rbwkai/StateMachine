"""Checklist 12: post-run sanity checks.

Once a model run exists, four checks decide whether the numbers can be believed:
accuracy below chance means a parsing bug, format compliance and truncation rate
are reported per cell, the strict/semantic gap is visible, and models beat the
baselines in cells where the baselines are weak. The checks are the real
`eval.post_run_sanity` functions, run against `evaluate_predictions` output from
the mock engine.
"""
from __future__ import annotations

import pytest

from eval.baselines import compute_mfc_baseline, run_all_baselines
from eval.engine import MockInferenceEngine
from eval.eval_harness import evaluate_predictions, format_prompt
from eval.post_run_sanity import sanity_checks, sanity_failures
from test.conftest import build_record, quiet
from test.test_baseline_solvers import chance_level


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


def _key(family, target_updates):
    """The harness condition key (always carries E and N)."""
    return f"{family}_T{target_updates}_D0_E1_N0"


# ---------------------------------------------------------------------------
# Accuracy below chance.
# ---------------------------------------------------------------------------

def test_below_chance_accuracy_is_flagged_as_a_parsing_bug(cells):
    instances, report = _run(cells, lambda record, response: "Final Answer: nowhere at all")
    stats = sanity_checks(instances, report)
    assert stats
    for key, cell in stats.items():
        assert cell["accuracy"] < cell["chance"], key
        assert cell["below_chance"] is True, key
    assert sanity_failures(stats)


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

def test_format_compliance_and_missing_final_rate_are_reported_per_cell(cells):
    """A response with no Final Answer line is missing its final line; it is
    not truncation, which needs finish_reason == "length"."""
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    marked = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    unmarked = _run(cells, lambda record, response: "I am not sure about this one.")

    compliant = sanity_checks(*marked)
    truncated = sanity_checks(*unmarked)
    for key in compliant:
        assert compliant[key]["format_compliance"] == 1.0, key
        assert compliant[key]["missing_final_rate"] == 0.0, key
        assert truncated[key]["format_compliance"] == 0.0, key
        assert truncated[key]["missing_final_rate"] == 1.0, key
        # No row recorded a finish_reason: truncation is unknown, not 0 or 1.
        assert truncated[key]["truncation_rate"] is None, key


def test_truncation_rate_comes_from_finish_reason(cells):
    instances, report = _run(cells, lambda record, response: "Final Answer: x")
    predictions = [{"instance_id": r["instance_id"],
                    "finish_reason": "length" if i % 2 else "eos_token"}
                   for i, r in enumerate(instances)]
    stats = sanity_checks(instances, report, predictions=predictions)
    for key, cell in stats.items():
        assert 0.0 < cell["truncation_rate"] < 1.0, key
    assert any("truncation rate" in msg for msg in sanity_failures(stats))


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
    assert stats[_key("basic_chain", 4)]["format_compliance"] == 1.0
    assert stats[_key("undo_redo_chain", 8)]["format_compliance"] == 0.0


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
    """Flat map: a tie with a weak baseline is flagged; a strong one never is."""
    instances, report = _run(cells, lambda record, response: "Final Answer: nowhere")
    keys = [_key(family, t) for family, t in cells]
    weak = sanity_checks(instances, report, baseline_accuracy={k: 0.0 for k in keys})
    strong = sanity_checks(instances, report, baseline_accuracy={k: 0.9 for k in keys})
    for key in keys:
        assert weak[key]["beats_weak_baseline"] is False, key
        assert strong[key]["beats_weak_baseline"] is True, key
    assert any("does not beat a weak baseline" in m for m in sanity_failures(weak))


def test_nested_run_all_baselines_output_is_matched_by_key(cells):
    """The harness and baseline keys agree, so the check actually runs."""
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    _, report = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    with quiet():
        baselines = run_all_baselines(instances)
    stats = sanity_checks(instances, report, baseline_accuracy=baselines)
    for family, t in cells:
        cell = stats[_key(family, t)]
        assert cell["baseline_name"] in ("stateless", "mfc")
        assert cell["beats_weak_baseline"] is True


def test_model_accuracy_from_the_harness_is_comparable_to_baseline_accuracy(cells):
    instances = [record for records in cells.values() for record in records]
    gold_answers = {r["instance_id"]: r["gold_answer"] for r in instances}
    _, report = _run(cells, lambda record, response: f"Final Answer: {gold_answers[record['instance_id']]}")
    stats = sanity_checks(instances, report)
    for key, records in ((_key(family, t), rs)
                         for (family, t), rs in cells.items()):
        baseline = compute_mfc_baseline(records)
        baseline_accuracy = sum(1 for r in baseline if r.is_correct) / len(baseline)
        assert stats[key]["accuracy"] >= baseline_accuracy


# ---------------------------------------------------------------------------
# Reusability.
# ---------------------------------------------------------------------------

def test_the_sanity_checks_are_available_as_a_helper():
    """[checklist 12] the checks ship with the eval layer so a real run can be
    screened, not only a test."""
    import analysis
    import eval

    helpers = [
        name for module in (analysis, eval)
        for name in dir(module)
        if "sanity" in name.lower() or "below_chance" in name.lower()
    ]
    assert helpers, "no sanity-check helper is exported by analysis or eval"

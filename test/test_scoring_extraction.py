"""Checklist 6: scoring and extraction.

Covers `Final Answer` line variants, container-substring ambiguity, numeric
answers, step-wise parsing, truncation semantics, candidate isolation and the
relationship between the strict, semantic and `protocol_compliant` flags.
Real defects are documented as plain failing `test_failsnow_*` tests.
"""
from __future__ import annotations

import pytest

from eval.scoring import (
    candidate_answers,
    extract_instance_answer,
    extract_step_answers,
    normalize_text,
    score_prediction,
)
from test.conftest import build_record, quiet


# ---------------------------------------------------------------------------
# Fixtures: two records that are built once for the whole module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def location_record():
    with quiet():
        record = build_record("basic_chain", 0, entity_count=1, target_updates=4)
    assert record, "baseline location cell must generate"
    return record


@pytest.fixture(scope="module")
def count_record():
    with quiet():
        record = build_record("split_chain", 0, entity_count=2, target_updates=6)
    assert record, "baseline count cell must generate"
    return record


def _extraction(text, record, chain_of_thought=False):
    return extract_instance_answer(
        text,
        candidate_answers(record),
        chain_of_thought=chain_of_thought,
        gold_answer=record["gold_answer"],
        instance=record,
    )


# ---------------------------------------------------------------------------
# Final Answer line variants that do not change the content.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", [
    "Final Answer: {gold}",
    "final answer: {gold}",
    "FINAL ANSWER: {gold}",
    "Final Answer: {gold}.",
    "Final Answer: {gold},",
    "**Final Answer: {gold}**",
    "Final Answer: \"{gold}\"",
    "Final Answer: '{gold}'",
    "  Final Answer:   {gold}  ",
    "Final Answer: {no_article}",
])
def test_content_preserving_variants_get_full_credit(template, location_record):
    gold = location_record["gold_answer"]
    extraction = _extraction(template.format(gold=gold, no_article=normalize_text(gold)),
                             location_record)
    assert extraction.has_final_answer
    assert extraction.answer == gold
    assert extraction.semantic_correct
    assert extraction.strict_correct


def test_missing_colon_gets_semantic_but_not_strict_credit(location_record):
    """A marker without a colon is not a Final Answer line, so it degrades to
    the truncation contract: semantic credit only."""
    extraction = _extraction(f"Final Answer {location_record['gold_answer']}",
                             location_record)
    assert not extraction.has_final_answer
    assert extraction.semantic_correct
    assert not extraction.strict_correct


# ---------------------------------------------------------------------------
# Ambiguous, negated, echoed and empty final answers.
# ---------------------------------------------------------------------------

def test_two_containers_in_one_final_line_get_no_credit(location_record):
    candidates = candidate_answers(location_record)
    assert len(candidates) >= 2
    extraction = _extraction(f"Final Answer: {candidates[0]} and {candidates[1]}",
                             location_record)
    assert extraction.answer == ""
    assert not extraction.semantic_correct


def test_negated_container_gets_no_credit(location_record):
    candidates = candidate_answers(location_record)
    other = next(c for c in candidates if c != location_record["gold_answer"])
    extraction = _extraction(f"Final Answer: {other}, not {location_record['gold_answer']}",
                             location_record)
    assert extraction.answer == ""
    assert not extraction.semantic_correct


@pytest.mark.parametrize("segment", [
    "<answer>",
    "<container>",
    "**answer**",
    "answer",
])
def test_echoed_placeholder_gets_no_credit(segment, location_record):
    extraction = _extraction(f"Final Answer: {segment}", location_record)
    assert extraction.answer == ""
    assert not extraction.semantic_correct


def test_first_final_answer_marker_wins(location_record):
    """The scoring contract uses the first marker; later markers are echo."""
    gold = location_record["gold_answer"]
    candidates = candidate_answers(location_record)
    other = next(c for c in candidates if c != gold)

    first_wrong = _extraction(f"Final Answer: {other}\nFinal Answer: {gold}", location_record)
    assert first_wrong.answer == other
    assert not first_wrong.semantic_correct

    first_right = _extraction(f"Final Answer: {gold}\nFinal Answer: {other}", location_record)
    assert first_right.answer == gold
    assert first_right.strict_correct


@pytest.mark.parametrize("text", [
    "",
    "   \n  \n",
    "Final Answer:",
    "Final Answer:    ",
    "Final Answer:\n",
])
def test_empty_or_whitespace_final_answers_get_no_credit(text, location_record):
    extraction = _extraction(text, location_record)
    assert extraction.answer == ""
    assert not extraction.semantic_correct
    assert not extraction.strict_correct


# ---------------------------------------------------------------------------
# Container-substring ambiguity.
# ---------------------------------------------------------------------------

def test_container_substring_ambiguity_resolves_by_exact_match():
    candidates = ["the box", "the red box"]
    for response in ("the box", "the red box", "box", "red box", "a red box"):
        extraction = extract_instance_answer(f"Final Answer: {response}", candidates)
        expected = "the box" if "red" not in response else "the red box"
        assert extraction.answer == expected, response


def test_bare_noun_matching_several_candidates_gets_no_credit():
    """No exact candidate, and the noun matches two names: refuse to guess."""
    extraction = extract_instance_answer("Final Answer: the box",
                                         ["the red box", "the blue box"])
    assert extraction.answer == ""


# ---------------------------------------------------------------------------
# Numeric answers.
# ---------------------------------------------------------------------------

def test_numeric_gold_extracts_and_is_correct(count_record):
    extraction = _extraction(f"Final Answer: {count_record['gold_answer']}", count_record)
    assert extraction.answer == count_record["gold_answer"]
    assert extraction.semantic_correct


def test_failsnow_wrong_numeric_answers_are_extracted_as_wrong_not_dropped(count_record):
    """[checklist 6] numeric answers `"2"`, `"two"`, `"2 tokens"`, `"12"`.

    [fails now] expected: a numeric answer that differs from the gold is still
    read as a number, so scoring records a wrong guess with `semantic_correct
    False`. currently `candidate_answers` only holds the gold value for a count
    cell, so every non-gold number falls through the placeholder branch and
    extracts as `''`, which is indistinguishable from no answer at all.
    """
    gold = count_record["gold_answer"]
    wrong = [value for value in ("2", "two", "2 tokens", "12") if value != gold]
    assert wrong, "cell must expose a non-gold numeric form"
    for value in wrong:
        extraction = _extraction(f"Final Answer: {value}", count_record)
        assert extraction.answer != "", (
            f"numeric answer {value!r} extracted as empty for gold {gold!r}"
        )
        assert not extraction.semantic_correct


# ---------------------------------------------------------------------------
# Step-wise parsing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", [
    "Step {i}. {value}",
    "Step {i}: {value}",
    "Step {i}) {value}",
    "Step {i}- {value}",
    "step {i}. {value}",
    "  Step {i}.   {value}",
])
def test_step_markers_are_parsed_and_aligned(template, location_record):
    golds = location_record["step_wise_gold_answers"]
    text = "\n".join(template.format(i=i + 1, value=value)
                     for i, value in enumerate(golds))
    assert extract_step_answers(text, candidate_answers(location_record),
                                num_steps=len(golds)) == golds


def test_missing_step_is_none_and_alignment_is_kept(location_record):
    golds = location_record["step_wise_gold_answers"]
    text = "\n".join(f"Step {i + 1}. {value}"
                     for i, value in enumerate(golds) if i != 2)
    parsed = extract_step_answers(text, candidate_answers(location_record),
                                  num_steps=len(golds))
    assert parsed[:2] == golds[:2]
    assert parsed[2] is None
    assert parsed[3:] == golds[3:]


def test_extra_steps_beyond_the_gold_length_are_dropped(location_record):
    golds = location_record["step_wise_gold_answers"]
    text = "\n".join(f"Step {i + 1}. {value}" for i, value in enumerate(golds))
    text += f"\nStep {len(golds) + 1}. {golds[0]}"
    parsed = extract_step_answers(text, candidate_answers(location_record),
                                  num_steps=len(golds))
    assert len(parsed) == len(golds)
    assert parsed == golds


def test_repeated_step_number_uses_the_last_occurrence(location_record):
    golds = location_record["step_wise_gold_answers"]
    text = f"Step 1. {golds[1]}\nStep 1. {golds[0]}"
    parsed = extract_step_answers(text, candidate_answers(location_record), num_steps=2)
    assert parsed == [golds[0], None]


def test_out_of_order_steps_are_aligned_by_number(location_record):
    golds = location_record["step_wise_gold_answers"]
    text = f"Step 2. {golds[1]}\nStep 1. {golds[0]}"
    parsed = extract_step_answers(text, candidate_answers(location_record), num_steps=2)
    assert parsed == [golds[0], golds[1]]


def test_prose_steps_extract_as_none(location_record):
    golds = location_record["step_wise_gold_answers"]
    text = ("Step 1. I am not certain where it went after the swap.\n"
            "Step 2. Let me re-check the sentence about the swap and merge.\n"
            f"Final Answer: {location_record['gold_answer']}")
    assert extract_step_answers(text, candidate_answers(location_record),
                                num_steps=len(golds)) == [None] * len(golds)


# ---------------------------------------------------------------------------
# Truncation, candidate isolation and flag agreement.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", [
    "The {noun} is now in {gold}, so the answer must be {gold}",
    "First {noun}. Then {noun}. Final line reasoning without a marker: {gold}",
])
def test_text_without_a_final_answer_marker_gets_semantic_but_never_strict_credit(
    template, location_record
):
    gold = location_record["gold_answer"]
    text = template.format(gold=gold, noun="key")
    extraction = _extraction(text, location_record)
    assert not extraction.has_final_answer
    assert not extraction.protocol_compliant
    assert not extraction.strict_correct
    assert extraction.semantic_correct


def test_candidates_never_include_another_instances_answers(location_record):
    """A neighbouring record's gold must not enter the candidate set."""
    foreign = "the glass jar"
    assert foreign not in candidate_answers(location_record)
    other = {
        "gold_answer": foreign,
        "step_wise_gold_answers": [foreign],
        "final_state": {"container_display_names": {"c0": foreign}},
    }
    candidates = candidate_answers(other, dataset_context=[location_record])
    assert foreign in candidates  # own gold is allowed
    assert not set(candidates) - {foreign}, "foreign answer leaked into candidates"

    extraction = extract_instance_answer(
        f"Final Answer: {location_record['gold_answer']}", candidates,
        gold_answer=foreign,
    )
    assert extraction.answer == ""
    assert not extraction.semantic_correct


@pytest.mark.parametrize("chain_of_thought,expect_strict", [
    (False, True),
    (True, False),
])
def test_chain_of_thought_flag_controls_protocol_checks(chain_of_thought,
                                                          expect_strict, location_record):
    extraction = _extraction(f"Final Answer: {location_record['gold_answer']}",
                             location_record, chain_of_thought=chain_of_thought)
    assert extraction.semantic_correct
    assert extraction.protocol_compliant is not chain_of_thought
    assert extraction.strict_correct is expect_strict


def test_chain_of_thought_answer_with_a_step_line_is_strict_correct(location_record):
    text = ("Step 1. I read the last sentence about the key.\n"
            f"Final Answer: {location_record['gold_answer']}")
    extraction = _extraction(text, location_record, chain_of_thought=True)
    assert extraction.protocol_compliant
    assert extraction.strict_correct


def test_score_prediction_exposes_the_extraction_verdict(location_record):
    prediction = {
        "instance_id": location_record["instance_id"],
        "raw_prediction": f"Final Answer: {location_record['gold_answer']}",
    }
    scored = score_prediction(prediction, location_record)
    assert scored["is_correct"] == scored["strict_correct"] is True
    assert scored["is_correct_semantic"] == scored["semantic_correct"] is True
    assert scored["protocol_compliant"]
    assert scored["extracted_answer"] == location_record["gold_answer"]
    assert scored["gold_answer"] == location_record["gold_answer"]

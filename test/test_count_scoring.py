"""Count-answer scoring: hedges and malformed integers are never credited.

Covers final-answer extraction, step extraction with count golds, step
coverage, and that run_eval builds the count prompt for count records.
"""

from __future__ import annotations

import pytest

from eval.engine import MockInferenceEngine
from eval.scoring import (
    extract_instance_answer,
    extract_step_answers,
    read_count_answer,
    score_prediction,
)
from run_eval import build_prompts

GOLD = "1"
COUNT_CANDIDATES = ["0", "1", "2"]
# The question supplies the only non-generic unit word a count may carry (L4).
QUESTION = "How many keys are in the box now?"
INSTANCE = {"question": QUESTION, "spec": {"query_type": "count"}}

# ====================================================================
# Final answer
# ====================================================================

NOT_CREDITED = ["1.5", "1 or 2", "-1", "1/2", "exactly 1", "1 or", "1 2", "1st"]
CREDITED = ["1", "1.", "there is 1 key", "one"]


@pytest.mark.parametrize("segment", NOT_CREDITED)
def test_hedged_final_answer_is_not_credited(segment: str) -> None:
    res = extract_instance_answer(
        f"Final Answer: {segment}", COUNT_CANDIDATES, gold_answer=GOLD
    )
    assert res.semantic_correct is False
    assert res.strict_correct is False
    assert read_count_answer(segment) is None


@pytest.mark.parametrize("segment", CREDITED)
def test_plain_final_answer_is_credited(segment: str) -> None:
    res = extract_instance_answer(
        f"Final Answer: {segment}", COUNT_CANDIDATES, gold_answer=GOLD,
        instance=INSTANCE,
    )
    assert res.answer == "1"
    assert res.strict_correct is True


def test_unit_word_answer_reads_its_own_number() -> None:
    phones = "How many phones are in the box now?"
    assert read_count_answer("12 phones", question=phones) == "12"
    res = extract_instance_answer(
        "Final Answer: 12 phones", COUNT_CANDIDATES, gold_answer=GOLD,
        instance={"question": phones},
    )
    assert res.answer == "12"
    assert res.semantic_correct is False


@pytest.mark.parametrize(
    "segment,question,expected",
    [
        ("12 phones", "How many phones are in the box now?", "12"),
        ("there are 3 keys.", QUESTION, "3"),
        ("three tokens", None, "3"),
        ("3 boxes", "How many boxes are in the bag now?", "3"),
        ("1 box", "How many boxes are in the bag now?", "1"),
        ("2 berries", "How many berries are in the bag now?", "2"),
    ],
)
def test_count_reader_keeps_supported_forms(segment, question, expected) -> None:
    assert read_count_answer(segment, question=question) == expected


# ====================================================================
# Step answers
# ====================================================================


@pytest.mark.parametrize("segment", NOT_CREDITED)
def test_hedged_step_is_not_credited(segment: str) -> None:
    raw = f"Step 1: {segment}\nStep 2: 1"
    for gold in (GOLD, None):  # explicit gold, and inferred from candidates
        parsed = extract_step_answers(raw, COUNT_CANDIDATES, num_steps=2, gold_answer=gold)
        assert parsed == [None, "1"]


@pytest.mark.parametrize("segment", CREDITED)
def test_plain_step_is_credited(segment: str) -> None:
    parsed = extract_step_answers(
        f"Step 1: {segment}", COUNT_CANDIDATES, num_steps=1, gold_answer=GOLD,
        instance=INSTANCE,
    )
    assert parsed == ["1"]


def test_wrong_step_count_is_recorded_as_that_count() -> None:
    parsed = extract_step_answers(
        "Step 1: 12 phones", COUNT_CANDIDATES, num_steps=1, gold_answer=GOLD,
        instance={"question": "How many phones are in the box now?"},
    )
    assert parsed == ["12"]


# ====================================================================
# Step coverage
# ====================================================================


def _count_record() -> dict:
    return {
        "instance_id": "c1",
        "context": "Ann puts a key in the box.",
        "question": "How many keys are in the box?",
        "gold_answer": "1",
        "step_wise_gold_answers": ["1", "2", "1"],
        "spec": {"query_type": "count"},
    }


def test_step_coverage_counts_parsed_steps() -> None:
    raw = "Step 1: 1\nStep 2: 1 or 2\nStep 3: 1\nFinal Answer: 1"
    scored = score_prediction({"raw_prediction": raw}, _count_record(), chain_of_thought=True)
    assert scored["step_coverage"] == pytest.approx(2 / 3)
    assert scored["protocol_compliant"] is True


def test_step_coverage_is_none_without_cot() -> None:
    scored = score_prediction({"raw_prediction": "Final Answer: 1"}, _count_record())
    assert scored["step_coverage"] is None
    assert scored["protocol_compliant"] is True


# ====================================================================
# run_eval prompt construction
# ====================================================================


@pytest.mark.parametrize("cot", [False, True])
def test_run_eval_count_record_gets_count_prompt(cot: bool) -> None:
    [prompt] = build_prompts(MockInferenceEngine(), [_count_record()], cot, "v2")
    assert "<a single integer>" in prompt
    assert "<name of the container>" not in prompt
    assert "{" not in prompt and "}" not in prompt


# ====================================================================
# Count grammar regressions (M1, M2, L4, Security-L2)
# ====================================================================


@pytest.mark.parametrize(
    "segment,expected",
    [
        ("2 in total", "2"),
        ("2 of them", "2"),
        ("there are 2 items in total", "2"),
        ("a total of 2", "2"),
        ("two in total", "2"),
        ("2 objects", "2"),
        ("007", "7"),
        ("0", "0"),
        ("00", "0"),
    ],
)
def test_neutral_fillers_and_leading_zeros_are_read(segment, expected) -> None:
    assert read_count_answer(segment) == expected


@pytest.mark.parametrize(
    "segment",
    [
        # Hedges and qualifiers: one rule for digits and number words.
        "about 2", "about two", "around 2", "around two", "approximately 2",
        "roughly two", "exactly 1", "exactly one",
        # Trailing words that are neither a count noun nor the asked noun.
        "3 not", "2 maybe", "2 each", "2 phones",
        # Non-ASCII digits are not converted.
        "\u0661\u0662", "\uff11\uff12",
    ],
)
def test_hedges_stray_words_and_non_ascii_digits_are_not_read(segment) -> None:
    assert read_count_answer(segment, question=QUESTION) is None
    res = extract_instance_answer(
        f"Final Answer: {segment}", COUNT_CANDIDATES, gold_answer=GOLD,
        instance=INSTANCE,
    )
    assert res.strict_correct is False
    assert res.method == "ambiguous"


def test_question_noun_needs_a_question() -> None:
    assert read_count_answer("2 keys") is None
    assert read_count_answer("2 keys", question=QUESTION) == "2"
    assert read_count_answer("1 key", question=QUESTION) == "1"


def test_fallback_skips_unparseable_segment_and_reads_last_line() -> None:
    """M1: an answer sentence that is not a count must not end the search."""
    res = extract_instance_answer(
        "The pen is in the red trunk.\n1", COUNT_CANDIDATES, gold_answer=GOLD,
    )
    assert res.answer == "1"
    assert res.method == "last_line"
    assert res.semantic_correct is True
    assert res.strict_correct is False  # no Final Answer line


def test_fallback_with_no_count_anywhere_is_ambiguous() -> None:
    res = extract_instance_answer(
        "The pen is in the red trunk.\nI am not sure.", COUNT_CANDIDATES,
        gold_answer=GOLD,
    )
    assert res.answer == ""
    assert res.method == "ambiguous"


def test_long_unnormalized_input_is_rejected_quickly() -> None:
    import time

    hostile = "1" + " \t" * 50_000 + "x"
    t0 = time.perf_counter()
    assert read_count_answer(hostile) is None
    assert time.perf_counter() - t0 < 1.0


# ====================================================================
# Step answer space from the instance (M3)
# ====================================================================


def test_count_steps_detected_from_instance_without_gold_answer() -> None:
    """candidate_answers() of a real count record also holds container names,
    so the candidate rule alone never saw a count cell."""
    from eval.scoring import candidate_answers

    record = dict(_count_record())
    record["final_state"] = {"container_names": {"c0": "the red box", "c1": "the old bin"}}
    cands = candidate_answers(record)
    assert "the red box" in cands
    parsed = extract_step_answers(
        "Step 1: 1\nStep 2: two\nStep 3: 1 key", cands, num_steps=3, instance=record,
    )
    assert parsed == ["1", "2", "1"]

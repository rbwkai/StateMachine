"""Answer-format robustness of eval/scoring.py.

Markdown / LaTeX wrapped answers, echoed prompt slots, the
``first_final_answer`` selector, and a ReDoS guard for the unwrap step and the
"Final Answer" marker regex (AGENTS.md §6.13).
"""
from __future__ import annotations

import time

import pytest

from eval.prompts import ANSWER_SLOTS
from eval.scoring import (
    candidate_answers,
    extract_instance_answer,
    extract_step_answers,
    read_count_answer,
    unwrap_answer_segment,
)
import eval.scoring as scoring
from test.conftest import build_record, quiet


@pytest.fixture(scope="module")
def location_record():
    with quiet():
        record = build_record("basic_chain", 0, entity_count=1, target_updates=4)
    assert record, "baseline location cell must generate"
    return record


@pytest.fixture(scope="module")
def count_record():
    with quiet():
        record = build_record(
            "split_chain", 0, entity_count=2, target_updates=6, query_type="count"
        )
    assert record, "baseline count cell must generate"
    assert str(record["gold_answer"]).isdigit()
    return record


def _extract(text, record, **kwargs):
    return extract_instance_answer(
        text,
        candidate_answers(record),
        gold_answer=record["gold_answer"],
        instance=record,
        **kwargs,
    )


def _other_location(record):
    gold = record["gold_answer"]
    return next(c for c in candidate_answers(record) if c != gold)


def _other_count(record):
    return str(int(record["gold_answer"]) + 1)


# ---------------------------------------------------------------------------
# Wrapped answers. "{a}" is the gold answer.
# ---------------------------------------------------------------------------

WRAPPED_FINAL = [
    "Final Answer: **{a}**",
    "Final Answer: `{a}`",
    "Final Answer: \\boxed{{{a}}}",
    "Final Answer: ${a}$",
    "Final Answer: *{a}*",
    "Final Answer: __{a}__",
    "Final Answer: \\({a}\\)",
    "Final Answer: \\[{a}\\]",
    "Final Answer: \\text{{{a}}}",
    "Final Answer: $\\boxed{{{a}}}$",
    "Final Answer: **{a}**.",
    "**Final Answer:** {a}",
    "**Final Answer: {a}**",
    "**Final Answer**: {a}",
    "Final Answer**:** {a}",
]


@pytest.mark.parametrize("template", WRAPPED_FINAL)
def test_wrapped_final_answer_count(template, count_record):
    gold = str(count_record["gold_answer"])
    res = _extract(template.format(a=gold), count_record)
    assert res.has_final_answer
    assert res.method == "final_answer"
    assert res.answer == gold
    assert res.strict_correct


@pytest.mark.parametrize("template", WRAPPED_FINAL)
def test_wrapped_final_answer_location(template, location_record):
    gold = location_record["gold_answer"]
    res = _extract(template.format(a=gold), location_record)
    assert res.has_final_answer
    assert res.answer == gold
    assert res.strict_correct


@pytest.mark.parametrize("template", WRAPPED_FINAL)
def test_wrapped_wrong_count_is_recorded_as_that_count(template, count_record):
    wrong = _other_count(count_record)
    res = _extract(template.format(a=wrong), count_record)
    assert res.answer == wrong
    assert res.method == "final_answer"
    assert not res.semantic_correct


@pytest.mark.parametrize("line", ["**{a}**", "${a}$", "`{a}`", "\\boxed{{{a}}}"])
def test_wrapped_fallback_last_line_count(line, count_record):
    gold = str(count_record["gold_answer"])
    res = _extract("Let me think.\n" + line.format(a=gold), count_record)
    assert not res.has_final_answer
    assert res.method == "last_line"
    assert res.answer == gold
    assert res.semantic_correct and not res.strict_correct


@pytest.mark.parametrize("line", ["**{a}**", "`{a}`", "\\boxed{{{a}}}"])
def test_wrapped_fallback_last_line_location(line, location_record):
    gold = location_record["gold_answer"]
    res = _extract("Let me think.\n" + line.format(a=gold), location_record)
    assert res.answer == gold
    assert res.semantic_correct and not res.strict_correct


# ---------------------------------------------------------------------------
# Hedges stay rejected after unwrapping.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", [
    "Final Answer: **about {a}**",
    "Final Answer: $approximately {a}$",
    "Final Answer: \\boxed{{{a} or {b}}}",
    "**Final Answer:** around {a}",
    "Final Answer: `{a} maybe`",
    "Final Answer: **{a}.5**",
    "Final Answer: \\boxed{{{a}}} or \\boxed{{{b}}}",
])
def test_hedge_after_unwrap_is_ambiguous(template, count_record):
    gold = str(count_record["gold_answer"])
    res = _extract(template.format(a=gold, b=_other_count(count_record)), count_record)
    assert res.answer == ""
    assert res.method == "ambiguous"
    assert not res.semantic_correct


def test_unwrap_does_not_strip_leading_dot():
    # read_count_answer(".5") is a separate, pre-existing normalize_text issue
    # (it strips a leading "."); the unwrap step must not add a second path.
    assert unwrap_answer_segment(".5") == ".5"
    assert unwrap_answer_segment("**.5**") == ".5"


def test_location_disjunction_after_unwrap_gets_no_credit(location_record):
    gold = location_record["gold_answer"]
    other = _other_location(location_record)
    res = _extract(f"Final Answer: **{gold} or {other}**", location_record)
    assert res.answer == ""
    assert not res.semantic_correct


def test_wrapped_count_step_lines(count_record):
    res = extract_step_answers(
        "Step 1: **2**\nStep 2: $3$\nStep 3: **about 2**",
        candidate_answers(count_record),
        instance=count_record,
    )
    assert res == ["2", "3", None]


# ---------------------------------------------------------------------------
# Echoed prompt slots (eval.prompts.ANSWER_SLOTS) are placeholders.
# ---------------------------------------------------------------------------

def _final_slot_payloads():
    return [slots["final"].split(":", 1)[1].strip() for slots in ANSWER_SLOTS.values()]


@pytest.mark.parametrize("payload", _final_slot_payloads())
@pytest.mark.parametrize("record_name", ["count_record", "location_record"])
def test_echoed_slot_then_real_answer(payload, record_name, request):
    record = request.getfixturevalue(record_name)
    gold = str(record["gold_answer"])
    res = _extract(f"Final Answer: {payload}\nFinal Answer: {gold}", record)
    assert res.answer == gold
    assert res.strict_correct


@pytest.mark.parametrize("payload", _final_slot_payloads() + [
    "**<a single integer>**",
    "<answer>",
    "**answer**",
])
@pytest.mark.parametrize("record_name", ["count_record", "location_record"])
def test_echoed_slot_alone_scores_nothing(payload, record_name, request):
    record = request.getfixturevalue(record_name)
    res = _extract(f"Final Answer: {payload}", record)
    assert res.has_final_answer
    assert res.answer == ""
    assert res.method == "final_answer"
    assert not res.semantic_correct


def test_step_slot_echo_is_placeholder(count_record):
    slot = ANSWER_SLOTS["count"]["step"]
    gold = str(count_record["gold_answer"])
    res = _extract(f"Final Answer: {slot}\nFinal Answer: {gold}", count_record)
    assert res.answer == gold


# ---------------------------------------------------------------------------
# first_final_answer selector.
# ---------------------------------------------------------------------------

def _two_markers(record, other):
    gold = str(record["gold_answer"])
    payload = ANSWER_SLOTS["count"]["final"].split(":", 1)[1].strip()
    return gold, (
        f"Final Answer: {payload}\nFinal Answer: {other}\n"
        f"Final Answer: **{gold}**\nFinal Answer: {payload}"
    )


@pytest.mark.parametrize("record_name,other_fn", [
    ("count_record", _other_count),
    ("location_record", _other_location),
])
def test_first_final_answer_modes(record_name, other_fn, request):
    record = request.getfixturevalue(record_name)
    other = other_fn(record)
    gold, text = _two_markers(record, other)

    default = _extract(text, record)
    first = _extract(text, record, first_final_answer=True)
    last = _extract(text, record, first_final_answer=False)

    assert default == first
    assert first.answer == other and not first.semantic_correct
    assert last.answer == gold and last.strict_correct


# ---------------------------------------------------------------------------
# Unwrap unit cases and ReDoS guard.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("**1**", "1"),
    ("`1`", "1"),
    ("$1$", "1"),
    ("\\boxed{1}", "1"),
    ("\\(\\boxed{1}\\)", "1"),
    ("$\\text{the red box}$", "the red box"),
    ("1**", "1"),
    ("**about 2**", "about 2"),
    ("plain", "plain"),
    ("", ""),
])
def test_unwrap_answer_segment(raw, expected):
    assert unwrap_answer_segment(raw) == expected


ADVERSARIAL = [
    "*" * 50000,
    "$" * 50000,
    "`" * 50000,
    "_" * 50000,
    "\\boxed{" * 10000,
    "\\boxed{" * 10000 + "}" * 10000,
    "\\(" * 25000,
    " " * 50000,
    "\t \n" * 17000,
    "*1" * 25000,
]


@pytest.mark.parametrize("payload", ADVERSARIAL, ids=lambda p: f"{p[:8]!r}x{len(p)}")
def test_redos_guard(payload, count_record, location_record):
    texts = [
        payload,
        "Final Answer: " + payload,
        "**Final Answer:**" + payload,
        "Final Answer" + payload + ":",
        ("final answer" + payload)[:50000],
    ]
    start = time.perf_counter()
    for text in texts:
        unwrap_answer_segment(text)
        read_count_answer(text)
        _extract(text, count_record)
        _extract(text, location_record)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"{elapsed:.3f}s"


def test_redos_guard_repeated_markers(count_record):
    text = "**Final Answer:** **" * 3000
    start = time.perf_counter()
    _extract(text, count_record)
    _extract(text, count_record, first_final_answer=False)
    assert time.perf_counter() - start < 0.5


@pytest.mark.parametrize("raw", [".5", "-1", "+2", "**.5**", "\\boxed{-1}"])
def test_signed_or_fractional_count_is_not_read(raw):
    assert read_count_answer(raw) is None


def test_empty_marker_line_does_not_swallow_the_next_line():
    # "[ \t]*" after the colon: an empty payload must not cross the newline.
    match = scoring._FINAL_ANSWER_MARKER.search("Final Answer:\nFinal Answer: 1")
    assert match is not None and match.group(1) == ""


@pytest.mark.parametrize("raw,gold", [
    ("Final Answer:\n2", "2"),
    ("Final Answer:\r\n2", "2"),
    ("Final Answer:\n**2**", "2"),
    ("Final Answer:\nFinal Answer: 1", "1"),
])
def test_answer_on_the_line_after_an_empty_marker_is_read(raw, gold):
    result = extract_instance_answer(raw, ["1", "2", "3"], gold_answer=gold)
    assert result.answer == gold and result.has_final_answer is True


def test_reply_truncated_after_the_marker_falls_back():
    result = extract_instance_answer(
        "The key is in the red box.\nFinal Answer:",
        ["the red box", "the blue bin"], gold_answer="the red box",
    )
    assert result.has_final_answer is False
    assert result.answer == "the red box" and result.strict_correct is False


def test_emphasis_inside_a_count_segment_is_ignored():
    assert read_count_answer("**2** keys", question="How many keys are in the box?") == "2"
    assert read_count_answer("**about** 2") is None

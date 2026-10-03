"""
test/test_scoring.py
====================
Unit tests for single scoring engine (eval/scoring.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.scoring import (
    AnswerExtraction,
    candidate_answers,
    extract_instance_answer,
    extract_step_answers,
    normalize_text,
    score_prediction,
)

CANDIDATES = ["the green box", "the large basket", "the red crate", "the blue shelf"]


def test_gold_large_basket_extracted() -> None:
    res = extract_instance_answer(
        "Final Answer: Large basket",
        CANDIDATES,
        gold_answer="the large basket",
    )
    assert res.answer == "the large basket"
    assert res.semantic_correct is True
    assert res.strict_correct is True
    assert res.protocol_compliant is True


def test_placeholder_followed_by_final_answer() -> None:
    res = extract_instance_answer(
        "Final Answer: <answer>\nFinal Answer: the red crate",
        CANDIDATES,
        gold_answer="the red crate",
    )
    assert res.answer == "the red crate"
    assert res.method == "final_answer"
    assert res.semantic_correct is True


def test_two_final_answer_lines_last_wins() -> None:
    res = extract_instance_answer(
        "Final Answer: the green box\nFinal Answer: the red crate",
        CANDIDATES,
        gold_answer="the red crate",
    )
    assert res.answer == "the red crate"


def test_ambiguous_final_answer_no_credit() -> None:
    for gold in ["the green box", "the large basket"]:
        res = extract_instance_answer(
            "Final Answer: the green box or the large basket",
            CANDIDATES,
            gold_answer=gold,
        )
        assert res.answer == ""
        assert res.semantic_correct is False
        assert res.strict_correct is False


def test_final_answer_none_no_fallthrough() -> None:
    res = extract_instance_answer(
        "The key is now in the large basket.\nFinal Answer: none",
        CANDIDATES,
        gold_answer="the large basket",
    )
    assert res.answer == ""
    assert res.semantic_correct is False
    assert res.strict_correct is False
    assert res.has_final_answer is True


def test_truncated_cot_fallback() -> None:
    res = extract_instance_answer(
        "Step 3: the large basket",
        CANDIDATES,
        chain_of_thought=True,
        gold_answer="the large basket",
    )
    assert res.answer == "the large basket"
    assert res.semantic_correct is True
    assert res.strict_correct is False
    assert res.protocol_compliant is False


def test_step_answers_parsing() -> None:
    raw = "Step 1: Green box\n2. foo\nStep 2: bar\nStep 3: Large basket"
    parsed = extract_step_answers(raw, CANDIDATES, num_steps=3)
    assert parsed == ["the green box", None, "the large basket"]


def main() -> None:
    print("=" * 70)
    print("RUNNING TEST_SCORING.PY")
    print("=" * 70)

    test_gold_large_basket_extracted()
    print("1. Gold 'the large basket': 'Final Answer: Large basket' -> PASS")

    test_placeholder_followed_by_final_answer()
    print("2. 'Final Answer: <answer>\\nFinal Answer: the red crate' -> PASS")

    test_two_final_answer_lines_last_wins()
    print("3. Two Final Answer lines -> last wins -> PASS")

    test_ambiguous_final_answer_no_credit()
    print("4. 'Final Answer: the green box or the large basket' -> '' -> PASS")

    test_final_answer_none_no_fallthrough()
    print("5. 'The key is now in the large basket.\\nFinal Answer: none' -> '' -> PASS")

    test_truncated_cot_fallback()
    print("6. Truncated CoT with 'Step 3: the large basket' -> semantic_correct True, strict_correct False -> PASS")

    test_step_answers_parsing()
    print("7. 'Step 1: Green box' parsed, '2. foo' ignored -> PASS")

    print("=" * 70)
    print("ALL SCORING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

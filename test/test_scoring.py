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


def test_first_final_answer_is_the_contract() -> None:
    res = extract_instance_answer(
        "Final Answer: <answer>\nFinal Answer: the red crate",
        CANDIDATES,
        gold_answer="the red crate",
    )
    assert res.answer == ""
    assert res.method == "final_answer"
    assert res.semantic_correct is False


def test_first_final_answer_wins() -> None:
    res = extract_instance_answer(
        "Final Answer: the green box\nFinal Answer: the red crate",
        CANDIDATES,
        gold_answer="the red crate",
    )
    assert res.answer == "the green box"


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


# ---- instance-aware extraction (was test/test_evaluator_v2.py, D-015) ----

PROBE_CANDIDATES = ["the blue shelf", "the large basket", "the old box"]


def test_marker_free_answer_is_semantically_extractable_but_not_protocol_compliant() -> None:
    result = extract_instance_answer(
        "The key is now on the blue shelf.", PROBE_CANDIDATES, chain_of_thought=False
    )
    assert result.answer == "the blue shelf"
    assert result.method == "answer_sentence"
    assert not result.has_final_answer
    assert not result.protocol_compliant


def test_final_answer_marker_is_strictly_compliant() -> None:
    result = extract_instance_answer(
        "Final Answer: the large basket", PROBE_CANDIDATES, chain_of_thought=False
    )
    assert result.answer == "the large basket"
    assert result.method == "final_answer"
    assert result.has_final_answer
    assert result.protocol_compliant


def test_articleless_answer_matches_article_prefixed_candidate() -> None:
    result = extract_instance_answer(
        "Step 1: large basket\nFinal Answer: large basket",
        PROBE_CANDIDATES,
        chain_of_thought=True,
    )
    assert result.answer == "the large basket"
    assert result.method == "final_answer"
    assert result.protocol_compliant


def test_ambiguous_candidate_mentions_are_invalid() -> None:
    result = extract_instance_answer(
        "The key moved from the old box to the blue shelf.",
        PROBE_CANDIDATES,
        chain_of_thought=False,
    )
    assert result.answer == ""
    assert result.method == "none"


def test_cot_requires_step_and_final_answer() -> None:
    without_step = extract_instance_answer(
        "Final Answer: the blue shelf", PROBE_CANDIDATES, chain_of_thought=True
    )
    with_step = extract_instance_answer(
        "Step 1: the large basket\nFinal Answer: the blue shelf",
        PROBE_CANDIDATES,
        chain_of_thought=True,
    )
    assert not without_step.protocol_compliant
    assert with_step.protocol_compliant

def main() -> None:
    print("=" * 70)
    print("RUNNING TEST_SCORING.PY")
    print("=" * 70)

    test_gold_large_basket_extracted()
    print("1. Gold 'the large basket': 'Final Answer: Large basket' -> PASS")

    # D-003: the scorer takes the FIRST 'Final Answer:' marker, so a leading
    # placeholder is the answer under test and the later real answer is ignored.
    test_first_final_answer_is_the_contract()
    print("2. 'Final Answer: <answer>\\nFinal Answer: the red crate' -> PASS")

    test_first_final_answer_wins()
    print("3. Two Final Answer lines -> first wins -> PASS")

    test_ambiguous_final_answer_no_credit()
    print("4. 'Final Answer: the green box or the large basket' -> '' -> PASS")

    test_final_answer_none_no_fallthrough()
    print("5. 'The key is now in the large basket.\\nFinal Answer: none' -> '' -> PASS")

    test_truncated_cot_fallback()
    print("6. Truncated CoT with 'Step 3: the large basket' -> semantic_correct True, strict_correct False -> PASS")

    test_step_answers_parsing()
    print("7. 'Step 1: Green box' parsed, '2. foo' ignored -> PASS")

    test_marker_free_answer_is_semantically_extractable_but_not_protocol_compliant()
    print("8. marker-free answer: extractable but not protocol compliant -> PASS")

    test_final_answer_marker_is_strictly_compliant()
    print("9. 'Final Answer:' marker is protocol compliant -> PASS")

    test_articleless_answer_matches_article_prefixed_candidate()
    print("10. articleless answer matches article-prefixed candidate -> PASS")

    test_ambiguous_candidate_mentions_are_invalid()
    print("11. ambiguous candidate mentions score no credit -> PASS")

    test_cot_requires_step_and_final_answer()
    print("12. CoT compliance needs both Step and Final Answer -> PASS")

    print("=" * 70)
    print("ALL SCORING TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()

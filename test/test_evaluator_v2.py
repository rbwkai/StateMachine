"""Focused tests for the instance-aware offline evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.evaluator_v2 import extract_instance_answer


CANDIDATES = ["the blue shelf", "the large basket", "the old box"]


def test_marker_free_answer_is_semantically_extractable_but_not_protocol_compliant() -> None:
    result = extract_instance_answer(
        "The key is now on the blue shelf.", CANDIDATES, chain_of_thought=False
    )
    assert result.answer == "the blue shelf"
    assert result.method == "answer_sentence"
    assert not result.has_final_answer
    assert not result.protocol_compliant


def test_final_answer_marker_is_strictly_compliant() -> None:
    result = extract_instance_answer(
        "Final Answer: the large basket", CANDIDATES, chain_of_thought=False
    )
    assert result.answer == "the large basket"
    assert result.method == "final_answer"
    assert result.has_final_answer
    assert result.protocol_compliant


def test_articleless_answer_matches_article_prefixed_candidate() -> None:
    result = extract_instance_answer(
        "Step 1: large basket\nFinal Answer: large basket",
        CANDIDATES,
        chain_of_thought=True,
    )
    assert result.answer == "the large basket"
    assert result.method == "final_answer"
    assert result.protocol_compliant


def test_ambiguous_candidate_mentions_are_invalid() -> None:
    result = extract_instance_answer(
        "The key moved from the old box to the blue shelf.",
        CANDIDATES,
        chain_of_thought=False,
    )
    assert result.answer == ""
    assert result.method == "none"


def test_cot_requires_step_and_final_answer() -> None:
    without_step = extract_instance_answer(
        "Final Answer: the blue shelf", CANDIDATES, chain_of_thought=True
    )
    with_step = extract_instance_answer(
        "Step 1: the large basket\nFinal Answer: the blue shelf",
        CANDIDATES,
        chain_of_thought=True,
    )
    assert not without_step.protocol_compliant
    assert with_step.protocol_compliant

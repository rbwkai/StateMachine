"""Instance-aware offline evaluator for existing DWS-Bench generations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


def normalize_text(value: object) -> str:
    """Normalize candidate labels without changing their semantic content."""
    normalized = re.sub(r"\s+", " ", str(value).strip().lower()).strip(
        " .,!?:;\"'"
    )
    return re.sub(r"^(?:the|a|an)\s+", "", normalized)


@dataclass(frozen=True)
class AnswerExtraction:
    answer: str
    method: str
    has_final_answer: bool
    protocol_compliant: bool


def _unique_candidate_match(text: str, candidates: Sequence[str]) -> Optional[str]:
    normalized_text = normalize_text(text)
    matches = {
        candidate
        for candidate in candidates
        if normalize_text(candidate) and normalize_text(candidate) in normalized_text
    }
    if len(matches) != 1:
        return None
    return next(iter(matches))


def _answer_segments(raw_response: str) -> Iterable[Tuple[str, str]]:
    """Yield answer-bearing segments in preference order."""
    final_matches = list(
        re.finditer(
            r"final\s+answer\s*:\s*(.+?)(?:\n|$)",
            raw_response,
            re.IGNORECASE,
        )
    )
    if final_matches:
        yield "final_answer", final_matches[-1].group(1)

    lines = [line.strip() for line in raw_response.splitlines() if line.strip()]
    answer_pattern = re.compile(
        r"(?:key|object|item|entity)\s+is\s+(?:now\s+)?"
        r"(?:in|on|inside|at)\s+[^.\n]+",
        re.IGNORECASE,
    )
    for line in reversed(lines):
        if answer_pattern.search(line):
            yield "answer_sentence", line

    if lines:
        yield "last_line", lines[-1]


def extract_instance_answer(
    raw_response: str,
    candidates: Sequence[str],
    chain_of_thought: bool = False,
) -> AnswerExtraction:
    """Extract one unique candidate from a final or answer-bearing segment."""
    has_final_answer = bool(
        re.search(r"final\s+answer\s*:", raw_response, re.IGNORECASE)
    )
    has_step = bool(re.search(r"(?:^|\n)\s*step\s*\d+\s*[:.)-]", raw_response, re.IGNORECASE))
    protocol_compliant = has_final_answer and (not chain_of_thought or has_step)

    for method, segment in _answer_segments(raw_response):
        answer = _unique_candidate_match(segment, candidates)
        if answer is not None:
            return AnswerExtraction(answer, method, has_final_answer, protocol_compliant)

    return AnswerExtraction("", "none", has_final_answer, protocol_compliant)

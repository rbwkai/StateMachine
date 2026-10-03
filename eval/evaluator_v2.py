"""Thin re-export module for backward compatibility."""

from __future__ import annotations

from eval.scoring import (
    AnswerExtraction,
    _unique_candidate_match,
    candidate_answers,
    extract_instance_answer,
    extract_step_answers,
    normalize_text,
    score_prediction,
)

__all__ = [
    "normalize_text",
    "AnswerExtraction",
    "_unique_candidate_match",
    "extract_instance_answer",
    "extract_step_answers",
    "candidate_answers",
    "score_prediction",
]

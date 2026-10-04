"""
eval/scoring.py
================
Single authoritative scoring module for DWS-Bench.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_WARNED_RQ2_CANDIDATES = False


def normalize_text(value: object) -> str:
    """Normalize candidate labels without changing their semantic content."""
    normalized = re.sub(r"\s+", " ", str(value).strip().lower()).strip(" .,!?:;\"'")
    return re.sub(r"^(?:the|a|an)\s+", "", normalized)


@dataclass(frozen=True)
class AnswerExtraction:
    answer: str
    method: str
    has_final_answer: bool
    protocol_compliant: bool
    semantic_correct: bool
    strict_correct: bool


def _is_placeholder(segment: str) -> bool:
    cleaned = segment.strip()
    if not cleaned:
        return True
    # Match placeholder patterns like <answer>, <container>, < answer >, etc.
    if re.fullmatch(r"<\s*(?:answer|container)\s*>", cleaned, re.IGNORECASE):
        return True
    # Match bare placeholder words
    if cleaned.lower() in {"<answer>", "<container>", "answer", "container"}:
        return True
    # Match "Answer:" or "Container:" prefix (with or without colon)
    if re.match(r"^(?:answer|container):?$", cleaned, re.IGNORECASE):
        return True
    # Match markdown bold placeholders
    if re.fullmatch(r"\*\*(?:answer|container)\*\*", cleaned, re.IGNORECASE):
        return True
    return False


def _unique_candidate_match(text: str, candidates: Sequence[str]) -> Optional[str]:
    norm_text = normalize_text(text)
    if not norm_text:
        return None

    cand_map: Dict[str, str] = {}
    for c in candidates:
        cn = normalize_text(c)
        if cn and cn not in cand_map:
            cand_map[cn] = str(c)

    if not cand_map:
        return None

    # 1. Exact normalized equality check
    if norm_text in cand_map:
        return cand_map[norm_text]

    # 2. Distinct word-boundary regex match
    matching_cands = []
    for cn, orig_c in cand_map.items():
        pattern = rf"\b{re.escape(cn)}\b"
        if re.search(pattern, norm_text, re.IGNORECASE):
            matching_cands.append(orig_c)

    if len(matching_cands) == 1:
        return matching_cands[0]

    return None


def _collect_object_nouns(
    instance: Optional[Dict[str, Any]] = None,
    object_types: Optional[Sequence[str]] = None,
) -> set[str]:
    nouns = {"key", "object", "item", "entity"}
    if object_types:
        for ot in object_types:
            if ot:
                nouns.add(str(ot).lower())
    if instance:
        trace = instance.get("canonical_trace") or instance.get("trace") or []
        for step in trace:
            if isinstance(step, dict) and step.get("obj_type"):
                nouns.add(str(step["obj_type"]).lower())
        obj_types = instance.get("object_types")
        if isinstance(obj_types, (list, tuple)):
            for ot in obj_types:
                if ot:
                    nouns.add(str(ot).lower())
    return nouns


def _fallback_answer_segments(
    raw_response: str,
    instance: Optional[Dict[str, Any]] = None,
    object_types: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, str]]:
    lines = [line.strip() for line in raw_response.splitlines() if line.strip()]

    base_nouns = _collect_object_nouns(instance=instance, object_types=object_types)
    nouns_pattern = "|".join(sorted((re.escape(n) for n in base_nouns), key=len, reverse=True))
    answer_pattern = re.compile(
        rf"(?:{nouns_pattern})\s+is\s+(?:now\s+)?"
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
    gold_answer: Optional[str] = None,
    instance: Optional[Dict[str, Any]] = None,
    object_types: Optional[Sequence[str]] = None,
    first_final_answer: bool = True,
) -> AnswerExtraction:
    """
    Extract one unique candidate from final answer line or fallback segments.
    """
    final_matches = list(
        re.finditer(
            r"final\s+answer\s*:\s*(.+?)(?:\n|$)",
            raw_response,
            re.IGNORECASE,
        )
    )
    has_final_answer = len(final_matches) > 0
    has_step = bool(re.search(r"(?:^|\n)\s*step\s*\d+\s*[:.)-]", raw_response, re.IGNORECASE))

    if has_final_answer:
        # The single scoring contract uses the first marker. Later markers may
        # be echoed or generated continuation text.
        final_match = final_matches[0]
        final_segment = final_match.group(1).strip()
        protocol_compliant = not chain_of_thought or has_step

        if _is_placeholder(final_segment):
            answer = ""
            method = "final_answer"
        else:
            cand_match = _unique_candidate_match(final_segment, candidates)
            if cand_match is not None:
                answer = cand_match
                method = "final_answer"
            else:
                answer = ""
                method = "final_answer"

        gold_norm = normalize_text(gold_answer) if gold_answer is not None else ""
        ans_norm = normalize_text(answer) if answer else ""
        semantic_correct = bool(gold_norm and ans_norm and ans_norm == gold_norm)
        strict_correct = semantic_correct and protocol_compliant

        return AnswerExtraction(
            answer=answer,
            method=method,
            has_final_answer=True,
            protocol_compliant=protocol_compliant,
            semantic_correct=semantic_correct,
            strict_correct=strict_correct,
        )

    # NO Final Answer line exists -> fall back to answer_sentence / last_line
    protocol_compliant = False

    for method, segment in _fallback_answer_segments(raw_response, instance=instance, object_types=object_types):
        cand_match = _unique_candidate_match(segment, candidates)
        if cand_match is not None:
            gold_norm = normalize_text(gold_answer) if gold_answer is not None else ""
            ans_norm = normalize_text(cand_match)
            semantic_correct = bool(gold_norm and ans_norm and ans_norm == gold_norm)
            return AnswerExtraction(
                answer=cand_match,
                method=method,
                has_final_answer=False,
                protocol_compliant=False,
                semantic_correct=semantic_correct,
                strict_correct=False,
            )

    gold_norm = normalize_text(gold_answer) if gold_answer is not None else ""
    return AnswerExtraction(
        answer="",
        method="none",
        has_final_answer=False,
        protocol_compliant=False,
        semantic_correct=False,
        strict_correct=False,
    )


def candidate_answers(
    instance: Dict[str, Any],
    dataset_context: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Build unique candidate answer display strings for an instance.

    candidate_answers(instance) = unique normalized(
        step_wise_gold_answers ∪ gold_answer ∪ final_state container display names if present
    )
    
    NOTE: Does NOT fall back to other instances' gold answers to avoid leakage.
    """
    raw_candidates: List[str] = []

    step_wise = instance.get("step_wise_gold_answers")
    if step_wise:
        raw_candidates.extend(str(s) for s in step_wise if s)
    # NOTE: We do NOT fall back to other instances' gold answers to avoid cross-instance leakage.
    # If step_wise_gold_answers is missing, we only use the instance's own gold_answer
    # and final_state container names.

    ga = instance.get("gold_answer")
    if ga is not None:
        raw_candidates.append(str(ga))

    final_state = instance.get("final_state", {})
    if isinstance(final_state, dict):
        container_names = final_state.get("container_names") or final_state.get("container_display_names")
        if container_names:
            if isinstance(container_names, dict):
                raw_candidates.extend(str(v) for v in container_names if v)

    unique: Dict[str, str] = {}
    for value in raw_candidates:
        norm = normalize_text(value)
        if norm and norm not in unique:
            unique[norm] = str(value)

    return list(unique.values())


def extract_step_answers(
    raw_response: str,
    candidate_answers: Sequence[str],
    num_steps: Optional[int] = None,
) -> List[Optional[str]]:
    r"""
    Extract step-wise container predictions from model responses.

    Gold step 1 is the initial Put location (state after op 0).

    Parses only lines matching ^\s*step\s*\d+\s*[:.)-]\s*(.*)$ (case-insensitive, 'step' keyword required).
    Matches article-insensitively with word boundaries, returning None for unparseable steps.
    Returns the list aligned by step number.
    """
    step_pattern = re.compile(
        r"^\s*step\s*(\d+)\s*[:.)-]\s*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    )

    parsed_steps: Dict[int, Optional[str]] = {}

    for match in step_pattern.finditer(raw_response):
        step_num = int(match.group(1))
        step_text = match.group(2).strip()
        matched = _unique_candidate_match(step_text, candidate_answers)
        parsed_steps[step_num] = matched

    if not parsed_steps and num_steps is None:
        return []

    max_step = max(parsed_steps.keys()) if parsed_steps else 0
    total = num_steps if num_steps is not None else max_step

    return [parsed_steps.get(i + 1) for i in range(total)]


def score_prediction(
    prediction: Dict[str, Any],
    instance: Dict[str, Any],
    chain_of_thought: bool = False,
    dataset_context: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Score a prediction dictionary against an instance record.
    """
    cands = candidate_answers(instance, dataset_context=dataset_context)
    raw_pred = str(prediction.get("raw_prediction") or prediction.get("pred_answer") or "").strip()
    gold_answer = instance.get("gold_answer")

    extraction = extract_instance_answer(
        raw_pred,
        cands,
        chain_of_thought=chain_of_thought,
        gold_answer=gold_answer,
        instance=instance,
        first_final_answer=True,
    )

    result = dict(prediction)
    result.update({
        "model": prediction.get("model") or instance.get("model"),
        "extracted_answer": extraction.answer,
        "is_correct": extraction.strict_correct,
        "is_correct_semantic": extraction.semantic_correct,
        "semantic_correct": extraction.semantic_correct,
        "strict_correct": extraction.strict_correct,
        "protocol_compliant": extraction.protocol_compliant,
        "answer_extracted": extraction.answer,
        "extraction_method": extraction.method,
        "has_final_answer": extraction.has_final_answer,
        "gold_answer": instance.get("gold_answer", ""),
        "family": instance.get("family", prediction.get("family", "")),
        "depth": instance.get("measured_factors", {}).get(
            "T_actual", instance.get("requested_factors", {}).get("T")
        ),
    })
    return result


def extract_answer(
    raw_response: str,
    candidate_containers: Optional[Sequence[str]] = None,
) -> str:
    """Lenient final-answer extraction for boolean and legacy analysis paths.

    Lives here, not in ``eval/eval_harness.py``, because AGENTS.md §5 makes this
    module the single owner of scoring and extraction. It differs from
    :func:`extract_instance_answer` in contract, not by accident: it accepts a
    bare ``Answer:`` prefix and coerces boolean answers to the candidate set
    ``{"True", "False"}``. Protocol-strict extraction stays in
    :func:`extract_instance_answer`.
    """
    cands = list(candidate_containers) if candidate_containers else []
    ans_match = re.search(
        r"(?:final\s+)?answer\s*:\s*(.+?)(?:\n|$)", raw_response, re.IGNORECASE
    )
    if not cands and ans_match:
        seg = ans_match.group(1).strip().lower()
        if seg in {"true", "false"}:
            cands = ["True", "False"]
    extraction = extract_instance_answer(raw_response, candidates=cands)
    if extraction.answer and (extraction.has_final_answer or ans_match):
        return extraction.answer
    return ""

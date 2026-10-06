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

# Prompt slot text is owned by eval/prompts.py (AGENTS.md §5). prompts imports
# nothing from eval, so this cannot cycle.
from eval.prompts import ANSWER_SLOTS

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


# One "Step k: ..." line of a chain-of-thought response.
_STEP_LINE = re.compile(
    r"^[ \t]*step[ \t]*([0-9]+)[ \t]*[:.)-][ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE | re.ASCII,
)


# ------------------------------------------------------------
# Markdown / LaTeX unwrapping of an extracted answer segment.
#
# Models wrap answers as "**1**", "`1`", "$1$", "\\boxed{1}", "\\(1\\)" and so on.
# The unwrap is plain string slicing (no regex over untrusted text, AGENTS.md
# §6.13): each pass is O(n) and the pass count is capped, so adversarial input
# such as "*" * 50000 stays linear. Only wrappers are removed; the inner text
# still goes through the count grammar or candidate matching, so a hedge such
# as "**about 2**" is still rejected. Leading punctuation is never stripped:
# ".5" must not become "5".
# ------------------------------------------------------------
_UNWRAP_EDGE_CHARS = " \t\r*_`$"
_UNWRAP_TRAILING_PUNCT = " \t\r.,;!?"
_UNWRAP_LATEX_COMMANDS: Tuple[str, ...] = ("\\boxed{", "\\textbf{", "\\text{")
_UNWRAP_LATEX_DELIMS: Tuple[Tuple[str, str], ...] = (("\\(", "\\)"), ("\\[", "\\]"))
_UNWRAP_MAX_PASSES = 8


def unwrap_answer_segment(segment: str) -> str:
    """Strip markdown emphasis, code ticks and LaTeX wrappers around an answer."""
    text = str(segment)
    for _ in range(_UNWRAP_MAX_PASSES):
        before = text
        text = text.rstrip(_UNWRAP_TRAILING_PUNCT).strip(_UNWRAP_EDGE_CHARS)
        for command in _UNWRAP_LATEX_COMMANDS:
            if text.startswith(command) and text.endswith("}"):
                text = text[len(command):-1]
                break
        for opener, closer in _UNWRAP_LATEX_DELIMS:
            if (
                len(text) >= len(opener) + len(closer)
                and text.startswith(opener)
                and text.endswith(closer)
            ):
                text = text[len(opener):-len(closer)]
                break
        if text == before:
            break
    return text.strip()


def _slot_payload(slot: str) -> str:
    """The placeholder part of a prompt slot ("Final Answer: <x>" -> "x")."""
    head, sep, tail = slot.partition(":")
    payload = tail if sep and normalize_text(head) == "final answer" else slot
    return normalize_text(payload.strip().strip("<>"))


# Every answer slot the prompts print. An echoed slot is template text, not an
# answer (bug: "Final Answer: <a single integer>" used to win over a real
# later marker).
_PROMPT_SLOT_PLACEHOLDERS = frozenset(
    _slot_payload(slot)
    for slots in ANSWER_SLOTS.values()
    for slot in slots.values()
)
_GENERIC_PLACEHOLDERS = frozenset({"answer", "container"})


def _is_placeholder(segment: str) -> bool:
    """True for an empty segment or an echoed template slot.

    Recognises the generic "<answer>" / "<container>" forms (with or without
    angle brackets, markdown emphasis or a trailing colon) and every slot in
    ``eval.prompts.ANSWER_SLOTS``.
    """
    cleaned = unwrap_answer_segment(segment)
    if not cleaned:
        return True
    # Bounded character strip, not a regex: "< answer >" -> "answer".
    inner = normalize_text(cleaned.strip("<> \t").rstrip(":"))
    if not inner:
        return True
    return inner in _GENERIC_PLACEHOLDERS or inner in _PROMPT_SLOT_PLACEHOLDERS


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


# Number words a count answer may be written with. Bounded table, not a general
# parser: the input is untrusted text and must never reach eval/exec
# (AGENTS.md §6.13).
_COUNT_NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

# ------------------------------------------------------------
# Count-answer grammar (one rule for digits and number words).
#
#   [neutral prefix] NUMBER [unit noun] [neutral suffix]
#
#   neutral prefix : "there are" | "there is" | "total of"  ("a total of"
#                    reaches here as "total of": normalize_text drops the article)
#   NUMBER         : ASCII digits [0-9]+ (leading zeros dropped, "007" -> "7")
#                    or a word from _COUNT_NUMBER_WORDS
#   unit noun      : a generic count noun (_GENERIC_COUNT_NOUNS) or the object
#                    noun of the instance's question, singular or plural
#   neutral suffix : "in total" | "total" | "of them" | "altogether"
#
# Anything else is not credited. In particular hedges and qualifiers
# ("about 2", "around two", "approximately 2", "exactly 1", "1 or 2", "2 maybe",
# "3 not", "2 each") are rejected for digits and number words alike: a hedged
# count must never be scored as the gold integer. Non-ASCII digits ("١٢",
# "１２") are rejected rather than converted.
#
# The reader tokenizes on single spaces after normalize_text has collapsed
# whitespace, so there is no backtracking regex over untrusted input.
# ------------------------------------------------------------
_COUNT_PREFIXES: Tuple[Tuple[str, ...], ...] = (
    ("there", "are"), ("there", "is"), ("total", "of"),
)
_COUNT_SUFFIXES: Tuple[Tuple[str, ...], ...] = (
    ("in", "total"), ("of", "them"), ("total",), ("altogether",),
)
# "token(s)" stays generic because checklist 6 names "2 tokens" as a numeric
# answer form that must be read (it was the original unit word).
_GENERIC_COUNT_NOUNS = frozenset({
    "object", "objects", "item", "items", "thing", "things", "ones",
    "token", "tokens",
})
_ASCII_DIGITS = re.compile(r"[0-9]+", re.ASCII)
# "How many <plural noun> are in ...": render/narrative.py::question_count.
_QUESTION_NOUN = re.compile(r"how many ([a-z-]+) (?:are|is) ", re.ASCII)


def _question_nouns(question: Optional[str]) -> frozenset:
    """Singular and plural forms of the object noun a count question asks about."""
    if not question:
        return frozenset()
    m = _QUESTION_NOUN.search(normalize_text(question))
    if not m:
        return frozenset()
    plural = m.group(1)
    forms = {plural}
    # Inverse of render.names.pluralize_object for the endings it produces.
    if plural.endswith("ies"):
        forms.add(plural[:-3] + "y")
    if plural.endswith("es"):
        forms.add(plural[:-2])
    if plural.endswith("s"):
        forms.add(plural[:-1])
    return frozenset(f for f in forms if f)


def _strip_prefix(tokens: List[str], options: Sequence[Tuple[str, ...]]) -> List[str]:
    for opt in options:
        if tuple(tokens[: len(opt)]) == opt:
            return tokens[len(opt):]
    return tokens


def _strip_suffix(tokens: List[str], options: Sequence[Tuple[str, ...]]) -> List[str]:
    for opt in options:
        if len(tokens) > len(opt) and tuple(tokens[-len(opt):]) == opt:
            return tokens[: -len(opt)]
    return tokens


def _count_token(token: str) -> Optional[str]:
    if _ASCII_DIGITS.fullmatch(token):
        return token.lstrip("0") or "0"
    return _COUNT_NUMBER_WORDS.get(token)


def _numeric_answer_space(gold_answer: Optional[object]) -> bool:
    """True when this instance is answered with a bare number (a count cell).

    A location cell's gold is a container display name, so this stays false for
    it and the numeric fallback never competes with candidate matching.
    """
    if gold_answer is None:
        return False
    return bool(_ASCII_DIGITS.fullmatch(normalize_text(gold_answer)))


def read_count_answer(segment: str, question: Optional[str] = None) -> Optional[str]:
    """Read a bare count answer such as ``"2"``, ``"two"`` or ``"12 phones"``.

    Returns the answer as a canonical digit string, or ``None`` when ``segment``
    does not follow the count grammar above. A wrong count is still returned
    (as that count) so it is recorded as a wrong guess instead of being dropped
    for not matching a candidate (checklist 6).

    ``question`` is the instance's question text; its object noun is the only
    non-generic unit word accepted ("2 keys" for "How many keys ...?"). Without
    a question only the generic nouns are accepted.

    Markdown / LaTeX wrappers ("**2**", "\\boxed{2}", "$2$") are removed first
    by :func:`unwrap_answer_segment`; the unwrapped text must still follow the
    grammar, so "**about 2**" is rejected.
    """
    # Emphasis inside the segment ("**2** keys") carries no content for a count.
    unwrapped = (
        unwrap_answer_segment(segment).replace("*", "").replace("`", "").strip()
    )
    # normalize_text trims a leading "." and the grammar ignores "-", so ".5"
    # and "-1" would otherwise read as 5 and 1; a signed or fractional
    # number is never a count.
    if unwrapped[:1] in (".", "-", "+") and unwrapped[1:2].isdigit():
        return None
    text = normalize_text(unwrapped)
    if not text:
        return None
    tokens = text.split(" ")
    tokens = _strip_prefix(tokens, _COUNT_PREFIXES)
    tokens = _strip_suffix(tokens, _COUNT_SUFFIXES)
    if not tokens or len(tokens) > 2:
        return None
    value = _count_token(tokens[0])
    if value is None:
        return None
    if len(tokens) == 2:
        unit = tokens[1]
        if unit not in _GENERIC_COUNT_NOUNS and unit not in _question_nouns(question):
            return None
    return value


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


# "Final Answer:" marker. Tolerates markdown emphasis around the marker and the
# colon ("**Final Answer:** 1", "**Final Answer**: 1", "Final Answer**:** 1");
# emphasis left in the payload ("1**") is removed by unwrap_answer_segment.
# No two adjacent quantifiers share characters, so nothing backtracks. The
# payload is the rest of the marker's own line; an empty one is resolved by
# _final_answer_segments, which reads the next line instead.
_FINAL_ANSWER_MARKER = re.compile(
    r"final[ \t]+answer[*_]{0,3}[ \t]*:[*_]{0,3}[ \t]*([^\n]*)",
    re.IGNORECASE,
)


def _final_answer_segments(raw_response: str) -> List[str]:
    """Payload of every "Final Answer:" marker, in order.

    A marker with nothing after it on its line takes the next line instead
    ("Final Answer:\n2"), unless that line is itself a marker. A marker with
    no payload on either line (a reply truncated right after "Final Answer:")
    is dropped, so the reply falls through to fallback extraction.
    """
    segments: List[str] = []
    for match in _FINAL_ANSWER_MARKER.finditer(raw_response):
        payload = match.group(1).strip()
        if not payload:
            end = match.end()
            if raw_response.startswith("\r\n", end):
                end += 2
            elif raw_response.startswith("\n", end):
                end += 1
            else:
                continue
            newline = raw_response.find("\n", end)
            next_line = raw_response[end:] if newline == -1 else raw_response[end:newline]
            if _FINAL_ANSWER_MARKER.search(next_line):
                continue
            payload = next_line.strip()
        if payload:
            segments.append(payload)
    return segments


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

    ``instance`` (when given) supplies the question text, whose object noun is
    the only non-generic unit word a count answer may carry.

    ``first_final_answer`` selects among "Final Answer:" markers whose payload
    is not an echoed placeholder (see :func:`_is_placeholder`): True (the
    scoring contract, used by :func:`score_prediction`) takes the first such
    marker, since later markers may be echoed or continuation text; False takes
    the last. When every marker is a placeholder the answer is empty.
    """
    question = str(instance.get("question") or "") if instance else None
    final_segments = _final_answer_segments(raw_response)
    has_final_answer = len(final_segments) > 0
    has_step = _STEP_LINE.search(raw_response) is not None

    if has_final_answer:
        real_segments = [seg for seg in final_segments if not _is_placeholder(seg)]
        if real_segments:
            final_segment = real_segments[0] if first_final_answer else real_segments[-1]
        else:
            final_segment = ""
        protocol_compliant = not chain_of_thought or has_step

        if not final_segment:
            answer = ""
            method = "final_answer"
        elif _numeric_answer_space(gold_answer):
            # Numeric answer space: skip candidate matching entirely. Parse the
            # segment as a bare count. A wrong integer is recorded as that
            # integer; ambiguous/unparseable segments get method="ambiguous".
            guess = read_count_answer(final_segment, question=question)
            if guess is not None:
                answer = guess
                method = "final_answer"
            else:
                answer = ""
                method = "ambiguous"
        else:
            cand_match = _unique_candidate_match(
                unwrap_answer_segment(final_segment), candidates
            )
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

    numeric = _numeric_answer_space(gold_answer)
    saw_segment = False
    for method, segment in _fallback_answer_segments(raw_response, instance=instance, object_types=object_types):
        saw_segment = True
        if numeric:
            # Numeric answer space: skip candidate matching. Parse as bare count.
            # An unparseable segment (e.g. a location sentence) does not end
            # the search: a later segment such as the last line may still
            # hold the count. "ambiguous" is decided after the loop.
            guess = read_count_answer(segment, question=question)
            if guess is not None:
                gold_norm = normalize_text(gold_answer) if gold_answer is not None else ""
                ans_norm = normalize_text(guess)
                semantic_correct = bool(gold_norm and ans_norm and ans_norm == gold_norm)
                return AnswerExtraction(
                    answer=guess,
                    method=method,
                    has_final_answer=False,
                    protocol_compliant=False,
                    semantic_correct=semantic_correct,
                    strict_correct=False,
                )
        else:
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

    return AnswerExtraction(
        answer="",
        # Numeric space with text but no readable count: ambiguous, not absent.
        method="ambiguous" if numeric and saw_segment else "none",
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
                raw_candidates.extend(str(v) for v in container_names.values() if v)

    unique: Dict[str, str] = {}
    for value in raw_candidates:
        norm = normalize_text(value)
        if norm and norm not in unique:
            unique[norm] = str(value)

    return list(unique.values())


def _numeric_step_space(
    candidate_answers: Sequence[str],
    gold_answer: Optional[str],
    instance: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when step answers are counts.

    Decided, in order, from the instance's query type, from ``gold_answer``,
    or from the candidates being all bare integers. The candidate rule alone is
    not enough for a real record: ``candidate_answers(instance)`` also carries
    the final-state container names, so a count record never looks all-numeric.
    """
    if instance is not None:
        # Function-local: eval.baselines imports this module, and
        # query_type_of is the single owner of the query-type lookup.
        from eval.baselines import query_type_of

        return query_type_of(instance) == "count"
    if gold_answer is not None:
        return _numeric_answer_space(gold_answer)
    cands = [c for c in candidate_answers if normalize_text(c)]
    return bool(cands) and all(_numeric_answer_space(c) for c in cands)


def count_step_lines(raw_response: str) -> int:
    """Number of "Step k" lines in a response, before any alignment."""
    return sum(1 for _ in _STEP_LINE.finditer(raw_response))


def extract_step_answers(
    raw_response: str,
    candidate_answers: Sequence[str],
    num_steps: Optional[int] = None,
    gold_answer: Optional[str] = None,
    instance: Optional[Dict[str, Any]] = None,
) -> List[Optional[str]]:
    r"""
    Extract step-wise predictions (containers or counts) from model responses.

    Step k is the state after the k-th narrated sentence, i.e. it pairs with
    ``step_wise_gold_answers[k-1]`` (no initial-state offset).

    Parses only lines matching ^\s*step\s*\d+\s*[:.)-]\s*(.*)$ (case-insensitive, 'step' keyword required).
    Location steps match a unique candidate article-insensitively with word
    boundaries. Count steps are parsed whole-segment with
    :func:`read_count_answer`, so hedges like "1 or 2" or "1.5" are not credited
    as a substring match of the gold integer. Unparseable steps are None.
    Returns the list aligned by step number. ``instance`` decides the answer
    space (query type) and supplies the question for count unit words.
    """
    numeric = _numeric_step_space(candidate_answers, gold_answer, instance)
    question = str(instance.get("question") or "") if instance else None

    parsed_steps: Dict[int, Optional[str]] = {}

    for match in _STEP_LINE.finditer(raw_response):
        step_num = int(match.group(1))
        step_text = match.group(2).strip()
        if numeric:
            matched = read_count_answer(step_text, question=question) if step_text else None
        else:
            matched = _unique_candidate_match(step_text, candidate_answers)
        parsed_steps[step_num] = matched

    if not parsed_steps and num_steps is None:
        return []

    max_step = max(parsed_steps.keys()) if parsed_steps else 0
    total = num_steps if num_steps is not None else max_step

    return [parsed_steps.get(i + 1) for i in range(total)]


def step_coverage_of(
    parsed_steps: Optional[Sequence[Optional[str]]],
    num_gold_steps: int,
) -> Optional[float]:
    """Fraction of gold step events with a parsed (non-None) step answer."""
    if parsed_steps is None or num_gold_steps <= 0:
        return None
    parsed = sum(1 for step in parsed_steps[:num_gold_steps] if step is not None)
    return parsed / num_gold_steps


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

    # Step coverage: parsed (non-None) steps over gold step events. Reported
    # beside protocol_compliant, which keeps its own meaning. Step lines are
    # only requested under CoT, so a non-CoT run has no coverage to report.
    gold_steps = instance.get("step_wise_gold_answers") or []
    step_coverage: Optional[float] = None
    if chain_of_thought and gold_steps:
        parsed_steps = extract_step_answers(
            raw_pred, cands, num_steps=len(gold_steps), gold_answer=gold_answer,
            instance=instance,
        )
        step_coverage = step_coverage_of(parsed_steps, len(gold_steps))

    result = dict(prediction)
    result.update({
        "model": prediction.get("model") or instance.get("model"),
        "extracted_answer": extraction.answer,
        "is_correct": extraction.strict_correct,
        "is_correct_semantic": extraction.semantic_correct,
        "semantic_correct": extraction.semantic_correct,
        "strict_correct": extraction.strict_correct,
        "protocol_compliant": extraction.protocol_compliant,
        "step_coverage": step_coverage,
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

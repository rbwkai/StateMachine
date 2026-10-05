"""
generator/instance.py
=====================
The ONE request -> validated-record path for DWS-Bench.

Two producers used to assemble their own record from the same generators:
``experiments/_common.py::generate_instance`` and
``generate.py::generate_family_example``. They disagreed on the key set, each
carried its own copy of the ``hasattr`` chain that serialises an operation, and
neither of them ever called ``metadata.verify_factors`` or
``metadata.verify_length``. The hard generation gate of AGENTS.md §6.3 and
requirements.md §7 therefore existed only as a test-time contract.

``build_validated_instance`` is now the only place that turns a generation
request into a record, and it runs the SPEC §3 gate in order:

    build_trajectory
        -> render_narrative
        -> measure_factors
        -> validate_structural_causality
        -> verify_factors
        -> verify_length

A rejected request comes back as a structured ``InstanceGateFailure`` naming the
check that fired and the requested-versus-measured values; it is never swallowed
into a bare ``ValueError`` and never silently skipped.
``generate_instance_with_retry`` counts those reasons and raises with the
histogram once the attempt budget is spent.

Determinism (hard rule 5): every draw comes from a ``random.Random`` passed in by
the caller, and this module contains no ``random.*`` module call. Layering
(AGENTS.md §4): the only imports are ``world``, ``render`` and this package's own
modules, so ``generator`` stays importable without torch/transformers.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from world import (
    GenerationError,
    Merge,
    Operation,
    Put,
    gold_count,
    replay_trace,
)

from render.names import (
    NameRegistry,
    make_distractor_sentences,
    splice_distractors,
)
from render.narrative import question_location, question_count, render_narrative

from .constants import (
    GENERATOR_VERSION,
    RENDERER_VERSION,
    SCORING_VERSION,
)
from .metadata import (
    MeasuredFactors,
    count_query_target,
    measure_factors,
    verify_factors,
    verify_length,
)
from .probes import CountQuery
from .structural import validate_structural_causality
from .trajectories import build_trajectory
from .trajectory_specs import TrajectorySpec


# ============================================================
# Gate vocabulary
# ============================================================

# Stable strings rather than an enum: these labels end up verbatim in the
# GenerationError histogram an operator reads when a condition turns out to be
# unreachable, so they are part of the release interface.
CHECK_BUILD = "build_trajectory"
CHECK_RENDER = "render_narrative"
CHECK_MEASURE = "measure_factors"
CHECK_STRUCTURE = "structural_causality"
CHECK_FACTORS = "verify_factors"
CHECK_LENGTH = "verify_length"
CHECK_LEAKAGE = "answer_leakage"
CHECK_DUPLICATE = "duplicate_trace"

# Retry budget, unchanged from the loop this module replaces. What changed is
# that the loop now reports why every attempt was rejected.
DEFAULT_MAX_ATTEMPTS = 50

# How many concrete failure records ride along in the exhaustion message: enough
# to diagnose a systematic cause, short enough to stay readable.
FAILURE_REPORT_LIMIT = 3

# Global deduplication registry: trace_hash -> instance_id that first produced it
# This is populated during generation and used to reject duplicates.
_SEEN_TRACE_HASHES: Dict[str, str] = {}


def reset_deduplication_registry() -> None:
    """Clear the deduplication registry (for testing or new generation runs)."""
    _SEEN_TRACE_HASHES.clear()


def get_deduplication_stats() -> Dict[str, int]:
    """Return deduplication statistics."""
    return {"unique_traces": len(_SEEN_TRACE_HASHES)}


# The record schema, declared once. Both producers build their record in
# build_validated_instance() and that function asserts this exact key order, so
# "the two producers agree" is enforced by the code rather than by convention.
INSTANCE_RECORD_KEYS: Tuple[str, ...] = (
    "instance_id",
    "family",
    "experiment",
    "condition_id",
    "seed",
    "trace_hash",
    "attempt",
    "generator_version",
    "renderer_version",
    "scoring_version",
    "requested_factors",
    "measured_factors",
    "spec",
    "canonical_trace",
    "sentences",
    "context",
    "question",
    "query_entity",
    "gold_container",
    "gold_answer",
    "step_wise_gold",
    "step_wise_gold_answers",
    "final_state",
)


# ============================================================
# Gate results
# ============================================================

@dataclass(frozen=True)
class InstanceGateFailure:
    """
    One rejected generation request.

    ``requested`` and ``measured`` share a key space (``E``, ``T``, ``D``, ``V``,
    ``V_min``, ``L_word``) so the disagreement can be read off directly instead
    of being parsed back out of an exception message. ``measured`` is empty for
    a stage that failed before measurement was possible.
    """

    check: str
    family: str
    instance_id: str
    message: str
    requested: Mapping[str, Any]
    measured: Mapping[str, Any]

    @property
    def reason(self) -> str:
        """
        Histogram label: the failing check plus the symbols that disagreed.
        """
        symbols = {
            key
            for key in self.requested
            if key in self.measured and self.requested[key] != self.measured[key]
        }
        # A floor (V_min) is a different comparison from an equality check, so it
        # never surfaces as an unequal value; name its base symbol instead.
        symbols |= {
            key[: -len("_min")]
            for key in self.requested
            if key.endswith("_min") and key[: -len("_min")] in self.measured
        }
        if not symbols:
            return self.check
        return f"{self.check}:{','.join(sorted(symbols))}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "family": self.family,
            "instance_id": self.instance_id,
            "message": self.message,
            "requested": dict(self.requested),
            "measured": dict(self.measured),
        }

    def describe(self) -> str:
        # No instance-id prefix here: the exhaustion message already names it,
        # and verify_factors/verify_length embed it in their own message too.
        return (
            f"[{self.check}] {self.message} "
            f"(requested={dict(self.requested)} measured={dict(self.measured)})"
        )


@dataclass(frozen=True)
class InstanceResult:
    """Either a JSON-safe record or a structured gate failure, never both."""

    record: Optional[Dict[str, Any]] = None
    failure: Optional[InstanceGateFailure] = None
    measured: Optional[MeasuredFactors] = None

    @property
    def ok(self) -> bool:
        return self.record is not None


def _failure(
    check: str,
    spec: TrajectorySpec,
    instance_id: str,
    exc: BaseException,
    requested: Mapping[str, Any],
    measured: Mapping[str, Any],
) -> InstanceGateFailure:
    return InstanceGateFailure(
        check=check,
        family=spec.family,
        instance_id=instance_id,
        message=str(exc) or repr(exc),
        requested=dict(requested),
        measured=dict(measured),
    )


def _requested_factors(spec: TrajectorySpec) -> Dict[str, int]:
    """The designed factors, read off the spec and therefore never trusted."""
    return {
        "E": spec.entity_count,
        "T": spec.target_updates,
        "D": spec.distractor_updates,
    }


def _measured_view(measured: MeasuredFactors) -> Dict[str, int]:
    """Measured factors under the same keys as the request, for the failure report."""
    return {
        "E": measured.E_actual,
        "T": measured.T_actual,
        "D": measured.D_actual,
        "V": measured.V_actual,
        "L_word": measured.L_word,
    }


def _gate_request(
    requested: Mapping[str, int],
    min_v: Optional[int],
    intended_v: Optional[int],
) -> Dict[str, Any]:
    """Requested factors plus whichever V qualification the caller asked for."""
    gate_requested: Dict[str, Any] = dict(requested)
    if min_v is not None:
        gate_requested["V_min"] = min_v
    if intended_v is not None:
        gate_requested["V"] = intended_v
    return gate_requested


# ============================================================
# Canonical trace serialisation
# ============================================================

def _canonical_trace(ops: Sequence[Operation]) -> List[Dict[str, Any]]:
    """
    Serialise the symbolic trace to plain JSON dicts.

    The attribute order below is FROZEN. ``trace_hash`` is a sha1 over this
    structure and instances are deduplicated by canonical trace (hard rule 10),
    so reordering a branch or renaming a field would silently change every hash
    in a released dataset. Extend this only with a SPEC version bump.
    """
    canonical_trace: List[Dict[str, Any]] = []
    for op in ops:
        d = {
            "op_type": type(op).__name__.upper(),
        }
        if hasattr(op, "obj_id"):
            d["obj_id"] = op.obj_id
        if hasattr(op, "dst"):
            d["dst"] = op.dst
        if hasattr(op, "obj_type"):
            d["obj_type"] = op.obj_type
        if hasattr(op, "container"):
            d["container"] = op.container
        if hasattr(op, "src_container"):
            d["src_container"] = op.src_container
        if hasattr(op, "dst_container"):
            d["dst_container"] = op.dst_container
        if hasattr(op, "container_a"):
            d["container_a"] = op.container_a
        if hasattr(op, "container_b"):
            d["container_b"] = op.container_b
        if hasattr(op, "source_obj_id"):
            d["source_obj_id"] = op.source_obj_id
        if hasattr(op, "new_obj_id"):
            d["new_obj_id"] = op.new_obj_id
        canonical_trace.append(d)
    return canonical_trace


# ============================================================
# Answer leakage validator
# ============================================================

def _check_answer_leakage(
    sentences: List[str],
    gold_answer: Optional[str],
    num_op_sentences: int,
    *,
    suffix_k: int = 3,
) -> bool:
    """
    Check if the gold answer appears in the final ``suffix_k`` sentences
    that are *not* operation descriptions (i.e., in distractor/tail sentences).

    Returns True if leakage is detected (answer found in tail), False otherwise.
    """
    if not gold_answer:
        return False

    norm_gold = gold_answer.lower().strip()
    # Only check sentences beyond the operation descriptions (distractors/tail)
    distractor_sentences = sentences[num_op_sentences:]
    if not distractor_sentences:
        return False

    # Check last k distractor sentences
    tail = distractor_sentences[-suffix_k:] if len(distractor_sentences) >= suffix_k else distractor_sentences

    for sent in tail:
        sent_lower = sent.lower()
        if norm_gold in sent_lower:
            return True
    return False


# ============================================================
# The gate
# ============================================================

def build_validated_instance(
    rng: random.Random,
    spec: TrajectorySpec,
    name_rng: random.Random,
    textual_distractor_count: int = 0,
    *,
    instance_id: str = "",
    experiment: str = "",
    condition_id: str = "",
    seed: Optional[int] = None,
    attempt: int = 0,
    min_v: Optional[int] = None,
    intended_v: Optional[int] = None,
) -> InstanceResult:
    """
    Turn one generation request into a validated instance record.

    Parameters
    ----------
    rng:
        Trajectory stream. ``build_trajectory`` draws its operations from it.
    spec:
        The requested trajectory shape; also the source of the requested $E,T,D$
        the gate checks the measurement against (SPEC §2).
    name_rng:
        Separate display-name stream, so a name change cannot shift a trace.
    textual_distractor_count:
        Pure-text distractor sentences to splice in (factor $N$, SPEC §2).

    Returns
    -------
    InstanceResult
        ``record`` on success, ``failure`` on rejection. Neither a bare
        ``ValueError`` nor a silent skip.
    """
    requested = _requested_factors(spec)
    gate_requested = _gate_request(requested, min_v, intended_v)

    # --------------------------------------------------------
    # 1. Trajectory (SPEC §3 stage 1-4: constructor, canonical
    # replay, constructor/replay equality, family validation).
    # --------------------------------------------------------

    try:
        trajectory = build_trajectory(rng, spec)
    except Exception as exc:
        return InstanceResult(
            failure=_failure(CHECK_BUILD, spec, instance_id, exc, gate_requested, {})
        )

    # --------------------------------------------------------
    # 2. Narrative (SPEC §3 rendering). The renderer never
    # re-derives state; it consumes the canonical replay.
    # --------------------------------------------------------

    try:
        names = NameRegistry(containers=trajectory.containers, rng=name_rng)
        op_sentences, final_state = render_narrative(
            trajectory.ops,
            trajectory.containers,
            names,
            include_move_sources=False,
        )

        # Compute step-wise gold for operation sentences (before splicing distractors)
        # Read through the same probe as the final question, so a count cell's
        # step-wise gold is a count at every step. Previously it always read the
        # target's container, which the CoT condition could not match against a
        # numeric gold.
        trace, _, _ = replay_trace(trajectory.ops, trajectory.containers)
        if spec.query_type == "count":
            probe = CountQuery(
                *count_query_target(
                    trajectory.ops, trajectory.final_state, trajectory.target_obj
                )
            )
            op_step_wise_gold_answers = [str(probe.read(after)) for _, _, after in trace]
        else:
            op_step_wise_gold_answers = [
                names.container(after.location.get(trajectory.target_obj))
                if after.location.get(trajectory.target_obj) is not None
                else "removed"
                for _, _, after in trace
            ]

        if textual_distractor_count:
            distractors = make_distractor_sentences(
                name_rng,
                textual_distractor_count,
                names,
                [op.obj_type for op in trajectory.ops if isinstance(op, Put)],
            )
            sentences = splice_distractors(name_rng, op_sentences, distractors)

            # Align step-wise gold with final narrative sentences.
            # Operation sentences are a subsequence of final sentences in order.
            # For distractor sentences, use the last operation's gold (no state change).
            aligned_step_wise_gold_answers = []
            op_idx = 0
            last_gold = op_step_wise_gold_answers[0] if op_step_wise_gold_answers else "removed"
            for sent in sentences:
                if op_idx < len(op_sentences) and sent == op_sentences[op_idx]:
                    # This is an operation sentence
                    if op_idx < len(op_step_wise_gold_answers):
                        gold = op_step_wise_gold_answers[op_idx]
                        aligned_step_wise_gold_answers.append(gold)
                        last_gold = gold
                    op_idx += 1
                else:
                    # This is a distractor sentence - state unchanged
                    aligned_step_wise_gold_answers.append(last_gold)
        else:
            sentences = op_sentences
            aligned_step_wise_gold_answers = op_step_wise_gold_answers

    except Exception as exc:
        return InstanceResult(
            failure=_failure(CHECK_RENDER, spec, instance_id, exc, gate_requested, {})
        )

    # --------------------------------------------------------
    # 3. Measured factors, now that sentences exist so that
    # $L_{word}$ is the rendered count rather than the -1
    # sentinel (RESOLVED-1).
    # --------------------------------------------------------

    try:
        measured = measure_factors(
            trajectory.ops,
            trajectory.containers,
            trajectory.target_obj,
            sentences=sentences,
            textual_distractor_count=textual_distractor_count,
        )
    except Exception as exc:
        return InstanceResult(
            failure=_failure(CHECK_MEASURE, spec, instance_id, exc, gate_requested, {})
        )

    measured_view = _measured_view(measured)

    # --------------------------------------------------------
    # 4a. Structural causality. build_trajectory() already runs
    # this validator; the second call is deliberate. SPEC §3
    # places structural causality in the *instance* gate, after
    # rendering, and requirements.md §8 asks for one
    # authoritative validator instead of a per-producer copy.
    # It is a pure function of (ops, containers, target, spec)
    # and consumes no RNG, so it cannot perturb the trace.
    # --------------------------------------------------------

    try:
        validate_structural_causality(
            trajectory.ops,
            trajectory.containers,
            trajectory.target_obj,
            spec,
        )
    except Exception as exc:
        return InstanceResult(
            failure=_failure(CHECK_STRUCTURE, spec, instance_id, exc, gate_requested, measured_view)
        )

    # --------------------------------------------------------
    # 4b. Requested E/T/D must equal the measurement; V is
    # gated only when the caller asks (hard rule 3, SPEC §2).
    # --------------------------------------------------------

    try:
        verify_factors(
            requested["E"],
            requested["T"],
            requested["D"],
            measured,
            spec.family,
            instance_id=instance_id,
            min_v=min_v,
            intended_v=intended_v,
        )
    except AssertionError as exc:
        return InstanceResult(
            failure=_failure(CHECK_FACTORS, spec, instance_id, exc, gate_requested, measured_view)
        )

    # --------------------------------------------------------
    # 4c. Rendered $L_{word}$ ceiling $L_{max}$ (RESOLVED-1).
    # --------------------------------------------------------

    try:
        verify_length(measured, instance_id=instance_id)
    except AssertionError as exc:
        return InstanceResult(
            failure=_failure(CHECK_LENGTH, spec, instance_id, exc, gate_requested, measured_view)
        )

    # --------------------------------------------------------
    # 5. Record assembly, from the one canonical replay pass
    # (hard rule 2): gold, narration and probes read the same
    # states, so they cannot drift apart.
    # --------------------------------------------------------

    canonical_trace = _canonical_trace(trajectory.ops)
    trace_hash = hashlib.sha1(
        json.dumps(canonical_trace, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    trace, _, _ = replay_trace(trajectory.ops, trajectory.containers)
    step_wise_gold = [
        after.location.get(trajectory.target_obj)
        for _, _, after in trace
    ]
    # Use aligned step-wise gold that accounts for distractors
    step_wise_gold_answers = aligned_step_wise_gold_answers

    question = question_location(trajectory.target_obj, final_state, names)
    target_container = final_state.location.get(trajectory.target_obj)
    gold_answer = names.container(target_container) if target_container else None

    # A count cell asks a counting question and answers with a number. Every
    # count family goes through this one branch; previously it was hard-coded to
    # split_chain, so undo_chain, swap_chain and merge_chain built with
    # query_type="count" still rendered "Where is ...?" with a container as gold.
    if spec.query_type == "count":
        count_container, count_type = count_query_target(
            trajectory.ops, final_state, trajectory.target_obj
        )
        question = question_count(count_container, count_type, names)
        gold_answer = str(gold_count(final_state, count_container, count_type))
        target_container = count_container

    # --------------------------------------------------------
    # 4d. Answer leakage gate: reject if gold answer appears
    # in the generated distractor sentences (before splicing).
    # --------------------------------------------------------

    try:
        if textual_distractor_count and _check_answer_leakage(distractors, gold_answer, 0):
            raise AssertionError(
                f"gold answer '{gold_answer}' appears in distractor sentences"
            )
    except AssertionError as exc:
        return InstanceResult(
            failure=_failure(CHECK_LEAKAGE, spec, instance_id, exc, gate_requested, measured_view)
        )

    # --------------------------------------------------------
    # 4e. Deduplication gate: reject if trace_hash already seen.
    # Register hash only after all checks pass to avoid poisoning registry.
    # --------------------------------------------------------

    try:
        if trace_hash in _SEEN_TRACE_HASHES:
            raise AssertionError(
                f"duplicate trace_hash {trace_hash[:12]}... "
                f"(first seen in {_SEEN_TRACE_HASHES[trace_hash]})"
            )
        _SEEN_TRACE_HASHES[trace_hash] = instance_id
    except AssertionError as exc:
        return InstanceResult(
            failure=_failure(CHECK_DUPLICATE, spec, instance_id, exc, gate_requested, measured_view)
        )

    initial_placements = sum(
        1 for op in trajectory.ops if isinstance(op, Put)
    )

    record: Dict[str, Any] = {
        "instance_id": instance_id,
        "family": spec.family,
        "experiment": experiment,
        "condition_id": condition_id,
        "seed": seed,
        "trace_hash": trace_hash,
        "attempt": attempt,

        # Release metadata (hard rule 10): a regenerated grid must never be
        # mistaken for the frozen one it replaces.
        "generator_version": GENERATOR_VERSION,
        "renderer_version": RENDERER_VERSION,
        "scoring_version": SCORING_VERSION,

        "requested_factors": requested,
        "measured_factors": measured.to_dict(),

        "spec": {
            "entity_count": spec.entity_count,
            "target_updates": spec.target_updates,
            "distractor_updates": spec.distractor_updates,
            "num_containers": spec.num_containers,
            "total_updates": spec.total_updates,
            "initial_placements": initial_placements,
            "total_transitions": len(trajectory.ops),
            # Serialised so a dataset can be audited per query type without
            # inferring the question from the family name: four families build
            # with query_type="count" but the field was absent, so a downstream
            # reader could not tell a count question from a location question
            # (SPEC §6 rule 5, checklist 5).
            "query_type": spec.query_type,
        },

        "canonical_trace": canonical_trace,
        "sentences": sentences,
        "context": " ".join(sentences),
        "question": question,
        "query_entity": trajectory.target_obj,
        "gold_container": target_container,
        "gold_answer": gold_answer,
        "step_wise_gold": step_wise_gold,
        "step_wise_gold_answers": step_wise_gold_answers,

        "final_state": {
            "location": final_state.location,
            "containers": sorted(final_state.containers),
            "container_names": names.container_names,
            "container_display_names": names.container_names,
        },
    }

    # Consistency gate: the schema is declared once, so a field added here but
    # not to INSTANCE_RECORD_KEYS (or the reverse) fails loudly instead of
    # reintroducing two divergent key sets. raise AssertionError, not assert, so
    # it survives python -O (AGENTS.md §14).
    if tuple(record) != INSTANCE_RECORD_KEYS:
        raise AssertionError(
            "instance record schema drifted from INSTANCE_RECORD_KEYS: "
            f"record={tuple(record)} declared={INSTANCE_RECORD_KEYS}"
        )

    return InstanceResult(record=record, measured=measured)


# ============================================================
# Bounded retry with reported reasons
# ============================================================

def attempt_seed(seed: int, attempt: int) -> int:
    """
    Sub-seed for one attempt.

    Attempt 0 uses the caller's seed verbatim; every later attempt derives its
    own stream from ``seed|attempt_<n>``, so a retry is a fresh trajectory
    rather than a rerun of the same draw.
    """
    if attempt == 0:
        return seed
    return int(
        hashlib.sha1(f"{seed}|attempt_{attempt}".encode("utf-8")).hexdigest()[:8],
        16,
    )


def _format_histogram(reasons: Mapping[str, int]) -> str:
    return ", ".join(
        f"{reason}={count}"
        for reason, count in sorted(reasons.items())
    )


def generate_instance_with_retry(
    seed: int,
    spec: TrajectorySpec,
    *,
    instance_id: str = "",
    experiment: str = "",
    condition_id: str = "",
    textual_distractor_count: int = 0,
    min_v: Optional[int] = None,
    intended_v: Optional[int] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Run the gate up to ``max_attempts`` times and return the accepted record.

    Every rejection is counted by reason instead of being discarded, so an
    exhausted budget reports *why* the condition is unreachable rather than
    returning ``None`` and leaving the caller to guess.

    ``rng`` seeds attempt 0 verbatim when supplied; retries always fall back to
    the deterministic sub-seed schedule, so a retry never depends on how much of
    an injected stream the previous attempt happened to consume.
    """
    reasons: Counter[str] = Counter()
    samples: List[InstanceGateFailure] = []

    for attempt in range(max_attempts):
        sub_seed = attempt_seed(seed, attempt)
        traj_rng = (
            rng if (attempt == 0 and rng is not None) else random.Random(sub_seed)
        )

        result = build_validated_instance(
            traj_rng,
            spec,
            random.Random(sub_seed * 2 + 1),
            textual_distractor_count,
            instance_id=instance_id,
            experiment=experiment,
            condition_id=condition_id,
            seed=seed,
            attempt=attempt,
            min_v=min_v,
            intended_v=intended_v,
        )

        if result.ok and result.record is not None:
            return result.record

        failure = result.failure
        reasons[failure.reason] += 1
        if len(samples) < FAILURE_REPORT_LIMIT:
            samples.append(failure)

    raise GenerationError(
        f"{instance_id or spec.family}: generation gate rejected all "
        f"{max_attempts} attempts (family={spec.family!r}, seed={seed}); "
        f"reason histogram: {_format_histogram(reasons)}; "
        f"first {len(samples)} failure(s): "
        + " | ".join(sample.describe() for sample in samples)
    )
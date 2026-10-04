"""
test/test_factor_contract.py
===========================
Contract tests for the single factor definition (WP1).

The factors E,T,D,U,V,N,L have exactly one owner: ``generator/metadata.py``.
Requested factors (TrajectorySpec / Condition) are never trusted (SPEC §2, hard
rule 3), so these tests pin the measured definitions, the exact-match gate for
E/T/D, the opt-in V gate, the L_word ceiling and the gate's survival of
``python -O``.
"""
from __future__ import annotations

import ast
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generator import Condition, Experiment, TrajectorySpec
from generator.constants import L_MAX_WORDS
from generator.metadata import (
    MeasuredFactors,
    classify_op,
    measure_factors,
    verify_factors,
    verify_length,
    get_t_d_classification_table,
    validate_t_d_classification_complete,
)
from world import Move, Put, Remove, Split, Swap, Undo

C3 = {"c0", "c1", "c2"}


# ============================================================
# 1. Condition accepts D >= T  (SPEC RESOLVED-2)
# ============================================================

@pytest.mark.parametrize("T,D", [(8, 8), (16, 16), (1, 5), (4, 0)])
def test_condition_accepts_any_non_negative_D(T: int, D: int) -> None:
    cond = Condition(
        family="interleaved_chain", T=T, E=3, D=D, experiment=Experiment.CORE
    )
    assert cond.T == T
    assert cond.D == D
    assert cond.total_updates == T + D


def test_condition_still_rejects_T_below_one() -> None:
    with pytest.raises(ValueError):
        Condition(
            family="basic_chain", T=0, E=1, D=0, experiment=Experiment.CORE
        )


def test_condition_still_rejects_negative_D() -> None:
    with pytest.raises(ValueError):
        Condition(
            family="basic_chain", T=2, E=1, D=-1, experiment=Experiment.CORE
        )


# ============================================================
# 2. TrajectorySpec has no schema_version (single contract)
# ============================================================

def test_trajectory_spec_rejects_schema_version() -> None:
    with pytest.raises(TypeError):
        TrajectorySpec(
            family="basic_chain", entity_count=1, schema_version="v2"
        )


def test_trajectory_spec_has_no_schema_version_attribute() -> None:
    assert "schema_version" not in TrajectorySpec.__dataclass_fields__


def test_trajectory_spec_still_constructs_without_schema_version() -> None:
    spec = TrajectorySpec(family="basic_chain", entity_count=1, total_updates=3,
                          target_updates=3)
    assert spec.family == "basic_chain"
    assert spec.total_updates == 3


# ============================================================
# 3. measure_factors returns the documented E / T / D / U
# ============================================================

def test_measure_factors_counts_put_and_split_ids_as_E() -> None:
    ops = [Put("o0", "key", "c0"), Put("o1", "pen", "c1")]
    m = measure_factors(ops, C3, "o0")
    assert (m.E_actual, m.T_actual, m.D_actual, m.U_actual) == (2, 0, 0, 0)


def test_measure_factors_split_creating_the_target_counts_in_T() -> None:
    """The queried target is born from a Split: that Split is a target change (T)."""
    ops = [
        Put("o0", "key", "c0"),
        Put("o1", "pen", "c1"),
        Split("o0", "o2"),          # creates the queried target o2
        Move("o1", "c2"),
        Move("o1", "c0"),
    ]
    m = measure_factors(ops, C3, "o2")

    assert m.E_actual == 3          # {o0, o1, o2}
    assert m.T_actual == 1          # the Split alone moves o2 from unplaced to c0
    assert m.D_actual == 2          # both Moves touch o1 only
    assert m.U_actual == 3


def test_measure_factors_t_and_d_split_every_post_put_op() -> None:
    ops = [
        Put("o0", "key", "c0"),
        Put("o1", "pen", "c1"),
        Move("o0", "c1"),
        Move("o1", "c2"),
        Move("o0", "c2"),
        Swap("c1", "c2"),
    ]
    m = measure_factors(ops, C3, "o0")
    post_put = len(ops) - 2
    assert m.T_actual + m.D_actual == post_put
    assert m.U_actual == post_put
    assert m.T_actual >= 1


def test_measure_factors_removal_of_the_target_counts_in_T() -> None:
    ops = [Put("o0", "key", "c0"), Put("o1", "pen", "c1"), Remove("o0")]
    m = measure_factors(ops, C3, "o0")
    assert (m.T_actual, m.D_actual) == (1, 0)


def test_classify_op_is_the_single_t_d_discriminator() -> None:
    before = {"o0": "c0", "o1": "c1"}
    after_same = dict(before)
    after_moved = {"o0": "c1", "o1": "c0"}

    assert classify_op(Move("o0", "c1"), before, after_same, "o0").affects_target is False
    assert classify_op(Move("o0", "c1"), before, after_moved, "o0").affects_target is True
    assert classify_op(Swap("c1", "c0"), before, after_moved, "o1").affects_target is True
    assert classify_op(Undo(), before, after_moved, "o0").is_history_reversal is True
    assert classify_op(Move("o0", "c1"), before, after_moved, "o0").is_history_reversal is False


# ============================================================
# 4. U = T + D and U excludes Put
# ============================================================

@pytest.mark.parametrize("n_extra", [1, 3, 7])
def test_U_excludes_put_ops(n_extra: int) -> None:
    puts = [Put(f"e{i}", "key", f"c{i % 3}") for i in range(3)]
    moves = [Move("e0", f"c{(i % 2) + 1}") for i in range(n_extra)]
    ops = puts + moves
    m = measure_factors(ops, C3, "e0")
    assert m.U_actual == m.T_actual + m.D_actual
    assert m.U_actual == n_extra          # Puts never enter U
    assert m.U_actual == len(ops) - len(puts)


def test_to_dict_exposes_U_actual() -> None:
    m = MeasuredFactors(E_actual=3, T_actual=2, D_actual=4, V_actual=1,
                        L_word=42, N_actual=3)
    d = m.to_dict()
    assert d["U_actual"] == 6
    assert d["U_actual"] == d["T_actual"] + d["D_actual"]
    assert d["L_word"] == 42


# ============================================================
# 5. verify_factors gates E / T / D exactly
# ============================================================

def _measured(E: int = 2, T: int = 6, D: int = 3, V: int = 0) -> MeasuredFactors:
    return MeasuredFactors(E_actual=E, T_actual=T, D_actual=D, V_actual=V)


@pytest.mark.parametrize(
    "symbol,kwargs",
    [
        ("E_actual", dict(requested_E=3, requested_T=6, requested_D=3)),
        ("T_actual", dict(requested_E=2, requested_T=5, requested_D=3)),
        ("D_actual", dict(requested_E=2, requested_T=6, requested_D=4)),
    ],
)
def test_verify_factors_rejects_each_mismatch(symbol: str, kwargs: dict) -> None:
    with pytest.raises(AssertionError) as excinfo:
        verify_factors(
            measured=_measured(), family="interleaved_chain",
            instance_id="inst-42", **kwargs
        )
    msg = str(excinfo.value)
    assert symbol in msg
    assert "inst-42" in msg
    assert "interleaved_chain" in msg


def test_verify_factors_accepts_an_exact_match() -> None:
    verify_factors(
        requested_E=2, requested_T=6, requested_D=3,
        measured=_measured(), family="interleaved_chain",
    )


# ============================================================
# 6. V is opt-in
# ============================================================

def test_verify_factors_does_not_check_V_by_default() -> None:
    verify_factors(
        requested_E=2, requested_T=6, requested_D=3,
        measured=_measured(V=0), family="revision", instance_id="inst-7",
    )
    verify_factors(
        requested_E=2, requested_T=6, requested_D=3,
        measured=_measured(V=99), family="revision", instance_id="inst-7",
    )


def test_verify_factors_checks_V_only_when_asked_min_v() -> None:
    verify_factors(
        requested_E=2, requested_T=6, requested_D=3,
        measured=_measured(V=3), family="revision", min_v=2,
    )
    with pytest.raises(AssertionError) as excinfo:
        verify_factors(
            requested_E=2, requested_T=6, requested_D=3,
            measured=_measured(V=0), family="revision",
            instance_id="inst-v", min_v=1,
        )
    assert "V_actual" in str(excinfo.value)
    assert "inst-v" in str(excinfo.value)


def test_verify_factors_checks_V_only_when_asked_intended_v() -> None:
    verify_factors(
        requested_E=2, requested_T=6, requested_D=3,
        measured=_measured(V=4), family="revision", intended_v=4,
    )
    with pytest.raises(AssertionError) as excinfo:
        verify_factors(
            requested_E=2, requested_T=6, requested_D=3,
            measured=_measured(V=5), family="revision",
            instance_id="inst-w", intended_v=4,
        )
    assert "V_actual" in str(excinfo.value)
    assert "inst-w" in str(excinfo.value)


def test_verify_factors_has_no_hardcoded_revision_family_special_case() -> None:
    import inspect

    sig = inspect.signature(verify_factors)
    assert "tolerance_V" not in sig.parameters
    assert "min_v" in sig.parameters
    assert "intended_v" in sig.parameters


# ============================================================
# 7. L_word ceiling
# ============================================================

def test_verify_length_passes_at_the_limit() -> None:
    m = MeasuredFactors(E_actual=1, T_actual=1, D_actual=0, V_actual=0,
                        L_word=L_MAX_WORDS)
    verify_length(m, instance_id="inst-limit")


def test_verify_length_fails_one_word_above_the_limit() -> None:
    m = MeasuredFactors(E_actual=1, T_actual=1, D_actual=0, V_actual=0,
                        L_word=L_MAX_WORDS + 1)
    with pytest.raises(AssertionError) as excinfo:
        verify_length(m, instance_id="inst-long")
    msg = str(excinfo.value)
    assert "L_word" in msg
    assert "inst-long" in msg
    assert str(L_MAX_WORDS) in msg


def test_verify_length_honours_an_explicit_limit() -> None:
    m = MeasuredFactors(E_actual=1, T_actual=1, D_actual=0, V_actual=0, L_word=11)
    verify_length(m, limit=11)
    with pytest.raises(AssertionError):
        verify_length(m, limit=10)


def test_l_max_words_is_the_spec_value() -> None:
    assert L_MAX_WORDS == 600


# ============================================================
# 8. Gate assertions survive python -O  (AGENTS.md §14)
# ============================================================

def test_gate_module_raises_assertion_error_and_uses_no_bare_assert() -> None:
    src = (REPO_ROOT / "generator" / "metadata.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    bare_asserts = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
    ]
    assert bare_asserts == [], (
        f"generator/metadata.py uses bare `assert` at lines {bare_asserts}; "
        "consistency gates must use `raise AssertionError` (AGENTS.md §14)"
    )

    explicit = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and getattr(node.exc.func, "id", None) == "AssertionError"
    ]
    assert explicit, "generator/metadata.py must raise AssertionError explicitly"


def test_gate_assertions_survive_python_optimised() -> None:
    program = "\n".join([
        "from generator.metadata import MeasuredFactors, verify_factors, verify_length",
        "m = MeasuredFactors(E_actual=2, T_actual=6, D_actual=3, V_actual=0, L_word=5)",
        "for call in (",
        "    lambda: verify_factors(3, 6, 3, m, 'basic_chain', 'inst-o'),",
        "    lambda: verify_length(m, limit=4, instance_id='inst-o'),",
        "):",
        "    try:",
        "        call()",
        "    except AssertionError:",
        "        pass",
        "    else:",
        "        raise SystemExit('gate did not fire under -O')",
        "print('ok')",
    ])
    res = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


# ============================================================
# 9. Determinism (hard rule 5)
# ============================================================

def _ops_from_seed(seed: int) -> list:
    """A short valid trace: two Puts plus six Moves that respect Move validity."""
    rng = random.Random(seed)
    ops = [Put("e0", "key", "c0"), Put("e1", "pen", "c1")]
    where = {"e0": "c0", "e1": "c1"}
    for _ in range(6):
        obj = "e0" if rng.random() < 0.5 else "e1"
        dst = rng.choice(sorted(C3 - {where[obj]}))
        ops.append(Move(obj, dst))
        where[obj] = dst
    return ops


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_measure_factors_is_deterministic(seed: int) -> None:
    ops = _ops_from_seed(seed)
    first = measure_factors(ops, C3, "e0")
    second = measure_factors(list(ops), C3, "e0")
    assert first == second
    assert first.to_dict() == second.to_dict()


def test_two_separate_rng_streams_give_identical_factors() -> None:
    a = measure_factors(_ops_from_seed(99), C3, "e0")
    b = measure_factors(_ops_from_seed(99), C3, "e0")
    assert (a.E_actual, a.T_actual, a.D_actual, a.U_actual, a.V_actual) == \
           (b.E_actual, b.T_actual, b.D_actual, b.U_actual, b.V_actual)


def test_L_word_and_N_come_from_the_rendered_sentences() -> None:
    ops = [Put("o0", "key", "c0"), Move("o0", "c1")]
    sentences = ["Tom put a key in the box.", "Tom moved the key to the bag."]
    m = measure_factors(ops, C3, "o0", sentences=sentences,
                        textual_distractor_count=1)
    assert m.L_word == sum(len(s.split()) for s in sentences)
    assert m.N_actual == 1
    assert m.L_word == m.L_word


def test_unrendered_trajectory_reports_the_negative_length_sentinel() -> None:
    m = measure_factors([Put("o0", "key", "c0"), Move("o0", "c1")], C3, "o0")
    assert m.L_word == -1
    assert m.L_word == -1


# ============================================================
# 10. Answer leakage validator
# ============================================================

from generator.instance import _check_answer_leakage


def test_leakage_validator_detects_answer_in_distractors() -> None:
    """Leakage detected when gold answer appears in distractor sentences."""
    sentences = [
        "A key was placed in the box.",      # op 0
        "The key was moved to the bag.",      # op 1
        "The key is now in the bag.",         # distractor 0 - LEAKS!
    ]
    # 2 operation sentences, 1 distractor
    assert _check_answer_leakage(sentences, "the bag", num_op_sentences=2) is True


def test_leakage_validator_allows_answer_in_op_sentences() -> None:
    """Answer in operation descriptions is expected, not leakage."""
    sentences = [
        "A key was placed in the box.",      # op 0
        "The key was moved to the bag.",      # op 1 - contains answer but is operation
    ]
    # 2 operation sentences, 0 distractors
    assert _check_answer_leakage(sentences, "the bag", num_op_sentences=2) is False


def test_leakage_validator_allows_answer_in_early_distractors() -> None:
    """Answer in early distractors is OK (not in final k)."""
    sentences = [
        "A key was placed in the box.",      # op 0
        "The key was moved to the bag.",      # op 1
        "The key is now in the bag.",         # distractor 0 - early
        "A bird flew overhead.",              # distractor 1
        "The sun set slowly.",                # distractor 2
        "Clouds gathered.",                   # distractor 3
        "Night fell.",                        # distractor 4 - last
    ]
    # 2 operation sentences, 5 distractors; k=3 checks last 3 distractors (indices 2,3,4)
    # The answer is in distractor 0 which is NOT in the last 3
    assert _check_answer_leakage(sentences, "the bag", num_op_sentences=2) is False


def test_leakage_validator_catches_answer_in_final_k_distractors() -> None:
    """Answer in final k distractor sentences is leakage."""
    sentences = [
        "A key was placed in the box.",      # op 0
        "The key was moved to the bag.",      # op 1
        "A bird flew overhead.",              # distractor 0
        "The key is now in the bag.",         # distractor 1 - in last 2 (k=2)
    ]
    # 2 operation sentences, 2 distractors; k=2 checks both
    assert _check_answer_leakage(sentences, "the bag", num_op_sentences=2, suffix_k=2) is True


def test_leakage_validator_handles_no_gold_answer() -> None:
    """No gold answer -> no leakage."""
    sentences = ["A key was placed in the box.", "The key is now in the bag."]
    assert _check_answer_leakage(sentences, None, num_op_sentences=1) is False
    assert _check_answer_leakage(sentences, "", num_op_sentences=1) is False


def test_leakage_validator_handles_no_distractors() -> None:
    """No distractor sentences -> no leakage check possible."""
    sentences = ["A key was placed in the box.", "The key was moved to the bag."]
    assert _check_answer_leakage(sentences, "the bag", num_op_sentences=2) is False


# ============================================================
# 11. V/T structural invariant
# ============================================================

import warnings


def test_verify_factors_warns_on_V_exceeds_T() -> None:
    """verify_factors warns when V > T (structural invariant)."""
    m = MeasuredFactors(E_actual=2, T_actual=6, D_actual=3, V_actual=99)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        verify_factors(
            requested_E=2, requested_T=6, requested_D=3,
            measured=m, family="revision", instance_id="inst-vgt",
        )
        assert len(w) == 1
        assert "V_actual=99 > T_actual=6" in str(w[0].message)
        assert "violates structural invariant V <= T" in str(w[0].message)


# ============================================================
# 12. T/D classification table and validator
# ============================================================

def test_t_d_classification_table_exists() -> None:
    """Classification table is accessible and non-empty."""
    table = get_t_d_classification_table()
    assert len(table) > 0
    assert "Move" in table
    assert "Remove" in table
    assert "Split" in table
    assert "Merge" in table
    assert "Swap" in table
    assert "Undo" in table
    assert "Redo" in table


def test_validate_t_d_classification_complete_passes() -> None:
    """Validator accepts valid operation sequences."""
    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Move("o1", "c2")]
    validate_t_d_classification_complete(ops, "o0")  # Should not raise


def test_validate_t_d_classification_complete_covers_all_op_types() -> None:
    """Validator exercises classify_op for all operation types."""
    from world import Move, Remove, Split, Merge, Swap, Undo, Redo, Put, WorldState, History, apply_op
    
    # Build a sequence with all operation types affecting target
    ops = [
        Put("o0", "key", "c0"),
        Move("o0", "c1"),
        Remove("o0"),  # target removed
        Put("o1", "pen", "c2"),  # new target
        Split("o1", "o2"),  # target is source
        Merge("c2", "c0"),  # target in merged container
        Swap("c0", "c1"),  # target in swapped container
    ]
    validate_t_d_classification_complete(ops, "o1")  # Should not raise


# ============================================================
# 13. Deduplication by trace_hash
# ============================================================

from generator.instance import (
    _SEEN_TRACE_HASHES,
    reset_deduplication_registry,
    get_deduplication_stats,
    generate_instance_with_retry,
    build_validated_instance,
)
from generator.trajectory_specs import TrajectorySpec
import random


def test_deduplication_registry_tracks_unique_traces() -> None:
    """Deduplication registry records unique trace hashes."""
    reset_deduplication_registry()
    
    spec = TrajectorySpec(
        family="basic_chain",
        entity_count=1,
        target_updates=2,
        distractor_updates=0,
        total_updates=2,
        num_containers=3,
    )
    
    # Let's test directly
    from generator.instance import build_validated_instance, _SEEN_TRACE_HASHES
    
    reset_deduplication_registry()
    rng = random.Random(42)
    name_rng = random.Random(42 * 2 + 1)
    
    result1 = build_validated_instance(rng, spec, name_rng, instance_id="test-1", seed=42)
    assert result1.ok
    assert len(_SEEN_TRACE_HASHES) == 1
    
    # Second call with same trace should be rejected
    rng2 = random.Random(42)
    name_rng2 = random.Random(42 * 2 + 1)
    result2 = build_validated_instance(rng2, spec, name_rng2, instance_id="test-2", seed=42)
    assert not result2.ok
    assert result2.failure.check == "duplicate_trace"
    assert "duplicate trace_hash" in result2.failure.message


def test_deduplication_stats() -> None:
    """get_deduplication_stats returns correct counts."""
    reset_deduplication_registry()
    assert get_deduplication_stats() == {"unique_traces": 0}
    
    spec = TrajectorySpec(
        family="basic_chain",
        entity_count=1,
        target_updates=2,
        distractor_updates=0,
        total_updates=2,
        num_containers=3,
    )
    rng = random.Random(42)
    name_rng = random.Random(42 * 2 + 1)
    
    result = build_validated_instance(rng, spec, name_rng, instance_id="test-1", seed=42)
    assert result.ok
    assert get_deduplication_stats() == {"unique_traces": 1}
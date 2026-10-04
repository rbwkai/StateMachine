"""Initial-audit findings, encoded as executable tests.

Convention (benchmark-qa skill):
  * xfail(strict=True)  -> a verified defect or spec disagreement. The test states
    the CORRECT behaviour. When the defect is fixed the test XPASSes, strict mode
    turns that into a failure, and you delete the marker. Nothing silently rots.
  * plain test          -> characterization of current behaviour pending a SPEC decision.
Finding ids (F1..F8) match reports/initial-audit.md.
"""
import random

import pytest

from world import Merge, Move, Put, Redo, Swap, Undo, History
from analysis import analyze_trajectory
from generator import (
    CountQuery, LocationQuery, Condition, Experiment, TrajectorySpec,
    build_trajectory, measure_factors, sample_sequence,
)

C3 = {"c0", "c1", "c2"}


def test_f1_object_type_vocabularies_agree():
    from generator.trajectories import OBJECT_TYPES as gen_types
    from render.names import OBJECT_TYPES as render_types
    assert sorted(gen_types) == sorted(render_types)


def test_f2_count_query_analysis_survives_move():
    ops = [Put("o0", "key", "c0"), Move("o0", "c1")]
    analyze_trajectory(ops, C3, CountQuery("c1", "key"))


def test_f3_swap_that_moves_target_is_relevant():
    ops = [Put("o0", "key", "c0"), Put("o1", "pen", "c1"), Swap("c0", "c1")]
    a = analyze_trajectory(ops, C3, LocationQuery("o0"))
    assert 2 in a.relevant_steps          # the Swap changes o0's location


def test_f4_count_question_pluralises_correctly():
    from render import NameRegistry, question_count
    names = NameRegistry(random.Random(0), C3)
    assert "watchs" not in question_count("c0", "watch", names)


def test_f5_condition_and_spec_agree_on_distractor_ceiling():
    """F5 is FIXED, so this is now a plain contract test rather than an xfail.

    SPEC RESOLVED-2: D >= T is valid and only T >= 1 is required. Condition no
    longer rejects a distractor-heavy request, so Condition and TrajectorySpec
    accept the same requests. Keep this unmarked: if the D >= T ceiling ever
    comes back, this test fails rather than silently rotting."""
    spec = TrajectorySpec(family="interleaved_chain", entity_count=3, total_updates=6,
                          target_updates=2, distractor_updates=4)
    build_trajectory(random.Random(0), spec)                       # builds fine
    Condition(family="interleaved_chain", T=2, E=3, D=4,
              experiment=Experiment.CORE)                          # must too


def test_f6_single_definition_of_question_helpers():
    """Question templates are owned by render/narrative.py alone (AGENTS.md §5).

    render/narrative.py previously re-exported templates defined in render/names.py,
    and question_redo_validity was defined in BOTH with byte-identical output. The
    assertion is now stronger than the old identity check: the helpers must exist
    in the owner and must NOT exist in render/names.py at all.
    """
    import render
    from render import names, narrative
    for fn in ("question_location", "question_count",
               "question_counterfactual", "question_redo_validity"):
        assert hasattr(narrative, fn), f"{fn} missing from render.narrative"
        assert not hasattr(names, fn), f"{fn} is duplicated in render.names"
    assert render.question_location is narrative.question_location
    assert render.question_redo_validity is narrative.question_redo_validity


def test_f7_sampler_update_count_excludes_puts():
    ops, *_ = sample_sequence(random.Random(1), entity_count=4, update_count=6,
                              operations_enabled=[Put, Move, Swap])
    assert sum(not isinstance(o, Put) for o in ops) == 6


def test_f8_characterize_V_counts_undo_twice():
    """OPEN-4: one target-affecting Undo contributes to BOTH the revisit count and
    history_reversal_count. Pins current behaviour; replace with the intended value
    once SPEC.md defines V unambiguously."""
    ops = [Put("o0", "key", "c0"), Move("o0", "c1"), Move("o0", "c2"), Undo()]
    m = measure_factors(ops, C3, "o0")
    assert (m.T_actual, m.V_actual) == (3, 2)


@pytest.mark.parametrize("family,kw", [
    ("basic_chain", dict(entity_count=1, total_updates=4, target_updates=4)),
    ("interleaved_chain", dict(entity_count=3, total_updates=6, target_updates=3, distractor_updates=3)),
    ("revision", dict(entity_count=1, total_updates=6, target_updates=6)),
])
def test_gate_builds_and_measures_match_request(family, kw):
    """Regression guard for the validation gate: measured == requested over a seed sweep."""
    for seed in range(50):
        r = build_trajectory(random.Random(seed), TrajectorySpec(family=family, **kw))
        m = r.measured_factors
        assert m.T_actual == kw["target_updates"]
        assert m.D_actual == kw.get("distractor_updates", 0)


def test_f9_zero_count_is_a_valid_candidate():
    from eval.scoring import candidate_answers
    cands = candidate_answers({"gold_answer": 0, "step_wise_gold_answers": [1, 0, 2]})
    assert "0" in cands


def test_f10_revision_count_is_enforced():
    spec = TrajectorySpec(family="revision", entity_count=1, total_updates=3,
                          target_updates=3, revision_count=10)
    try:
        r = build_trajectory(random.Random(0), spec)
    except (ValueError, AssertionError):
        return                                    # correctly rejected an unreachable request
    assert r.measured_factors.V_actual >= spec.revision_count   # or the request must be honoured


def test_f11_render_has_no_duplicate_module():
    """There is exactly one renderer: render/narrative.py (AGENTS.md §5).

    render/templates.py was a byte-for-byte duplicate of render/narrative.py --
    all eight per-operation render functions plus render_narrative were defined in
    both -- and is deleted (D-015). This used to `return` early on
    ModuleNotFoundError, so it passed without checking anything; it now asserts the
    duplicate is absent and that the surviving renderer is the real one.
    """
    import importlib
    import pytest as _pytest
    with _pytest.raises(ModuleNotFoundError):
        importlib.import_module("render.templates")
    from render import narrative
    assert callable(narrative.render_narrative)

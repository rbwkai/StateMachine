"""
test/test_robustness_wiring.py
==============================
Regression tests for the robustness wiring fixes:

- experiments/rq3_scale_reasoning.py reads the keys the harness actually emits
  (``overall_accuracy``, ``error_analysis.error_type``).
- eval/robustness.py: no duplicate paraphrase rules, hyphenated names are
  paraphrased, every rendered op sentence is rewritten, and the minimal prompt
  variant follows the record's query type.
- run_eval.py: ``--robustness`` is opt-in, leaves the metrics schema
  unchanged, and writes ``<dataset>.robustness.json`` with a safe name.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.eval_harness import evaluate_predictions  # noqa: E402
from eval.prompts import QUERY_TYPES  # noqa: E402
from eval.robustness import (  # noqa: E402
    MINIMAL_INSTRUCTIONS,
    PARAPHRASE_RULES,
    apply_paraphrase,
    build_prompt_variants,
    evaluate_prompt_sensitivity,
)
from test.conftest import build_record, quiet  # noqa: E402

RQ3_PATH = _REPO_ROOT / "experiments" / "rq3_scale_reasoning.py"

# One cell per released family so every op sentence shape is rendered.
FAMILY_CELLS = [
    ("basic_chain", 1, 4, 0),
    ("interleaved_chain", 2, 4, 2),
    ("revision", 1, 4, 0),
    ("split_chain", 2, 4, 0),
    ("merge_chain", 2, 4, 0),
    ("swap_chain", 2, 4, 0),
    ("undo_chain", 1, 4, 0),
    ("undo_redo_chain", 1, 8, 0),
]

# Verb phrases of render/narrative.py op sentences; none may survive paraphrase.
_ORIGINAL_VERBS = re.compile(
    r"was placed in|was moved|was taken out of|were swapped|was undone|was redone"
)


@pytest.fixture(scope="module")
def family_records():
    records = []
    with quiet():
        for family, e, t, d in FAMILY_CELLS:
            for seed in range(7):
                rec = build_record(
                    family, seed, entity_count=e, target_updates=t,
                    distractor_updates=d, max_attempts=20,
                )
                if rec is not None:
                    records.append(rec)
    return records


# ============================================================
# rq3_scale_reasoning.py
# ============================================================

def _rq3_module():
    import importlib
    return importlib.import_module("experiments.rq3_scale_reasoning")


def test_rq3_saved_accuracy_key_exists_in_harness_output() -> None:
    """The serialised 'accuracy' must read a key evaluate_predictions returns."""
    scored = evaluate_predictions(
        [{"instance_id": "a", "gold_answer": "the red box", "family": "basic_chain"}],
        [{"instance_id": "a", "pred_answer": "the red box"}],
    )
    tree = ast.parse(RQ3_PATH.read_text(encoding="utf-8"))
    read_keys = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant) and key.value == "accuracy"
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "get"
                and value.args and isinstance(value.args[0], ast.Constant)
            ):
                read_keys.append(value.args[0].value)
    assert read_keys, "no 'accuracy': v.get(...) entry found in rq3"
    for k in read_keys:
        assert k in scored, f"rq3 saves v.get({k!r}) but the harness emits {sorted(scored)}"


def test_rq3_analysis_reads_nested_error_type() -> None:
    rq3 = _rq3_module()
    results = [{
        "model": "m",
        "prompt_mode": "direct",
        "conditions": {
            "cond": {
                "instance_results": [
                    {"is_correct": False, "error_analysis": {"error_type": "STALE_STATE"}},
                    {"is_correct": False, "error_analysis": {"error_type": "STALE_STATE"}},
                    {"is_correct": False, "error_analysis": None},
                    {"is_correct": True, "error_analysis": None},
                ],
            },
        },
    }]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), tempfile.TemporaryDirectory() as tmp:
        rq3.analyze_results(results, Path(tmp))
    out = buf.getvalue()
    assert "dom=STALE_STATE (2)" in out, out


# ============================================================
# eval/robustness.py
# ============================================================

def test_paraphrase_rules_have_no_duplicates() -> None:
    patterns = [p for p, _ in PARAPHRASE_RULES]
    dupes = [p for p, n in collections.Counter(patterns).items() if n > 1]
    assert not dupes, dupes


@pytest.mark.parametrize("sentence,expected", [
    ("The second key was moved from the red-bag to the box.",
     "The second key goes from the red-bag to the box."),
    ("A key was placed in the red-bag.", "A key is put into the red-bag."),
    ("The key was moved to the red-bag.", "The key is moved to the red-bag."),
    ("The x-ray was taken out of the old-box.", "The x-ray is removed from the old-box."),
    ("Everything in the red-bag was moved into the box.",
     "All items in the red-bag are transferred to the box."),
    ("The contents of the red-bag and the box were swapped.",
     "the red-bag and the box exchange their contents."),
])
def test_paraphrase_handles_hyphenated_names(sentence: str, expected: str) -> None:
    assert apply_paraphrase(sentence) == expected


def test_every_op_sentence_is_fully_paraphrased(family_records) -> None:
    """Every rendered op sentence is rewritten with no original verb left over."""
    assert len(family_records) >= 50
    total = collections.Counter()
    missed = collections.defaultdict(list)
    for record in family_records:
        ops = [s["op_type"] for s in record["canonical_trace"]]
        assert len(ops) == len(record["sentences"])
        for op, sentence in zip(ops, record["sentences"]):
            total[op] += 1
            out = apply_paraphrase(sentence)
            if out == sentence or _ORIGINAL_VERBS.search(out):
                missed[op].append(sentence)
    assert set(total) >= {"PUT", "MOVE", "SPLIT", "MERGE", "SWAP", "UNDO", "REDO"}
    assert not missed, {op: f"{len(v)}/{total[op]} e.g. {v[0]!r}" for op, v in missed.items()}


def test_minimal_instructions_cover_answer_slots() -> None:
    assert set(MINIMAL_INSTRUCTIONS) == set(QUERY_TYPES)


def test_minimal_variant_follows_query_type() -> None:
    loc = build_prompt_variants("ctx", "Where is the key?", query_type="location")
    cnt = build_prompt_variants("ctx", "How many keys are in the box?", query_type="count")
    assert MINIMAL_INSTRUCTIONS["location"] in loc["v3_minimal"]
    assert MINIMAL_INSTRUCTIONS["count"] in cnt["v3_minimal"]
    assert "Where is the object?" not in cnt["v3_minimal"]
    assert "<a single integer>" in cnt["v3_minimal"]
    with pytest.raises(ValueError):
        build_prompt_variants("ctx", "q", query_type="redo_validity")


def test_prompt_sensitivity_uses_record_query_type() -> None:
    record = {
        "instance_id": "c1",
        "gold_answer": "2",
        "spec": {"query_type": "count"},
        "context": "A key was placed in the box. A key was placed in the box.",
        "question": "How many keys are in the box?",
        "sentences": [],
    }
    seen = []

    def predict(prompt: str) -> str:
        seen.append(prompt)
        return "Final Answer: 2"

    evaluate_prompt_sensitivity(record, predict)
    minimal = [p for p in seen if MINIMAL_INSTRUCTIONS["count"] in p]
    assert len(minimal) == 1
    assert not any("Where is the object?" in p for p in seen)


# ============================================================
# run_eval.py --robustness
# ============================================================

def _run(records, tmp: Path, robustness: bool):
    from eval.models import CORE_MODELS
    from run_eval import run_evaluation

    cfg = next(iter(CORE_MODELS.values()))
    with quiet():
        metrics = run_evaluation(
            model_config=cfg,
            dataset_records=records,
            dataset_name="tiny",
            output_dir=tmp,
            max_new_tokens=16,
            mock=True,
            robustness=robustness,
        )
    return cfg, metrics


def test_robustness_flag_off_by_default_and_writes_nothing(family_records) -> None:
    import inspect
    from run_eval import run_evaluation

    assert inspect.signature(run_evaluation).parameters["robustness"].default is False
    with tempfile.TemporaryDirectory() as tmp:
        _run(family_records[:4], Path(tmp), robustness=False)
        assert not list(Path(tmp).rglob("*.robustness.json"))


def test_robustness_flag_writes_summary_and_keeps_metrics_schema(family_records) -> None:
    records = family_records[:4]
    with tempfile.TemporaryDirectory() as tmp_off, tempfile.TemporaryDirectory() as tmp_on:
        _, m_off = _run(records, Path(tmp_off), robustness=False)
        cfg, m_on = _run(records, Path(tmp_on), robustness=True)
        assert set(m_off) == set(m_on)

        written = list(Path(tmp_on).rglob("*.robustness.json"))
        assert len(written) == 1
        path = written[0]
        # L7: named after the dataset, not the metrics file.
        assert path.name == "tiny.robustness.json"
        assert path.parent == Path(tmp_on) / cfg.name / "no_cot_16"

        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["total_instances"] == len(records)
        assert set(summary["robustness"]) >= {"prompt_sensitivity", "paraphrase_robustness"}
        assert len(summary["robustness"]["per_instance_prompt"]) == len(records)
        assert summary["solubility"]["total_audited"] == len(records)


def test_robustness_path_is_sanitised() -> None:
    from run_eval import robustness_path_for

    base = Path("/out/model")
    assert robustness_path_for(base / "tiny_metrics.json") == base / "tiny_metrics.robustness.json"
    unsafe = robustness_path_for(base / "..secret name$.json")
    assert unsafe.parent == base
    assert re.fullmatch(r"[A-Za-z0-9._-]+", unsafe.name)
    assert not unsafe.name.startswith(".")


def test_cli_exposes_robustness_flag() -> None:
    import run_eval

    src = Path(run_eval.__file__).read_text(encoding="utf-8")
    assert '"--robustness"' in src
    assert "robustness=args.robustness" in src


def test_robustness_prompt_text_lives_in_prompts_module() -> None:
    """M5 (AGENTS.md §5): prompt strings are owned by eval/prompts.py and
    eval/robustness.py only re-exports them."""
    import eval.prompts as prompts
    import eval.robustness as robustness

    assert robustness.MINIMAL_INSTRUCTIONS is prompts.MINIMAL_INSTRUCTIONS
    assert robustness.PROMPT_TEMPLATES is prompts.PROMPT_TEMPLATES
    assert robustness.build_prompt_variants is prompts.build_prompt_variants
    tree = ast.parse(Path(robustness.__file__).read_text(encoding="utf-8"))
    assigned = {
        target.id
        for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    assert not assigned & {"MINIMAL_INSTRUCTIONS", "PROMPT_TEMPLATES"}

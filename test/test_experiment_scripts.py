"""Checklist 9: experiment scripts.

Runs the RQ scripts as subprocesses: `--dry-run`, the JSONL schema, the output
path, the `--family`/`--instances` filters, seed uniqueness, byte-identical
re-runs, and the RQ2 paired-cell design. Real defects are documented as plain
failing `test_failsnow_*` tests.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from generator.instance import INSTANCE_RECORD_KEYS


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["rq1_mutation_depth", "rq2_interference", "rq3_scale_reasoning"]
TIMEOUT = 900


def _run(script, *args, expect=None):
    result = subprocess.run(
        [sys.executable, "-m", f"experiments.{script}", *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=TIMEOUT,
    )
    if expect is not None:
        assert result.returncode == expect, (
            f"{script} {args} exited {result.returncode}\n{result.stdout[-2000:]}\n"
            f"{result.stderr[-2000:]}"
        )
    return result


@pytest.fixture(scope="module")
def rq1_dry_run():
    return _run("rq1_mutation_depth", "--dry-run")


@pytest.fixture(scope="module")
def rq2_dry_run():
    return _run("rq2_interference", "--dry-run")


@pytest.fixture(scope="module")
def rq2_output(tmp_path_factory):
    path = tmp_path_factory.mktemp("rq2") / "rq2.jsonl"
    _run("rq2_interference", "--instances", 2, "--output", path, expect=0)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Dry runs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["rq1_mutation_depth", "rq2_interference"])
def test_generation_scripts_accept_dry_run(script):
    result = _run(script, "--dry-run")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "DRY-RUN" in result.stdout


def test_dry_run_writes_nothing_to_disk(tmp_path):
    target = tmp_path / "should_not_exist.jsonl"
    for script in ("rq1_mutation_depth", "rq2_interference"):
        _run(script, "--dry-run", "--output", target, expect=0)
    assert not target.exists()


def test_failsnow_every_script_accepts_dry_run():
    """[checklist 9] '`--dry-run` passes for every script.'

    [fails now] expected: all three scripts take `--dry-run` and exit 0 without
    writing data. currently rq3_scale_reasoning has no such flag, so argparse
    aborts with exit code 2.
    """
    failures = {}
    for script in SCRIPTS:
        result = _run(script, "--dry-run")
        if result.returncode != 0:
            failures[script] = result.returncode
    assert not failures, failures


# ---------------------------------------------------------------------------
# RQ1 reachability probe versus the real gate.
# ---------------------------------------------------------------------------

def test_dry_run_probe_reports_the_cells_that_cannot_generate():
    """[checklist 9] Probe calls the full gate: unreachable cell reports FAIL.

    Premise changed: merge/swap at SPEC E=2 now generate, so the old
    unreachable expectation is gone. Force an unreachable cell (merge_chain
    E=1, builder requires >=2) and assert the probe reports FAIL while the
    SPEC-correct E=2 cell reports OK.
    """
    from experiments._common import probe_reachability

    bad = probe_reachability(
        family="merge_chain", entity_count=1, target_updates=8,
        distractor_updates=0, query_type="count",
    )
    assert not bad.reachable
    good = probe_reachability(
        family="merge_chain", entity_count=2, target_updates=8,
        distractor_updates=0, query_type="count",
    )
    assert good.reachable


def test_failsnow_rq1_full_run_writes_records(tmp_path):
    """[checklist 9] the RQ1 script must produce its JSONL.

    [fails now] expected: a small run writes records. currently the generation
    loop references an undefined `condition_id` and dies with
    `NameError: name 'condition_id' is not defined`, so RQ1 produces no data at
    all while its dry run still reports success.
    """
    output = tmp_path / "rq1.jsonl"
    result = _run("rq1_mutation_depth", "--family", "basic_chain",
                  "--instances", 1, "--output", output)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert output.exists()
    assert [line for line in output.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# RQ2 output, schema, filters, determinism.
# ---------------------------------------------------------------------------

def test_output_path_is_honoured_and_jsonl_schema_is_stable(rq2_output):
    expected_keys = set(INSTANCE_RECORD_KEYS) | {"experiment", "condition_id", "attempt"}
    assert rq2_output
    for record in rq2_output:
        assert set(record) == expected_keys, sorted(set(record) ^ expected_keys)


def test_instances_filter_sets_the_records_per_condition(rq2_output):
    counts = {}
    for record in rq2_output:
        counts[record["condition_id"]] = counts.get(record["condition_id"], 0) + 1
    assert counts
    assert set(counts.values()) == {2}


def test_seeds_are_unique_across_families_and_conditions(rq2_output):
    instance_ids = [record["instance_id"] for record in rq2_output]
    assert len(set(instance_ids)) == len(instance_ids)
    pairs = {(record["condition_id"], record["seed"]) for record in rq2_output}
    assert len(pairs) == len(rq2_output)
    families = {(record["condition_id"], record["family"]) for record in rq2_output}
    assert all(isinstance(family, str) and family for _condition, family in families)


def test_rerun_regenerates_a_byte_identical_file(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _run("rq2_interference", "--instances", 2, "--output", first, expect=0)
    _run("rq2_interference", "--instances", 2, "--output", second, expect=0)
    assert first.read_bytes() == second.read_bytes()


def test_family_filter_rejects_an_unknown_family():
    result = _run("rq1_mutation_depth", "--dry-run", "--family", "no_such_family")
    assert result.returncode == 1
    assert "Unknown family" in result.stdout


def test_family_filter_restricts_the_probed_conditions():
    result = _run("rq1_mutation_depth", "--dry-run", "--family", "revision", expect=0)
    probed = [line for line in result.stdout.splitlines() if "[OK]" in line]
    assert probed
    assert all("revision" in line for line in probed)


# ---------------------------------------------------------------------------
# RQ2 paired-cell design.
# ---------------------------------------------------------------------------

def _by_condition(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["condition_id"], []).append(record)
    return grouped


def test_rq2_cells_keep_the_same_container_count_and_design(rq2_output):
    grouped = _by_condition(rq2_output)
    state_cells = {c: rs for c, rs in grouped.items() if c != "revision_T8"}
    # U grows with D by design; the design invariant is the container count, the
    # initial placement count and the target depth.
    designs = {
        (len(record["final_state"]["containers"]),
         record["spec"]["initial_placements"],
         record["measured_factors"]["T_actual"])
        for records in state_cells.values()
        for record in records
    }
    assert len(designs) == 1, f"cells differ in design: {sorted(designs)}"


def test_rq2_cells_share_the_same_target_trajectory(rq2_output):
    """[checklist 9] N-cells at fixed D share the state trace (text-only diff).

    D-varying cells necessarily differ in ops; only D4 vs D4_N* pairs must
    share trace_hash (same seed_group, same D, N is text-only).
    """
    grouped = _by_condition(rq2_output)
    base = grouped.get("D4")
    assert base, f"no D4 cell in {sorted(grouped)}"
    base_hashes = {r["trace_hash"] for r in base}
    n_cells = {c: rs for c, rs in grouped.items() if c.startswith("D4_N")}
    assert n_cells, f"no D4_N* cells in {sorted(grouped)}"
    for cond, rs in n_cells.items():
        other = {r["trace_hash"] for r in rs}
        assert base_hashes & other, f"{cond} shares no trace with D4"


@pytest.mark.xfail(reason="RQ2 not length-matched by design (paper §5); length-control module per LENGTH_CONTROLS.md not built", strict=False)
def test_failsnow_rq2_cells_are_length_matched_within_three_words(rq2_output):
    """[checklist 9] 'length-matched within +-3 words.'

    xfail: D8/D16 differ in trace from D4 so ±3 vs D4 baseline can never hold;
    proper test pairs D vs N at matching length once the length-control module
    lands. Paper already states RQ2 is not length-matched.
    """
    grouped = _by_condition(rq2_output)
    lengths = {c: [len(r["context"].split()) for r in rs] for c, rs in grouped.items()}
    base = lengths.get("D4")
    assert base
    for condition, values in lengths.items():
        for value in values:
            assert abs(value - base[0]) <= 3, (
                f"{condition} is {value} words against the D4 baseline {base[0]}"
            )


# ---------------------------------------------------------------------------
# RQ3 wiring.
# ---------------------------------------------------------------------------

def _imported_names(module_name):
    tree = ast.parse((REPO_ROOT / "experiments" / f"{module_name}.py").read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("eval"):
            names.append((node.module, [alias.name for alias in node.names]))
    return names


@pytest.mark.parametrize("script", ["rq1_mutation_depth", "rq2_interference",
                                    "rq3_scale_reasoning"])
def test_every_eval_symbol_the_scripts_import_exists(script):
    """Guards the checklist's RQ3 finding about importing `format_prompt_v2`,
    which does not exist."""
    import importlib

    for module_name, names in _imported_names(script):
        module = importlib.import_module(module_name)
        missing = [name for name in names if not hasattr(module, name)]
        assert not missing, f"{script} imports missing {module_name}.{missing}"


def test_engine_generate_batch_calls_match_the_engine_signature():
    """The checklist's second RQ3 finding: `batch_size=` is not a parameter of
    `generate_batch`, so any call passing it raises TypeError."""
    import inspect

    from eval.engine import InferenceEngine, MockInferenceEngine

    parameters = set(inspect.signature(InferenceEngine.generate_batch).parameters)
    for script in SCRIPTS:
        source = (REPO_ROOT / "experiments" / f"{script}.py").read_text()
        assert "batch_size=" not in source, f"{script} passes batch_size="
        assert "format_prompt_v2" not in source, f"{script} imports format_prompt_v2"
    assert "max_new_tokens" in parameters
    assert MockInferenceEngine.generate_batch is not None


def test_failsnow_rq3_condition_ids_are_unique_per_family():
    """[checklist 9] 'RQ3 loads the right family per condition. [fails now]
    `condition_id="T8"` is shared across families.'

    [fails now] expected: every representative condition names a distinct
    (experiment_tag, condition_id) pair, so `load_dataset` cannot return the
    wrong family. currently seven RQ1 entries share `("rq1_mutation_depth",
    "T8")` while claiming seven different families.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from experiments.rq3_scale_reasoning import REPRESENTATIVE_CONDITIONS

    pairs = [(tag, condition) for tag, condition, _description in REPRESENTATIVE_CONDITIONS]
    duplicates = {pair for pair in pairs if pairs.count(pair) > 1}
    assert not duplicates, f"shared condition ids: {sorted(duplicates)}"


def test_failsnow_every_generation_script_exposes_the_shared_cli():
    """[checklist 9] 'Output paths, JSONL schema, `--family` and `--instances`
    filters.'"""
    failures = {}
    for script in SCRIPTS:
        result = _run(script, "--help")
        if result.returncode != 0:
            failures[script] = "help failed"
            continue
        flags = {token for token in ("--dry-run", "--output", "--instances")
                 if token not in result.stdout}
        if flags:
            failures[script] = sorted(flags)
    assert not failures, failures

"""
test/test_determinism_manifest.py
=================================
Determinism under PYTHONHASHSEED (AGENTS.md §6 rule 5) for the random sampler
and the redo-validity probe builder, plus the per-data-file manifest contract
(AGENTS.md §9: no tokens, home paths or hostnames).

The hash-seed checks run each case in a fresh interpreter, because set
iteration order is fixed per process and cannot be varied in-process.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HASH_SEEDS = ("0", "1", "2")

# Prints one canonical JSON document for a handful of seeds. Merge is enabled
# because its constructor used to iterate state.containers (a set) straight
# into rng.choice.
_SCRIPT = r"""
import dataclasses, json, random, sys
sys.path.insert(0, sys.argv[1])
from world import Merge, Move, Put, Redo, Remove, Split, Swap, Undo
from generator.sampler import sample_sequence
from generator.probes import build_redo_validity_examples

def op_view(op):
    return [type(op).__name__, dataclasses.asdict(op)]

def state_view(state):
    return {
        "object_type": list(state.object_type.items()),
        "location": list(state.location.items()),
        "containers": sorted(state.containers),
    }

ALL_OPS = [Put, Move, Remove, Split, Merge, Swap, Undo, Redo]
out = {"sample_sequence": [], "redo": []}
for seed in (0, 1, 7, 42, 1234):
    ops, state, history, containers = sample_sequence(
        random.Random(seed), entity_count=4, update_count=10,
        operations_enabled=ALL_OPS, num_containers=4,
    )
    out["sample_sequence"].append({
        "ops": [op_view(op) for op in ops],
        "state": state_view(state),
        "containers": sorted(containers),
    })
for seed in (3, 99):
    examples, balance = build_redo_validity_examples(
        random.Random(seed), 3, 6, [Put, Move, Merge, Swap, Undo, Redo], 3,
        n_per_class=2,
    )
    out["redo"].append({
        "balance": balance,
        "examples": [
            {"ops": [op_view(op) for op in ex.ops],
             "state": state_view(ex.state),
             "valid": ex.would_be_valid}
            for ex in examples
        ],
    })
sys.stdout.write(json.dumps(out, sort_keys=True))
"""


def _run(hash_seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(REPO_ROOT)],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def test_sampler_and_redo_examples_are_identical_across_pythonhashseed():
    outputs = {seed: _run(seed) for seed in HASH_SEEDS}
    assert outputs["0"], "subprocess produced no output"
    assert len(set(outputs.values())) == 1, (
        "output depends on PYTHONHASHSEED (set iteration order leaked)"
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _common():
    sys.path.insert(0, str(REPO_ROOT))
    import experiments._common as common

    return common


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_manifest_is_named_after_the_data_file(tmp_path):
    common = _common()
    data = tmp_path / "rq1_mutation_depth.jsonl"
    _write_jsonl(data, [{"a": 1}])
    path = common.write_manifest(data, {"experiment": "rq1"})
    assert path == tmp_path / "rq1_mutation_depth.manifest.json"
    assert path.exists()


def test_manifests_in_a_shared_directory_do_not_overwrite(tmp_path):
    common = _common()
    rq1 = tmp_path / "rq1.jsonl"
    rq2 = tmp_path / "rq2.jsonl"
    _write_jsonl(rq1, [{"a": 1}])
    _write_jsonl(rq2, [{"b": 2}, {"b": 3}])
    p1 = common.write_manifest(rq1, {"experiment": "rq1"})
    p2 = common.write_manifest(rq2, {"experiment": "rq2"})
    assert p1 != p2
    assert json.loads(p1.read_text())["experiment"] == "rq1"
    assert json.loads(p2.read_text())["experiment"] == "rq2"


def test_manifest_records_file_and_code_provenance(tmp_path):
    import hashlib

    from generator.constants import SPEC_VERSION

    common = _common()
    data = tmp_path / "d.jsonl"
    _write_jsonl(data, [{"x": 1}, {"x": 2}, {"x": 3}])
    manifest = json.loads(common.write_manifest(data, {"experiment": "t"}).read_text())

    assert manifest["data_file"]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert manifest["data_file"]["records"] == 3
    assert manifest["spec_version"] == SPEC_VERSION
    assert manifest["seed_scheme"] == common.GENERATION_SEED_SCHEME
    assert set(manifest["git"]) == {"commit", "dirty"}
    assert manifest["environment"]["python"]
    assert set(manifest["environment"]["packages"]) == {
        "numpy", "scipy", "torch", "transformers",
    }


def test_manifest_leaks_no_home_path_hostname_or_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETSECRETSECRET")
    common = _common()
    data = tmp_path / "leak.jsonl"
    _write_jsonl(data, [{"x": 1}])
    text = common.write_manifest(data, {"experiment": "t"}).read_text()

    assert json.loads(text)["data_file"]["name"] == "leak.jsonl"
    assert str(tmp_path) not in text
    assert str(Path.home()) not in text
    assert "hf_SECRET" not in text
    hostname = socket.gethostname()
    if len(hostname) >= 4:
        assert hostname not in text


@pytest.mark.parametrize("name", ["x.json", "x.txt"])
def test_manifest_record_count_is_none_for_non_jsonl(tmp_path, name):
    common = _common()
    data = tmp_path / name
    data.write_text("{}\n", encoding="utf-8")
    manifest = json.loads(common.write_manifest(data, {}).read_text())
    assert manifest["data_file"]["records"] is None

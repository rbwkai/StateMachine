"""Checklist 11: reproducibility and freeze.

Checks dataset hashes under different `PYTHONHASHSEED` values, cross-interpreter
stability, offline generation, the dataset manifest, and that
`test_eval_pipeline.py` is collectable. Real defects are documented as plain
failing `test_failsnow_*` tests; checks that need a second interpreter skip with
a reason instead of pretending to pass.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["0", "1", "12345"]
TIMEOUT = 900


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generate(script, output, pythonhashseed="0", env_extra=None):
    import os

    env = dict(os.environ, PYTHONHASHSEED=pythonhashseed)
    env.update(env_extra or {})
    result = subprocess.run(
        [sys.executable, "-m", f"experiments.{script}", "--instances", "1",
         "--output", str(output)],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert Path(output).exists()
    return _digest(output)


def _other_interpreters():
    found = []
    for candidate in ("python3.12", "python3.13", "python3.11", "python3.10"):
        path = shutil.which(candidate)
        if path:
            found.append(path)
    return found


# ---------------------------------------------------------------------------
# Hash stability.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["rq2_interference"])
def test_dataset_hash_is_identical_across_pythonhashseed_values(script, tmp_path):
    """Set-iteration order must not leak into generated content (SPEC §6 rule 5)."""
    digests = {
        seed: _generate(script, tmp_path / f"{seed}.jsonl", pythonhashseed=seed)
        for seed in SEEDS
    }
    assert len(set(digests.values())) == 1, digests


def test_dataset_hash_is_identical_across_interpreters(tmp_path):
    others = _other_interpreters()
    if not others:
        pytest.skip(
            "only one interpreter available; cross-version hashes cannot be compared here"
        )
    here = _generate("rq2_interference", tmp_path / "here.jsonl")
    import os

    env = dict(os.environ, PYTHONHASHSEED="0")
    other = others[0]
    result = subprocess.run(
        [other, "-m", "experiments.rq2_interference", "--instances", "1",
         "--output", str(tmp_path / "other.jsonl")],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert _digest(tmp_path / "other.jsonl") == here


# ---------------------------------------------------------------------------
# Offline generation.
# ---------------------------------------------------------------------------

def test_generation_runs_with_the_network_disabled(tmp_path):
    wrapper = tmp_path / "offline_generate.py"
    output = tmp_path / "offline.jsonl"
    wrapper.write_text(
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import socket\n"
        "def _blocked(*args, **kwargs):\n"
        "    raise OSError('network access is disabled for this test')\n"
        "socket.socket = _blocked\n"
        "socket.create_connection = _blocked\n"
        "socket.getaddrinfo = _blocked\n"
        f"sys.argv = ['rq2_interference', '--instances', '1', '--output', {str(output)!r}]\n"
        "from experiments.rq2_interference import main\n"
        "main()\n"
    )
    result = subprocess.run(
        [sys.executable, str(wrapper)], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert output.exists() and _digest(output)


# ---------------------------------------------------------------------------
# Manifest.
# ---------------------------------------------------------------------------

def _manifest_candidates():
    sys.path.insert(0, str(REPO_ROOT))
    import experiments._common as common

    return [name for name in dir(common) if "manifest" in name.lower()]


def test_failsnow_a_dataset_manifest_builder_exists():
    """[checklist 11] 'Dataset manifest with generator, evaluator and model
    commits plus per-RQ dataset hashes.'

    [fails now] expected: a helper writes a manifest next to each dataset, so a
    release can be traced to the code that produced it. currently
    `experiments/_common.py` has no manifest helper at all.
    """
    assert _manifest_candidates(), (
        "experiments/_common.py exposes no manifest builder "
        "(write_jsonl is the only output path)"
    )


def test_failsnow_a_manifest_file_exists_next_to_the_data():
    """[checklist 11] dataset manifest."""
    data_dir = REPO_ROOT / "data"
    manifests = sorted(data_dir.glob("*/manifest*.json")) if data_dir.exists() else []
    assert manifests, f"no manifest under {data_dir}"


def test_failsnow_manifest_records_code_and_dataset_provenance():
    """[checklist 11] 'generator, evaluator and model commits plus per-RQ
    dataset hashes.'"""
    sys.path.insert(0, str(REPO_ROOT))
    import experiments._common as common

    builder = getattr(common, "build_manifest", None)
    assert builder is not None, "experiments._common.build_manifest is missing"
    manifest = builder([{"trace_hash": "abc", "condition_id": "D4"}], experiment_tag="rq2")
    for field in ("generator_commit", "evaluator_commit", "model_commits",
                  "dataset_hashes"):
        assert field in manifest, field


# ---------------------------------------------------------------------------
# Collection health.
# ---------------------------------------------------------------------------

def test_eval_pipeline_module_is_collectable():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "test/test_eval_pipeline.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "error" not in result.stdout.lower()
    assert "test" in result.stdout

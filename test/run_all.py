"""Master test runner for all DWS-Bench test and validation suites."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

TEST_SCRIPTS = [
    "smoke_test.py",
    "smoke_test_trajectories.py",
    "test_invariants.py",
    "test_measured_factors.py",
    "test_analysis_and_eval.py",
    "test_eval_pipeline.py",
    "test_scoring.py",
]


def _child_env() -> dict:
    """Put the repo root on the child's import path.

    ``python test/<script>.py`` sets ``sys.path[0]`` to ``test/``, not the repo
    root, so every ``import generator`` in these scripts fails even though
    ``cwd`` is the repo root.
    """
    env = dict(os.environ)
    parts = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_all_tests() -> None:
    print("=" * 76)
    print("DWS-BENCH: RUNNING ALL TEST AND SMOKE TEST SUITES")
    print("=" * 76)

    env = _child_env()
    for script in TEST_SCRIPTS:
        script_path = Path(__file__).resolve().parent / script
        print(f"\n>>> Running: test/{script} ...")
        res = subprocess.run(
            [sys.executable, str(script_path)], cwd=str(REPO_ROOT), env=env
        )
        if res.returncode != 0:
            print(f"\n[FAIL] test/{script} failed with return code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 76)
    print("[SUCCESS] All test suites completed successfully!")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()

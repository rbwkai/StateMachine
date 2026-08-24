"""Master test runner for all DWS-Bench test and validation suites."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

TEST_SCRIPTS = [
    "smoke_test.py",
    "smoke_test_trajectories.py",
    "test_invariants.py",
]


def run_all_tests() -> None:
    print("=" * 76)
    print("DWS-BENCH: RUNNING ALL TEST AND SMOKE TEST SUITES")
    print("=" * 76)

    for script in TEST_SCRIPTS:
        script_path = Path(__file__).resolve().parent / script
        print(f"\n>>> Running: test/{script} ...")
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(REPO_ROOT))
        if res.returncode != 0:
            print(f"\n[FAIL] test/{script} failed with return code {res.returncode}")
            sys.exit(res.returncode)

    print("\n" + "=" * 76)
    print("[SUCCESS] All test suites completed successfully!")
    print("=" * 76)


if __name__ == "__main__":
    run_all_tests()

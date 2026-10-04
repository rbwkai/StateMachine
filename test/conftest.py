"""Make sibling packages (world, generator, render, analysis, eval) importable
from the repo root regardless of where pytest is launched."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

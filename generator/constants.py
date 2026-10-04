"""
generator/constants.py
=====================
Single owner of every threshold and version literal in the DWS-Bench pipeline.

AGENTS.md §5 makes this module the one place where thresholds ($L_{max}$,
$\\tau$) live, so callers never repeat a literal. The values below mirror the
``constants:`` block of the SPEC.md frontmatter; when one changes, SPEC.md
changes first (AGENTS.md §2 authority order) and then this file.

Layering: this module imports nothing from the repo, so ``generator`` and
``world`` stay importable without ``torch``/``transformers`` (AGENTS.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ============================================================
# Contract version stamps
#
# Records must name the code that produced them so a regenerated grid is never
# mistaken for the frozen one it replaces (AGENTS.md §6.6). They track the
# SPEC.md frontmatter `version:`; bump them with the SPEC version.
# ============================================================

SPEC_VERSION: str = "0.2.0-v2"
GENERATOR_VERSION: str = SPEC_VERSION
RENDERER_VERSION: str = SPEC_VERSION
SCORING_VERSION: str = SPEC_VERSION

# ============================================================
# Generation thresholds
# ============================================================

# $L_{max}$ — SPEC §2 $L$ is a word count (RESOLVED-1), so the ceiling is in
# words. Enforced by generator.metadata.verify_length; a tokenizer-measured
# length is an evaluation-side diagnostic, not a generation gate.
L_MAX_WORDS: int = 600

# $\tau$ — failure onset $L_f = \min\{x : A(x) < \tau\}$ (SPEC §7). Owned here,
# consumed by analysis.failure_onset (which still hardcodes its own default).
FAILURE_THRESHOLD_TAU: float = 0.70


# ============================================================
# World-size limits (SPEC constants block, TrajectorySpec)
# ============================================================

NUM_CONTAINERS_DEFAULT: int = 3
NUM_CONTAINERS_MIN: int = 2


# ============================================================
# Analysis vocabulary
# ============================================================

# SPEC §7: candidate accuracy curves, selected by $R^2$ and AIC.
CURVE_MODELS: Tuple[str, ...] = ("linear", "exponential", "sigmoid")

# SPEC constants block; eval/prompts.py still branches on these literals.
PROMPT_VERSIONS: Tuple[str, ...] = ("v1", "v2")


# ============================================================
# Decoding
# ============================================================

@dataclass(frozen=True)
class DecodingConfig:
    """
    Greedy decoding settings shared by every engine, so cross-model comparison
    varies only the model (SPEC §6).

    Duplicated by eval.models.ModelConfig and by the generate_batch defaults in
    eval/engine.py, which currently disagree on max_new_tokens (SPEC OPEN-8).
    eval/models.py::ModelConfig stays the owner until that is reconciled.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    max_new_tokens: int = 256


DECODING: DecodingConfig = DecodingConfig()

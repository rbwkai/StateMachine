"""
eval/models.py
==============
Model Registry and Standardized Generation Configurations for DWS-Bench.

Implements §13 (Model Selection) and §14 (Standardized Evaluation):
- 5 Core Models:
  1. Qwen/Qwen2.5-0.5B-Instruct (Scaling anchor - small)
  2. Qwen/Qwen2.5-3B-Instruct   (Scaling anchor - medium)
  3. Qwen/Qwen2.5-7B-Instruct   (Scaling anchor - large)
  4. meta-llama/Llama-3.2-3B-Instruct (Cross-family comparison at ~3B)
  5. allenai/OLMo-2-0425-1B-Instruct (Open architecture & weights)
- Optional Models:
  - microsoft/Phi-4-mini-instruct
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Standardized decoding configuration ensuring fair cross-model comparison.

    All core models use greedy decoding (temperature=0.0, do_sample=False) by default.
    Model revisions are pinned to specific commit hashes for reproducibility.
    """
    name: str
    hf_model_id: str
    family: str
    parameter_count_b: float
    revision: str  # Pinned commit hash for reproducibility
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    do_sample: bool = False
    system_prompt: Optional[str] = (
        "You are an expert dynamic state reasoning assistant. "
        "Answer the question based only on the given narrative state changes. "
        "Give your final answer clearly."
    )


# ============================================================
# Revision validation
# ============================================================

# A Hugging Face repo sha is 40 lowercase hex characters. Anything else is a
# placeholder or a floating ref and must be rejected before a weight download
# starts (AGENTS.md §9: no floating `main`).
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_PLACEHOLDER_REVISIONS = {"", "main", "master", "head", "latest", "none", "null"}
_PLACEHOLDER_WORDS = re.compile(r"todo|tbd|fixme|placeholder|x{3,}", re.IGNORECASE)
# A hand-typed fake hash ("a" * 40, "0123456789abcdef" repeated, "deadbeef" * 5)
# is well-formed hex, so the shape check alone accepts it. A real sha is a
# digest: it has many distinct digits and no short repeating period.
_MIN_DISTINCT_HEX_DIGITS = 3
_MAX_FAKE_PERIOD = 16


def _looks_fabricated(revision: str) -> bool:
    """True when ``revision`` is too regular to be a real commit digest."""
    if len(set(revision)) < _MIN_DISTINCT_HEX_DIGITS:
        return True
    # Periodic: the string equals its own prefix of length ``period`` repeated
    # (a trailing partial repeat included, e.g. "0123456789abcdef" * 2.5).
    for period in range(1, _MAX_FAKE_PERIOD + 1):
        if all(revision[i] == revision[i % period] for i in range(len(revision))):
            return True
    return False


def validate_pinned_revision(config: ModelConfig) -> str:
    """Return the pinned commit hash of ``config``, or raise ``ValueError``.

    Fails loudly on a placeholder (``main``, a TODO marker) or malformed
    revision instead of letting ``from_pretrained`` resolve it to whatever the
    branch points at today. Callers run this before loading weights.
    """
    revision = str(config.revision or "").strip()
    if revision.lower() in _PLACEHOLDER_REVISIONS or _PLACEHOLDER_WORDS.search(revision):
        raise ValueError(
            f"model {config.name!r} has a placeholder revision {config.revision!r}; "
            "pin a 40-character commit hash (AGENTS.md §9)"
        )
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(
            f"model {config.name!r} has an unpinned revision {config.revision!r}; "
            "expected a 40-character lowercase commit hash (AGENTS.md §9)"
        )
    if _looks_fabricated(revision):
        raise ValueError(
            f"model {config.name!r} has a fabricated-looking revision {config.revision!r}; "
            "pin the real commit hash read from the Hub (AGENTS.md §9)"
        )
    return revision


def validate_registry(configs: Dict[str, ModelConfig]) -> None:
    """Validate every registered config. Called at import so the registry itself
    cannot carry a floating revision."""
    for key, config in configs.items():
        try:
            validate_pinned_revision(config)
        except ValueError as error:
            raise ValueError(f"registry entry {key!r}: {error}") from error


# ============================================================
# Model registry
# ============================================================

# 5 Core Models (§13); revisions pinned to specific commits.
# Hashes below are the current `sha` of each repository on the Hugging Face Hub,
# read from https://huggingface.co/api/models/<id>. Re-pin deliberately before a
# release run; a wrong revision makes from_pretrained fail loudly, not silently.
CORE_MODELS: Dict[str, ModelConfig] = {
    "qwen2.5-0.5b": ModelConfig(
        name="qwen2.5-0.5b",
        hf_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        family="qwen",
        parameter_count_b=0.5,
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
    ),
    "qwen2.5-3b": ModelConfig(
        name="qwen2.5-3b",
        hf_model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen",
        parameter_count_b=3.0,
        revision="aa8e72537993ba99e69dfaafa59ed015b17504d1",
    ),
    "qwen2.5-7b": ModelConfig(
        name="qwen2.5-7b",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        parameter_count_b=7.0,
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        parameter_count_b=3.2,
        revision="0cb88a4f764b7a12671c53f0838cd831a0843b95",
    ),
    "olmo-2-1b-instruct": ModelConfig(
        name="olmo-2-1b-instruct",
        # Instruct variant: the base OLMo-2-0425-1B ships no chat template, so
        # it would be the only core model prompted without its chat format.
        hf_model_id="allenai/OLMo-2-0425-1B-Instruct",
        family="olmo",
        parameter_count_b=1.48,  # Hub safetensors total
        revision="48d788eca847d4d7548f375ad03d3c9312f6139e",
    ),
}

# Optional models for secondary exploration
OPTIONAL_MODELS: Dict[str, ModelConfig] = {
    "phi-4-mini": ModelConfig(
        name="phi-4-mini",
        hf_model_id="microsoft/Phi-4-mini-instruct",
        family="phi",
        parameter_count_b=3.8,
        revision="cfbefacb99257ffa30c83adab238a50856ac3083",
    ),
    "olmo-2-7b": ModelConfig(
        name="olmo-2-7b",
        hf_model_id="allenai/OLMo-2-1124-7B-Instruct",
        family="olmo",
        parameter_count_b=7.0,
        revision="470b1fba1ae01581f270116362ee4aa1b97f4c84",
    ),
}

validate_registry({**CORE_MODELS, **OPTIONAL_MODELS})

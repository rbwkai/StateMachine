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
  5. allenai/OLMo-2-1124-7B-Instruct or allenai/OLMo-2-1B (Open architecture & weights)
- Optional Models:
  - microsoft/Phi-4-mini-instruct
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelConfig:
    """Standardized decoding configuration ensuring fair cross-model comparison."""
    name: str
    hf_model_id: str
    family: str
    parameter_count_b: float
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    do_sample: bool = False
    system_prompt: Optional[str] = (
        "You are an expert dynamic state reasoning assistant. "
        "Answer the question based only on the given narrative state changes. "
        "Give your final answer clearly."
    )


# 5 Core Models (§13)
CORE_MODELS: Dict[str, ModelConfig] = {
    "qwen2.5-0.5b": ModelConfig(
        name="qwen2.5-0.5b",
        hf_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        family="qwen",
        parameter_count_b=0.5,
    ),
    "qwen2.5-3b": ModelConfig(
        name="qwen2.5-3b",
        hf_model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen",
        parameter_count_b=3.0,
    ),
    "qwen2.5-7b": ModelConfig(
        name="qwen2.5-7b",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        parameter_count_b=7.0,
    ),
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        parameter_count_b=3.2,
    ),
    "olmo-2-1b": ModelConfig(
        name="olmo-2-1b",
        hf_model_id="allenai/OLMo-2-1B",
        family="olmo",
        parameter_count_b=1.0,
    ),
}

# Optional models for secondary exploration
OPTIONAL_MODELS: Dict[str, ModelConfig] = {
    "phi-4-mini": ModelConfig(
        name="phi-4-mini",
        hf_model_id="microsoft/Phi-4-mini-instruct",
        family="phi",
        parameter_count_b=3.8,
    ),
    "olmo-2-7b": ModelConfig(
        name="olmo-2-7b",
        hf_model_id="allenai/OLMo-2-1124-7B-Instruct",
        family="olmo",
        parameter_count_b=7.0,
    ),
}

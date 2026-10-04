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
  5. allenai/OLMo-2-0425-1B (Open architecture & weights)
- Optional Models:
  - microsoft/Phi-4-mini-instruct
"""

from __future__ import annotations

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


# 5 Core Models (§13); revisions pinned to specific commits 
# NOTE: These revision hashes are PLACEHOLDERS and MUST be replaced with actual
# commit hashes from the respective model repositories on Hugging Face Hub.
# To find the correct commit hash:
#   1. Go to the model page on huggingface.co (e.g., https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
#   2. Click on "Files and versions" → "History" or use the API
#   3. Copy the full commit hash (40-char SHA)
# Example: "7b3c8d8e1f2a4b5c6d7e8f9a0b1c2d3e4f5a6b7c"
CORE_MODELS: Dict[str, ModelConfig] = {
    "qwen2.5-0.5b": ModelConfig(
        name="qwen2.5-0.5b",
        hf_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        family="qwen",
        parameter_count_b=0.5,
        revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # REPLACE with actual commit hash
    ),
    "qwen2.5-3b": ModelConfig(
        name="qwen2.5-3b",
        hf_model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen",
        parameter_count_b=3.0,
        revision="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",  # REPLACE with actual commit hash
    ),
    "qwen2.5-7b": ModelConfig(
        name="qwen2.5-7b",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        parameter_count_b=7.0,
        revision="cccccccccccccccccccccccccccccccccccccccc",  # REPLACE with actual commit hash
    ),
    "llama-3.2-3b": ModelConfig(
        name="llama-3.2-3b",
        hf_model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        parameter_count_b=3.2,
        revision="dddddddddddddddddddddddddddddddddddddddd",  # REPLACE with actual commit hash
    ),
    "olmo-2-1b": ModelConfig(
        name="olmo-2-1b",
        hf_model_id="allenai/OLMo-2-0425-1B",
        family="olmo",
        parameter_count_b=1.0,
        revision="eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",  # REPLACE with actual commit hash
    ),
}

# Optional models for secondary exploration
OPTIONAL_MODELS: Dict[str, ModelConfig] = {
    "phi-4-mini": ModelConfig(
        name="phi-4-mini",
        hf_model_id="microsoft/Phi-4-mini-instruct",
        family="phi",
        parameter_count_b=3.8,
        revision="ffffffffffffffffffffffffffffffffffffffff",  # REPLACE with actual commit hash
    ),
    "olmo-2-7b": ModelConfig(
        name="olmo-2-7b",
        hf_model_id="allenai/OLMo-2-1124-7B-Instruct",
        family="olmo",
        parameter_count_b=7.0,
        revision="gggggggggggggggggggggggggggggggggggggggg",  # REPLACE with actual commit hash
    ),
}

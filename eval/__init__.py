"""
eval package for DWS-Bench.
"""

from .engine import HuggingFaceEngine, InferenceEngine, MockInferenceEngine
from .eval_harness import (
    ConditionEvalSummary,
    InstanceEvalResult,
    evaluate_predictions,
    extract_answer,
    format_prompt,
)
from .models import CORE_MODELS, OPTIONAL_MODELS, ModelConfig

__all__ = [
    "format_prompt",
    "extract_answer",
    "InstanceEvalResult",
    "ConditionEvalSummary",
    "evaluate_predictions",
    "ModelConfig",
    "CORE_MODELS",
    "OPTIONAL_MODELS",
    "InferenceEngine",
    "HuggingFaceEngine",
    "MockInferenceEngine",
]

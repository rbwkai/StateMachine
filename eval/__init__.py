from .engine import HuggingFaceEngine, InferenceEngine, MockInferenceEngine, create_engine
from .eval_harness import (
    ConditionEvalSummary,
    InstanceEvalResult,
    evaluate_predictions,
    extract_answer,
    format_prompt,
)
from .models import CORE_MODELS, OPTIONAL_MODELS, ModelConfig
from .baselines import (
    BaselineResult,
    compute_stateless_baseline,
    compute_mfc_baseline,
    run_all_baselines,
    summarize_baselines,
)
from .robustness import (
    RobustnessResult,
    build_prompt_variants,
    paraphrase_narrative,
    evaluate_prompt_sensitivity,
    evaluate_paraphrase_robustness,
    run_robustness_suite,
)

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
    "create_engine",
    "BaselineResult",
    "compute_stateless_baseline",
    "compute_mfc_baseline",
    "run_all_baselines",
    "summarize_baselines",
    "RobustnessResult",
    "build_prompt_variants",
    "paraphrase_narrative",
    "evaluate_prompt_sensitivity",
    "evaluate_paraphrase_robustness",
    "run_robustness_suite",
]

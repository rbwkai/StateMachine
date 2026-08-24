"""
eval/engine.py
==============
Inference engine supporting HuggingFace Transformers, quantized execution,
chat template application, and mock dry-run evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from eval.models import ModelConfig

logger = logging.getLogger(__name__)

# Optional top-level imports for static analysis and type checking
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    HAS_TRANSFORMERS = False


class InferenceEngine:
    """Base interface for model generation engines."""

    def format_input(
        self,
        context: str,
        question: str,
        chain_of_thought: bool = False,
    ) -> str:
        """Format input narrative and question into model prompt."""
        if chain_of_thought:
            return f"Narrative:\n{context}\n\nQuestion:\n{question}\n\nLet's trace step by step:\nAnswer:"
        return f"Narrative:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    def generate_batch(
        self,
        prompts: Sequence[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        raise NotImplementedError


class MockInferenceEngine(InferenceEngine):
    """
    Mock inference engine for dry-runs, debugging, and verification
    without downloading neural weights or requiring GPUs.
    """

    def __init__(self, model_name: str = "mock-model", behavior: str = "smart_heuristic"):
        self.model_name = model_name
        self.behavior = behavior

    def generate_batch(
        self,
        prompts: Sequence[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> List[str]:
        responses: List[str] = []
        for p in prompts:
            if isinstance(p, list):
                text = " ".join([m.get("content", "") for m in p])
            else:
                text = str(p)

            if "Where is" in text or "where is" in text:
                responses.append("Answer: the green box")
            elif "True or False" in text or "true or false" in text:
                responses.append("Answer: True")
            else:
                responses.append("Answer: container")
        return responses


class HuggingFaceEngine(InferenceEngine):
    """
    Production HuggingFace Transformer inference engine with support for
    device placement, quantization, chat templates, and batched generation.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        device: str = "auto",
        precision: str = "bfloat16",
        hf_token: Optional[str] = None,
    ):
        if not HAS_TRANSFORMERS or torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "PyTorch and HuggingFace Transformers must be installed to run SLMs.\n"
                "Install them via: pip install -r requirements.txt"
            )

        self.config = model_config
        self.device_str = device
        self.precision = precision

        logger.info(f"Loading tokenizer for {model_config.hf_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.hf_model_id,
            token=hf_token,
            trust_remote_code=True,
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = self.tokenizer.unk_token or "[PAD]"
        self.tokenizer.padding_side = "left"

        # Model loading kwargs
        torch_dtype = torch.bfloat16 if precision == "bfloat16" else (
            torch.float16 if precision == "float16" else torch.float32
        )

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "token": hf_token,
        }

        if precision == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        elif precision == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch_dtype
            if device == "auto":
                model_kwargs["device_map"] = "auto"

        logger.info(f"Loading model weights for {model_config.hf_model_id} ({precision})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_model_id,
            **model_kwargs,
        )

        if device != "auto" and precision not in ("4bit", "8bit"):
            self.model = self.model.to(device)

        self.model.eval()

    def format_input(
        self,
        context: str,
        question: str,
        chain_of_thought: bool = False,
    ) -> str:
        """Apply model-specific chat template if available, else standard text."""
        system_content = self.config.system_prompt or "You are a precise state reasoning assistant."
        user_content = f"Narrative:\n{context}\n\nQuestion:\n{question}"
        if chain_of_thought:
            user_content += "\n\nPlease think step by step and conclude with 'Answer: <location>'."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return str(formatted)
            except Exception as e:
                logger.debug(f"Chat template application failed ({e}), falling back to direct prompt.")

        return f"Instructions: {system_content}\n\n{user_content}\n\nAnswer:"

    def generate_batch(
        self,
        prompts: Sequence[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> List[str]:
        if torch is None:
            raise RuntimeError("PyTorch is not available.")

        text_prompts: List[str] = []
        for p in prompts:
            if isinstance(p, list):
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                    text_prompts.append(
                        self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    text_prompts.append(" ".join([m.get("content", "") for m in p]))
            else:
                text_prompts.append(str(p))

        inputs = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample and temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        input_len = input_ids.shape[1]
        new_tokens = output_ids[:, input_len:]

        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [d.strip() for d in decoded]

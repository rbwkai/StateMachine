"""
test/test_engine_tokenization.py
================================
BOS handling in ``eval.engine`` and the fabricated-revision check in
``eval.models``. Runs without torch/transformers: the tokenizer is a fake that
records its call kwargs and prepends BOS exactly like a Llama-style tokenizer
does when ``add_special_tokens=True``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

import eval.engine as engine_module
from eval.engine import (
    ChatTemplatedPrompt,
    HuggingFaceEngine,
    is_chat_templated,
    tokenize_prompts,
    tokenizer_call_kwargs,
)
from eval.models import (
    CORE_MODELS,
    OPTIONAL_MODELS,
    ModelConfig,
    validate_pinned_revision,
    validate_registry,
)

BOS = "<|begin_of_text|>"


# ============================================================
# Fake tokenizer
# ============================================================

class FakeTokenizer:
    """Whitespace tokenizer with a Llama-3-style chat template.

    ``__call__`` prepends BOS when ``add_special_tokens`` is true (the
    transformers default), and keeps a BOS already present in the text as its
    own token, so a doubled BOS is observable in ``input_ids``.
    """

    def __init__(self, chat_template: Optional[str] = "llama3", bos_token: Optional[str] = BOS):
        self.chat_template = chat_template
        self.bos_token = bos_token
        self.calls: List[Dict[str, Any]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        prefix = self.bos_token or ""
        return prefix + body + ("<assistant>" if add_generation_prompt else "")

    def _encode(self, text: str, add_special_tokens: bool) -> List[str]:
        tokens: List[str] = [self.bos_token] if (add_special_tokens and self.bos_token) else []
        rest = text
        if self.bos_token and rest.startswith(self.bos_token):
            tokens.append(self.bos_token)
            rest = rest[len(self.bos_token):]
        tokens.extend(rest.split())
        return tokens

    def __call__(self, texts, add_special_tokens: bool = True, **kwargs):
        self.calls.append({"texts": list(texts), "add_special_tokens": add_special_tokens, **kwargs})
        return {"input_ids": [self._encode(t, add_special_tokens) for t in texts]}


def _bos_count(tokens: List[str]) -> int:
    return sum(1 for token in tokens if token == BOS)


def _engine(tokenizer: FakeTokenizer) -> HuggingFaceEngine:
    """An engine shell without weights: format_input needs only config + tokenizer."""
    engine = object.__new__(HuggingFaceEngine)
    engine.config = CORE_MODELS["llama-3.2-3b"]
    engine.tokenizer = tokenizer
    engine.max_prompt_length = 0
    return engine


CONTEXT = "The key was in the blue box. The key was moved to the red trunk."
QUESTION = "Where is the key now?"


# ============================================================
# Double BOS
# ============================================================

def test_chat_template_path_tokenizes_with_exactly_one_bos():
    tokenizer = FakeTokenizer()
    prompt = _engine(tokenizer).format_input(CONTEXT, QUESTION)

    assert isinstance(prompt, ChatTemplatedPrompt)
    assert prompt.startswith(BOS)

    inputs = tokenize_prompts(tokenizer, [prompt])
    assert tokenizer.calls[-1]["add_special_tokens"] is False
    assert _bos_count(inputs["input_ids"][0]) == 1
    assert inputs["input_ids"][0][0] == BOS


def test_plain_text_fallback_path_tokenizes_with_exactly_one_bos():
    tokenizer = FakeTokenizer(chat_template=None)
    prompt = _engine(tokenizer).format_input(CONTEXT, QUESTION)

    assert not isinstance(prompt, ChatTemplatedPrompt)
    assert prompt.startswith("Instructions: ")

    inputs = tokenize_prompts(tokenizer, [prompt])
    assert tokenizer.calls[-1]["add_special_tokens"] is True
    assert _bos_count(inputs["input_ids"][0]) == 1
    assert inputs["input_ids"][0][0] == BOS


def test_failed_chat_template_falls_back_to_plain_text_with_one_bos():
    class BrokenTemplate(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise RuntimeError("template rejects the system role")

    tokenizer = BrokenTemplate()
    prompt = _engine(tokenizer).format_input(CONTEXT, QUESTION)
    assert not isinstance(prompt, ChatTemplatedPrompt)

    inputs = tokenize_prompts(tokenizer, [prompt])
    assert tokenizer.calls[-1]["add_special_tokens"] is True
    assert _bos_count(inputs["input_ids"][0]) == 1


def test_unmarked_string_that_already_opens_with_bos_is_not_given_a_second_one():
    tokenizer = FakeTokenizer()
    text = BOS + "<user>" + QUESTION + "<assistant>"
    assert is_chat_templated(tokenizer, text)

    inputs = tokenize_prompts(tokenizer, [text, text])
    assert tokenizer.calls[-1]["add_special_tokens"] is False
    assert [_bos_count(ids) for ids in inputs["input_ids"]] == [1, 1]


def test_template_without_bos_is_still_tokenized_without_added_specials():
    """Qwen-style: the template owns its special tokens even when none is BOS."""
    tokenizer = FakeTokenizer(bos_token=None)
    prompt = _engine(tokenizer).format_input(CONTEXT, QUESTION)
    assert isinstance(prompt, ChatTemplatedPrompt)
    tokenize_prompts(tokenizer, [prompt])
    assert tokenizer.calls[-1]["add_special_tokens"] is False


def test_call_kwargs_keep_the_existing_padding_and_truncation_contract():
    for flags, add_special in (([True, True], False), ([False], True)):
        kwargs = tokenizer_call_kwargs(flags)
        assert kwargs == {
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
            "add_special_tokens": add_special,
        }


def test_mixed_batch_is_refused_rather_than_given_zero_or_two_bos():
    tokenizer = FakeTokenizer()
    templated = _engine(tokenizer).format_input(CONTEXT, QUESTION)
    with pytest.raises(ValueError, match="mixes chat-templated and plain-text"):
        tokenize_prompts(tokenizer, [templated, "Instructions: plain"])


def test_chat_templated_prompt_is_an_ordinary_string():
    prompt = ChatTemplatedPrompt(BOS + "hello")
    assert prompt == BOS + "hello"
    assert hash(prompt) == hash(BOS + "hello")
    assert isinstance(prompt, str)


class _StopAfterTokenize(Exception):
    pass


@pytest.mark.parametrize("chat_template", ["llama3", None])
def test_generate_batch_routes_every_prompt_kind_through_tokenize_prompts(monkeypatch, chat_template):
    """generate_batch must keep the ChatTemplatedPrompt marker for strings from
    format_input and add it for message lists it templates itself."""
    captured: Dict[str, Any] = {}

    def fake_tokenize(tokenizer, text_prompts):
        captured["texts"] = list(text_prompts)
        captured["inputs"] = tokenize_prompts(tokenizer, text_prompts)
        raise _StopAfterTokenize

    # generate_batch only checks that torch is present before tokenizing.
    monkeypatch.setattr(engine_module, "torch", object())
    monkeypatch.setattr(engine_module, "tokenize_prompts", fake_tokenize)

    tokenizer = FakeTokenizer(chat_template=chat_template)
    engine = _engine(tokenizer)
    from_format_input = engine.format_input(CONTEXT, QUESTION)
    messages = [{"role": "user", "content": QUESTION}]

    for batch in ([from_format_input, from_format_input], [messages, messages]):
        with pytest.raises(_StopAfterTokenize):
            engine.generate_batch(batch)
        expected_marker = chat_template is not None
        assert all(isinstance(t, ChatTemplatedPrompt) == expected_marker for t in captured["texts"])
        assert [_bos_count(ids) for ids in captured["inputs"]["input_ids"]] == [1, 1]


# ============================================================
# Fabricated revisions
# ============================================================

def _config(revision: str) -> ModelConfig:
    return ModelConfig(name="x", hf_model_id="x/y", family="x", parameter_count_b=0.1, revision=revision)


@pytest.mark.parametrize(
    "revision",
    [
        "a" * 40,
        "0" * 40,
        "ab" * 20,
        "abab" * 10,
        "a" * 39 + "b",            # two distinct digits, no period
        "abcd" * 10,
        "deadbeef" * 5,
        ("0123456789abcdef" * 3)[:40],  # "0123456789abcdef" * 2.5
    ],
)
def test_fabricated_revisions_are_rejected(revision):
    assert len(revision) == 40
    with pytest.raises(ValueError, match="fabricated-looking"):
        validate_pinned_revision(_config(revision))
    with pytest.raises(ValueError, match="registry entry 'k'"):
        validate_registry({"k": _config(revision)})


@pytest.mark.parametrize("revision", ["main", "HEAD", "todo", "TBD", "placeholder", "xxx", ""])
def test_placeholder_revisions_are_still_rejected(revision):
    with pytest.raises(ValueError):
        validate_pinned_revision(_config(revision))


def test_every_registered_revision_passes_the_fabrication_check():
    for key, config in {**CORE_MODELS, **OPTIONAL_MODELS}.items():
        assert validate_pinned_revision(config) == config.revision, key


def test_olmo_core_entry_is_the_instruct_checkpoint():
    config = CORE_MODELS["olmo-2-1b-instruct"]
    assert config.hf_model_id == "allenai/OLMo-2-0425-1B-Instruct"
    assert config.revision == "48d788eca847d4d7548f375ad03d3c9312f6139e"

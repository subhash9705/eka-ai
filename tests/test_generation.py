"""
tests/test_generation.py — Integration tests for the EKA public API.

These tests mock file I/O so they do NOT require the real model checkpoint or
tokenizer to be present. They verify the generation logic using a tiny
in-memory model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from eka_ai.config import EKAConfig
from eka_ai.model import EKA1Model
from eka_ai.generation import EKA, GenerationResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tiny_model() -> EKA1Model:
    cfg = EKAConfig(
        vocab_size=256, n_layers=2, n_heads=4, n_kv_heads=4,
        d_model=64, d_ffn=256, context_length=32,
    )
    return EKA1Model(cfg).eval()


def _make_mock_tokenizer(vocab_size: int = 256) -> MagicMock:
    """Return a mock EKATokenizer that encodes text as ASCII bytes."""
    tok = MagicMock()
    tok.vocab_size = vocab_size
    tok.eos_id = 2
    tok.bos_id = 1
    tok.USER_TOKEN = "<|user|>"
    tok.SYSTEM_TOKEN = "<|system|>"
    tok.ASSISTANT_TOKEN = "<|assistant|>"
    tok.encode.side_effect = lambda text, **kwargs: [ord(c) % vocab_size for c in text[:16]]
    tok.decode.side_effect = lambda ids: "".join(chr(max(32, i % 95 + 32)) for i in ids)
    tok.decode_token.side_effect = lambda i: chr(max(32, i % 95 + 32))
    tok.apply_chat_template.side_effect = lambda msgs, **kw: " ".join(
        m["content"] for m in msgs
    )
    return tok


def _make_eka_instance() -> EKA:
    """Create an EKA instance without any file I/O."""
    obj = object.__new__(EKA)
    obj.device = torch.device("cpu")
    obj.dtype = torch.float32
    obj._verbose = False
    obj._system_prompt = "You are a helpful assistant."
    obj.model = _make_tiny_model()
    obj.tokenizer = _make_mock_tokenizer()
    return obj


# ── GenerationResult ──────────────────────────────────────────────────────────

class TestGenerationResult:
    def test_str_returns_text(self):
        r = GenerationResult(text="hello", tokens_generated=5, elapsed=1.0, device="cpu")
        assert str(r) == "hello"

    def test_tokens_per_second(self):
        r = GenerationResult(text="x", tokens_generated=100, elapsed=2.0, device="cpu")
        assert r.tokens_per_second == pytest.approx(50.0)

    def test_repr_contains_key_info(self):
        r = GenerationResult(text="x", tokens_generated=10, elapsed=1.0, device="cuda")
        assert "cuda" in repr(r)
        assert "10" in repr(r)


# ── EKA.generate() ────────────────────────────────────────────────────────────

class TestEKAGenerate:
    def setup_method(self):
        self.model = _make_eka_instance()

    def test_returns_string_by_default(self):
        result = self.model.generate("Hello", max_new_tokens=16)
        assert isinstance(result, str)

    def test_returns_generation_result(self):
        result = self.model.generate("Hello", max_new_tokens=8, return_result=True)
        assert isinstance(result, GenerationResult)

    def test_max_new_tokens_respected(self):
        result = self.model.generate("Test", max_new_tokens=5, return_result=True)
        assert result.tokens_generated <= 5

    def test_empty_prompt_works(self):
        result = self.model.generate("", max_new_tokens=8)
        assert isinstance(result, str)

    def test_temperature_extremes(self):
        """Very low temperature should still return a string."""
        result = self.model.generate("A", max_new_tokens=4, temperature=0.01)
        assert isinstance(result, str)


# ── EKA.chat() ────────────────────────────────────────────────────────────────

class TestEKAChat:
    def setup_method(self):
        self.model = _make_eka_instance()

    def test_returns_string(self):
        reply = self.model.chat("Who was Caesar?", max_new_tokens=8)
        assert isinstance(reply, str)

    def test_with_history(self):
        history = [("Hello", "Hi there!")]
        reply = self.model.chat("Tell me more", history=history, max_new_tokens=8)
        assert isinstance(reply, str)

    def test_custom_system_prompt(self):
        reply = self.model.chat(
            "Hi",
            system_prompt="You are a pirate.",
            max_new_tokens=8,
        )
        assert isinstance(reply, str)


# ── EKA.stream() ─────────────────────────────────────────────────────────────

class TestEKAStream:
    def setup_method(self):
        self.model = _make_eka_instance()

    def test_is_iterator(self):
        gen = self.model.stream("Hello", max_new_tokens=8)
        # Must be iterable
        tokens = list(gen)
        assert isinstance(tokens, list)

    def test_yields_strings(self):
        for token in self.model.stream("Hello", max_new_tokens=4):
            assert isinstance(token, str)

    def test_stream_and_generate_consistent_length(self):
        """stream() and generate() should produce the same number of tokens."""
        torch.manual_seed(0)
        stream_tokens = list(self.model.stream("abc", max_new_tokens=10))

        torch.manual_seed(0)
        result = self.model.generate("abc", max_new_tokens=10, return_result=True)

        assert len(stream_tokens) == result.tokens_generated


# ── EKA.info() ────────────────────────────────────────────────────────────────

class TestEKAInfo:
    def setup_method(self):
        self.model = _make_eka_instance()

    def test_info_keys(self):
        info = self.model.info()
        required_keys = {
            "parameters", "device", "dtype", "vocab_size",
            "n_layers", "d_model", "context_length",
        }
        assert required_keys.issubset(info.keys())

    def test_info_parameters_positive(self):
        info = self.model.info()
        assert info["parameters"] > 0

    def test_info_device_string(self):
        info = self.model.info()
        assert isinstance(info["device"], str)

"""
tests/test_utils.py — Unit tests for sampling utility functions.
"""

from __future__ import annotations

import pytest
import torch

from eka_ai.utils import (
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_token,
)

VOCAB_SIZE = 100


def make_logits(vocab_size: int = VOCAB_SIZE, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(vocab_size)


class TestApplyTemperature:
    def test_scales_logits(self):
        logits = make_logits()
        scaled = apply_temperature(logits, 2.0)
        assert torch.allclose(scaled, logits / 2.0)

    def test_temperature_one_is_noop(self):
        logits = make_logits()
        assert torch.allclose(apply_temperature(logits, 1.0), logits)

    def test_zero_temperature_raises(self):
        with pytest.raises(ValueError):
            apply_temperature(make_logits(), 0.0)

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError):
            apply_temperature(make_logits(), -0.5)


class TestApplyTopK:
    def test_only_top_k_tokens_remain(self):
        logits = make_logits()
        k = 10
        filtered = apply_top_k(logits, k)
        # Positions not in top-k should be -inf
        non_inf = (filtered != float("-inf")).sum()
        assert non_inf == k

    def test_top_k_zero_is_noop(self):
        logits = make_logits()
        assert torch.allclose(apply_top_k(logits, 0), logits)

    def test_top_k_larger_than_vocab(self):
        logits = make_logits()
        filtered = apply_top_k(logits, VOCAB_SIZE * 2)
        # All positions should survive
        assert (filtered != float("-inf")).all()

    def test_top_1_returns_argmax(self):
        logits = make_logits()
        filtered = apply_top_k(logits, 1)
        finite_pos = (filtered != float("-inf")).nonzero().item()
        assert finite_pos == logits.argmax().item()


class TestApplyTopP:
    def test_top_p_one_is_noop(self):
        logits = make_logits()
        filtered = apply_top_p(logits, 1.0)
        assert torch.allclose(filtered, logits)

    def test_top_p_removes_low_probability_tokens(self):
        logits = make_logits()
        apply_top_k(logits, 0)  # noop
        filtered_nucleus = apply_top_p(logits, 0.5)
        # Nucleus should have fewer non-inf positions than the full vocab
        remaining = (filtered_nucleus != float("-inf")).sum()
        assert remaining < VOCAB_SIZE

    def test_output_contains_at_least_one_token(self):
        logits = make_logits()
        filtered = apply_top_p(logits, 0.0001)  # very tight
        assert (filtered != float("-inf")).sum() >= 1


class TestApplyRepetitionPenalty:
    def test_no_penalty_is_noop(self):
        logits = make_logits()
        penalised = apply_repetition_penalty(logits, list(range(10)), 1.0)
        assert torch.allclose(penalised, logits)

    def test_positive_logits_are_reduced(self):
        logits = torch.zeros(VOCAB_SIZE)
        logits[0] = 2.0  # positive
        penalised = apply_repetition_penalty(logits, [0], 2.0)
        # Positive logits should be divided by penalty
        assert penalised[0] == pytest.approx(1.0)

    def test_negative_logits_are_pushed_lower(self):
        logits = torch.zeros(VOCAB_SIZE)
        logits[5] = -1.0  # negative
        penalised = apply_repetition_penalty(logits, [5], 2.0)
        # Negative logits should be multiplied by penalty
        assert penalised[5] == pytest.approx(-2.0)

    def test_does_not_modify_original(self):
        logits = make_logits()
        original = logits.clone()
        apply_repetition_penalty(logits, [0, 1, 2], 1.5)
        assert torch.allclose(logits, original)


class TestSampleToken:
    def test_returns_valid_token_id(self):
        logits = make_logits()
        token_id = sample_token(logits)
        assert 0 <= token_id < VOCAB_SIZE

    def test_deterministic_with_very_low_temperature(self):
        """With temperature → 0, should always return argmax."""
        logits = make_logits()
        argmax = logits.argmax().item()
        results = {sample_token(logits, temperature=1e-5, top_k=0, top_p=1.0) for _ in range(20)}
        assert results == {argmax}

    def test_top_k_1_always_returns_argmax(self):
        logits = make_logits()
        argmax = logits.argmax().item()
        for _ in range(10):
            assert sample_token(logits, temperature=1.0, top_k=1) == argmax

    def test_all_sampling_params_combined(self):
        logits = make_logits()
        token_id = sample_token(
            logits,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            input_ids=list(range(10)),
            repetition_penalty=1.2,
        )
        assert 0 <= token_id < VOCAB_SIZE

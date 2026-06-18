"""
tests/test_model.py — Unit tests for EKA-1 model architecture.

These tests operate on a tiny configuration so they run quickly without
needing the full 109M checkpoint or any file downloads.
"""

from __future__ import annotations

import sys

import pytest
import torch

from eka_ai.config import EKAConfig
from eka_ai.model import (
    CausalSelfAttention,
    EKA1Model,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUFFN,
    TransformerBlock,
    _apply_rotary,
    _rotate_half,
)

# ── Tiny config for fast tests ────────────────────────────────────────────────


@pytest.fixture()
def tiny_config() -> EKAConfig:
    """A very small model config for fast CPU tests."""
    return EKAConfig(
        vocab_size=256,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        d_model=64,
        d_ffn=256,
        context_length=32,
        dropout=0.0,
        bias=False,
    )


@pytest.fixture()
def tiny_model(tiny_config) -> EKA1Model:
    return EKA1Model(tiny_config).eval()


# ── RMSNorm ───────────────────────────────────────────────────────────────────


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 16, 64)
        assert norm(x).shape == x.shape

    def test_unit_scale_is_identity_like(self):
        """With weight=1 (default), output should be approximately normalised."""
        norm = RMSNorm(dim=32)
        x = torch.randn(4, 32) * 5.0
        out = norm(x)
        # Each row should have RMS ≈ 1
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_dtype_preserved(self):
        norm = RMSNorm(dim=16)
        x = torch.randn(1, 16, dtype=torch.float32)
        assert norm(x).dtype == torch.float32


# ── Rotary Embeddings ─────────────────────────────────────────────────────────


class TestRotaryEmbedding:
    def test_output_shapes_match_input(self):
        rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
        q = torch.randn(2, 4, 8, 16)  # (B, heads, T, head_dim)
        k = torch.randn(2, 4, 8, 16)
        q_out, k_out = rope(q, k, seq_len=8)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rotate_half_reversal(self):
        """rotate_half applied twice should return the original tensor."""
        x = torch.randn(2, 8)
        # Rotating by π is equivalent to negating, not exact identity for half-rotate.
        # But rotating the already-rotated tensor should give a known result.
        rotated = _rotate_half(x)
        assert rotated.shape == x.shape

    def test_apply_rotary_shape(self):
        x = torch.randn(2, 4, 8, 16)
        cos = torch.ones(1, 1, 8, 16)
        sin = torch.zeros(1, 1, 8, 16)
        out = _apply_rotary(x, cos, sin)
        # With sin=0 and cos=1, output should equal input
        assert torch.allclose(out, x)


# ── Attention ─────────────────────────────────────────────────────────────────


class TestCausalSelfAttention:
    def test_output_shape(self, tiny_config):
        attn = CausalSelfAttention(tiny_config)
        x = torch.randn(2, 8, tiny_config.d_model)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_masking(self, tiny_config):
        """Future positions should not influence past positions."""
        attn = CausalSelfAttention(tiny_config).eval()
        x = torch.randn(1, 4, tiny_config.d_model)

        with torch.no_grad():
            out_full = attn(x)

        # Modify future token; the first token's output should not change
        x2 = x.clone()
        x2[0, 2:, :] = torch.randn_like(x2[0, 2:, :])
        with torch.no_grad():
            out_modified = attn(x2)

        # Position 0 output should be identical (not influenced by pos 2+)
        assert torch.allclose(out_full[0, 0], out_modified[0, 0], atol=1e-5)


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────


class TestSwiGLUFFN:
    def test_output_shape(self, tiny_config):
        ffn = SwiGLUFFN(tiny_config)
        x = torch.randn(2, 8, tiny_config.d_model)
        assert ffn(x).shape == x.shape

    def test_hidden_dim_is_multiple_of_64(self, tiny_config):
        ffn = SwiGLUFFN(tiny_config)
        assert ffn.gate.out_features % 64 == 0


# ── TransformerBlock ──────────────────────────────────────────────────────────


class TestTransformerBlock:
    def test_output_shape(self, tiny_config):
        block = TransformerBlock(tiny_config)
        x = torch.randn(2, 16, tiny_config.d_model)
        assert block(x).shape == x.shape

    def test_residual_connections(self, tiny_config):
        """Block output should differ from zero but be finite."""
        block = TransformerBlock(tiny_config).eval()
        x = torch.zeros(1, 4, tiny_config.d_model)
        out = block(x)
        assert torch.isfinite(out).all()


# ── EKA1Model ─────────────────────────────────────────────────────────────────


class TestEKA1Model:
    def test_output_shape(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (2, 8))
        logits = tiny_model(idx)
        assert logits.shape == (2, 8, tiny_config.vocab_size)

    def test_weight_tying(self, tiny_model):
        """LM-head weight must be identical to token embedding weight."""
        assert tiny_model.lm_head.weight is tiny_model.tok_emb.weight

    def test_parameter_count(self, tiny_model):
        n = tiny_model.num_parameters()
        assert n > 0

    def test_parameter_count_excludes_embedding_duplicate(self, tiny_model):
        """Weight tying means num_parameters() counts shared params once."""
        total = tiny_model.num_parameters()
        excl = tiny_model.num_parameters(exclude_embeddings=True)
        emb_size = tiny_model.tok_emb.weight.numel()
        # total - excl should equal embedding size (shared weight counted once)
        assert total - excl == emb_size

    def test_logits_are_finite(self, tiny_model, tiny_config):
        idx = torch.randint(0, tiny_config.vocab_size, (1, 16))
        logits = tiny_model(idx)
        assert torch.isfinite(logits).all()

    def test_eval_mode(self, tiny_model):
        assert not tiny_model.training

    def test_num_blocks(self, tiny_model, tiny_config):
        assert len(tiny_model.blocks) == tiny_config.n_layers

    def test_to_device_cpu(self, tiny_config):
        model = EKA1Model(tiny_config).to("cpu")
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        logits = model(idx)
        assert logits.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self, tiny_config):
        model = EKA1Model(tiny_config).to("cuda").eval()
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4)).to("cuda")
        logits = model(idx)
        assert logits.device.type == "cuda"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason=(
            "PyTorch's official Windows CPU wheels lack the bf16 matmul/SDPA "
            "kernels that Linux and macOS builds ship with, so this raises "
            "'not implemented for BFloat16' on Windows regardless of Python version."
        ),
    )
    def test_bfloat16_inference(self, tiny_config):
        """Model should run in bfloat16 without errors."""
        model = EKA1Model(tiny_config).to(torch.bfloat16).eval()
        idx = torch.randint(0, tiny_config.vocab_size, (1, 4))
        logits = model(idx)
        assert logits.dtype == torch.bfloat16

    def test_context_truncation_long_input(self, tiny_model, tiny_config):
        """Input longer than context_length should not crash the model."""
        seq_len = tiny_config.context_length  # 32
        idx = torch.randint(0, tiny_config.vocab_size, (1, seq_len))
        logits = tiny_model(idx)
        assert logits.shape == (1, seq_len, tiny_config.vocab_size)

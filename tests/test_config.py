"""
tests/test_config.py — Unit tests for EKAConfig.
"""

import pytest

from eka_ai.config import EKAConfig


class TestEKAConfigDefaults:
    """Verify that default values match the released 109M checkpoint."""

    def test_default_vocab_size(self):
        cfg = EKAConfig()
        assert cfg.vocab_size == 32_000

    def test_default_n_layers(self):
        assert EKAConfig().n_layers == 12

    def test_default_n_heads(self):
        assert EKAConfig().n_heads == 12

    def test_default_n_kv_heads(self):
        assert EKAConfig().n_kv_heads == 12

    def test_default_d_model(self):
        assert EKAConfig().d_model == 768

    def test_default_d_ffn(self):
        assert EKAConfig().d_ffn == 3_072

    def test_default_context_length(self):
        assert EKAConfig().context_length == 512

    def test_default_dropout(self):
        assert EKAConfig().dropout == 0.0

    def test_default_bias(self):
        assert EKAConfig().bias is False

    def test_head_dim_property(self):
        cfg = EKAConfig()
        assert cfg.head_dim == cfg.d_model // cfg.n_heads  # 64


class TestEKAConfigValidation:
    """Verify that invalid configurations raise errors."""

    def test_kv_heads_greater_than_heads_raises(self):
        with pytest.raises(ValueError, match="n_kv_heads"):
            EKAConfig(n_heads=8, n_kv_heads=12)

    def test_d_model_not_divisible_by_n_heads_raises(self):
        with pytest.raises(ValueError, match="d_model"):
            EKAConfig(d_model=769, n_heads=12)


class TestEKAConfigSerialization:
    """Round-trip to/from dict and JSON."""

    def test_to_dict_has_all_fields(self):
        cfg = EKAConfig()
        d = cfg.to_dict()
        assert "vocab_size" in d
        assert "n_layers" in d
        assert "d_model" in d

    def test_from_dict_roundtrip(self):
        cfg = EKAConfig(d_model=512, n_heads=8, n_kv_heads=8)
        cfg2 = EKAConfig.from_dict(cfg.to_dict())
        assert cfg == cfg2

    def test_json_roundtrip(self):
        cfg = EKAConfig()
        cfg2 = EKAConfig.from_json(cfg.to_json())
        assert cfg == cfg2

    def test_from_dict_with_legacy_keys(self):
        """Legacy checkpoint key names should be normalised."""
        legacy = {
            "vocab_size": 32_000,
            "n_layer": 12,  # legacy → n_layers
            "n_head": 12,  # legacy → n_heads
            "n_embd": 768,  # legacy → d_model
            "block_size": 512,  # legacy → context_length
        }
        cfg = EKAConfig.from_dict(legacy)
        assert cfg.n_layers == 12
        assert cfg.n_heads == 12
        assert cfg.d_model == 768
        assert cfg.context_length == 512

    def test_from_dict_ignores_unknown_keys(self):
        """Extra / unknown keys in the dict must be silently dropped."""
        d = EKAConfig().to_dict()
        d["unknown_future_key"] = "irrelevant_value"
        cfg = EKAConfig.from_dict(d)  # should not raise
        assert cfg == EKAConfig()

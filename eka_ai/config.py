"""
EKA-1 model configuration.

Matches the 109,529,856-parameter checkpoint trained on Project Gutenberg.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class EKAConfig:
    """
    Configuration for EKA-1 decoder-only transformer.

    All defaults correspond to the released 109M checkpoint.

    Attributes
    ----------
    vocab_size : int
        Size of the token vocabulary (SentencePiece BPE, 32 000 tokens).
    n_layers : int
        Number of transformer decoder blocks.
    n_heads : int
        Number of query attention heads.
    n_kv_heads : int
        Number of key/value heads (equals n_heads for MHA; < n_heads for GQA).
    d_model : int
        Hidden/embedding dimension.
    d_ffn : int
        Feed-forward inner dimension **before** SwiGLU halving.
        The actual up/gate projections use ``d_ffn`` units; the true hidden
        dimension after SwiGLU gating is also ``d_ffn`` (gate ⊙ up → down).
    context_length : int
        Maximum sequence length the model was trained with.
    dropout : float
        Dropout probability (0.0 at inference).
    bias : bool
        Whether linear projections include bias terms.
    rope_base : int
        Base frequency for Rotary Position Embeddings.
    norm_eps : float
        Epsilon for RMSNorm numerical stability.
    """

    vocab_size: int = 32_000
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12
    d_model: int = 768
    d_ffn: int = 3_072
    context_length: int = 512
    dropout: float = 0.0
    bias: bool = False
    rope_base: int = 10_000
    norm_eps: float = 1e-5

    # ── Aliases for legacy checkpoints that used different key names ──────────
    # These are handled in from_dict() so the dataclass itself stays clean.

    def __post_init__(self) -> None:
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return config as a plain Python dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return config as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "EKAConfig":
        """
        Build a config from a raw dict, tolerating legacy key names used in
        older checkpoints.

        Legacy → canonical key mapping
        --------------------------------
        n_layer   → n_layers
        n_head    → n_heads
        n_embd    → d_model
        block_size → context_length
        """
        _aliases = {
            "n_layer": "n_layers",
            "n_head": "n_heads",
            "n_embd": "d_model",
            "block_size": "context_length",
        }
        normalized: dict = {}
        for key, value in data.items():
            canonical = _aliases.get(key, key)
            normalized[canonical] = value

        # Keep only keys that belong to this dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in normalized.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str) -> "EKAConfig":
        """Build config from a JSON string."""
        return cls.from_dict(json.loads(json_str))

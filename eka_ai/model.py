"""
EKA-1 model architecture.

Decoder-only Transformer with:
- RMSNorm
- Rotary Position Embeddings (RoPE)
- SwiGLU feed-forward network
- Multi-Head Causal Attention (MHA) via PyTorch SDPA
- Weight tying between token embedding and LM head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eka_ai.config import EKAConfig

# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean subtraction).

    References
    ----------
    Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class RotaryEmbedding(nn.Module):
    """
    Pre-computed Rotary Position Embeddings (RoPE).

    Caches cos/sin tables up to ``max_seq_len`` at construction time.
    The cache is stored as a buffer (moves to device with the module).

    References
    ----------
    Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000) -> None:
        super().__init__()
        # Inverse frequencies: shape (head_dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin tables: shape (1, 1, max_seq_len, head_dim)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :], persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cache[:, :, :seq_len, :]
        sin = self.sin_cache[:, :, :seq_len, :]
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to tensor ``x``."""
    return x * cos + _rotate_half(x) * sin


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with Rotary Position Embeddings.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) which
    automatically selects FlashAttention when available, falling back to the
    efficient memory-saving kernel or the reference implementation.

    Supports Grouped-Query Attention (GQA) when ``n_kv_heads < n_heads``;
    for the released checkpoint ``n_kv_heads == n_heads`` (standard MHA).
    """

    def __init__(self, config: EKAConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads  # GQA repeat factor

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

        self.rope = RotaryEmbedding(self.head_dim, config.context_length, config.rope_base)
        self.attn_dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and reshape to (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, T)

        # Expand KV heads for GQA (no-op when n_rep == 1)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # PyTorch SDPA — uses FlashAttention / memory-efficient / math kernel
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
        )

        # Reassemble: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network (SwiGLU)
# ─────────────────────────────────────────────────────────────────────────────


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    output = down( silu(gate(x)) ⊙ up(x) )

    The inner dimension is rounded up to the nearest multiple of 64 for
    hardware efficiency.

    References
    ----------
    Shazeer (2020) "GLU Variants Improve Transformer"
    https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: EKAConfig) -> None:
        super().__init__()
        # Compute hidden dim: 2/3 * d_ffn (SwiGLU convention), rounded to 64
        hidden = int(config.d_ffn * 2 / 3)
        hidden = (hidden + 63) // 64 * 64

        self.gate = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.up = nn.Linear(config.d_model, hidden, bias=config.bias)
        self.down = nn.Linear(hidden, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Single pre-norm decoder block: Attention → FFN."""

    def __init__(self, config: EKAConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# EKA-1 Model
# ─────────────────────────────────────────────────────────────────────────────


class EKA1Model(nn.Module):
    """
    EKA-1: 109M parameter decoder-only Transformer.

    Architecture summary
    --------------------
    - Token embedding + weight-tied LM head
    - 12 × TransformerBlock (RMSNorm → CausalSelfAttention + RoPE → SwiGLUFFN)
    - Final RMSNorm before logits
    - No positional embedding table (RoPE handles position information)
    """

    def __init__(self, config: EKAConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_out = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embedding and LM-head share the same parameter matrix
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-style initialisation for linear layers, normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : torch.Tensor
            Token index tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        x = self.drop(self.tok_emb(idx))  # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.lm_head(x)  # (B, T, vocab_size)

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count trainable parameters."""
        params = list(self.parameters())
        if exclude_embeddings:
            params = [p for p in params if p is not self.tok_emb.weight]
        return sum(p.numel() for p in params if p.requires_grad)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> EKA1Model:
        """
        Load model from a ``.pt`` checkpoint file.

        The checkpoint must contain:
        - ``"model"``  : state_dict
        - ``"config"`` : dict or EKAConfig (optional — defaults used if absent)

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        device : torch.device, optional
            Target device. Defaults to CUDA if available, else CPU.
        dtype : torch.dtype, optional
            Cast model to this dtype after loading. ``None`` keeps the
            checkpoint dtype. On CPU ``torch.bfloat16`` is recommended.

        Returns
        -------
        EKA1Model
            Model in evaluation mode on ``device``.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Resolve config -------------------------------------------------------
        raw_cfg = checkpoint.get("config", {})
        if isinstance(raw_cfg, EKAConfig):
            config = raw_cfg
        elif isinstance(raw_cfg, dict):
            config = EKAConfig.from_dict(raw_cfg)
        else:
            config = EKAConfig()

        # Infer n_layers from state_dict if missing in checkpoint
        state = checkpoint["model"]
        if isinstance(state, dict):
            layer_indices = [int(k.split(".")[1]) for k in state.keys() if k.startswith("blocks.")]
            if layer_indices:
                config.n_layers = max(layer_indices) + 1

        # Build and load -------------------------------------------------------
        model = cls(config)
        missing, unexpected = model.load_state_dict(state, strict=False)

        # Weight tying: lm_head.weight is not in the checkpoint (expected)
        real_missing = [k for k in missing if k != "lm_head.weight"]
        if real_missing:
            import warnings

            warnings.warn(f"Missing keys in checkpoint: {real_missing[:10]}", stacklevel=2)
        if unexpected:
            import warnings

            warnings.warn(f"Unexpected keys in checkpoint: {unexpected[:10]}", stacklevel=2)

        model.eval()
        model.to(device)

        if dtype is not None:
            model = model.to(dtype)

        return model

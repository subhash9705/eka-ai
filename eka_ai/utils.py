"""
Sampling utilities for EKA-1 token generation.

Implements:
- Top-k filtering
- Top-p (nucleus) filtering
- Temperature scaling
- Repetition penalty
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by ``temperature`` to control randomness.

    Parameters
    ----------
    logits : torch.Tensor
        Raw (unnormalised) logit vector. Shape: ``(vocab_size,)``.
    temperature : float
        Sampling temperature.  Must be > 0.
        - ``temperature < 1`` → sharper / more deterministic distribution.
        - ``temperature > 1`` → flatter / more random distribution.
        - ``temperature == 1`` → no effect.

    Returns
    -------
    torch.Tensor
        Scaled logits.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Zero out all logits except the ``top_k`` highest values.

    Parameters
    ----------
    logits : torch.Tensor
        Logit vector. Shape: ``(vocab_size,)``.
    top_k : int
        Number of top tokens to keep.  Use 0 or a negative value to disable.

    Returns
    -------
    torch.Tensor
        Masked logits with -inf at filtered positions.
    """
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens whose cumulative
    probability mass exceeds ``top_p``.

    Parameters
    ----------
    logits : torch.Tensor
        Logit vector. Shape: ``(vocab_size,)``.
    top_p : float
        Cumulative probability threshold in ``(0, 1]``.  Use 1.0 to disable.

    Returns
    -------
    torch.Tensor
        Masked logits with -inf at filtered positions.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above top_p
    # (shift right by one so the last token that tips cumsum over top_p is kept)
    sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))

    # Scatter back to original ordering
    return torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """
    Penalise tokens that have already appeared in ``input_ids``.

    For positive logits (likely tokens) the score is divided by ``penalty``,
    making it less likely.  For negative logits the score is multiplied by
    ``penalty``, pushing it further below zero.

    Parameters
    ----------
    logits : torch.Tensor
        Logit vector. Shape: ``(vocab_size,)``.
    input_ids : list[int]
        Token IDs seen so far (prompt + generated tokens).
    penalty : float
        Penalty factor.  Must be >= 1.0.  Use 1.0 to disable.

    Returns
    -------
    torch.Tensor
        Penalised logits.
    """
    if penalty == 1.0:
        return logits
    logits = logits.clone()
    for token_id in set(input_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    input_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
) -> int:
    """
    Apply the full sampling pipeline to a logit vector and return a token ID.

    Pipeline
    --------
    1. Repetition penalty
    2. Temperature scaling
    3. Top-k filtering
    4. Top-p (nucleus) filtering
    5. Softmax → multinomial sample

    Parameters
    ----------
    logits : torch.Tensor
        Raw logit vector of shape ``(vocab_size,)``.
    temperature : float
        Sampling temperature (default ``1.0``).
    top_k : int
        Keep only the top-k tokens (default ``0`` = disabled).
    top_p : float
        Nucleus sampling threshold (default ``1.0`` = disabled).
    input_ids : list[int], optional
        Previously seen token IDs for repetition penalty.
    repetition_penalty : float
        Repetition penalty factor (default ``1.0`` = disabled).

    Returns
    -------
    int
        Sampled token ID.
    """
    logits = logits.float()

    if input_ids and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)

    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

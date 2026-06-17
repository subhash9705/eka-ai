"""
Tokenizer wrapper around a SentencePiece model.

Provides encode / decode and special-token helpers used by the generation
pipeline.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union


class EKATokenizer:
    """
    Thin wrapper around a SentencePiece ``SentencePieceProcessor``.

    Parameters
    ----------
    model_path : str
        Path to the ``.model`` file produced by SentencePiece training.
    """

    # Special tokens embedded in the chat prompt template
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    SYSTEM_TOKEN = "<|system|>"

    def __init__(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at '{model_path}'.\n"
                "Run EKA() to automatically download it, or call "
                "eka_ai.downloader.download_all() manually."
            )
        try:
            import sentencepiece as spm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required: pip install sentencepiece"
            ) from exc

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)

    # ── Core interface ────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode a string to a list of token IDs.

        Parameters
        ----------
        text : str
            Input text.
        add_bos : bool
            Prepend the BOS token ID.
        add_eos : bool
            Append the EOS token ID.

        Returns
        -------
        List[int]
        """
        ids: List[int] = self._sp.Encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.

        Parameters
        ----------
        ids : List[int]
            Token IDs to decode.

        Returns
        -------
        str
        """
        return self._sp.Decode(ids)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to a string."""
        return self._sp.IdToPiece(token_id).replace("▁", " ")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self._sp.GetPieceSize()

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return self._sp.bos_id()

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self._sp.eos_id()

    @property
    def pad_id(self) -> int:
        """Padding token ID (may be -1 if not defined)."""
        return self._sp.pad_id()

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self._sp.unk_id()

    # ── Chat prompt helpers ───────────────────────────────────────────────────

    def apply_chat_template(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a list of chat messages into the EKA-1 prompt template.

        Parameters
        ----------
        messages : list[dict]
            Each dict has ``"role"`` (``"user"`` / ``"assistant"``) and
            ``"content"``.
        system_prompt : str, optional
            Optional system instruction prepended before the conversation.
        add_generation_prompt : bool
            If ``True``, append the ``<|assistant|>\\n`` prefix so the model
            knows to generate an assistant turn.

        Returns
        -------
        str
            Formatted prompt string ready for tokenization.
        """
        parts: List[str] = []

        if system_prompt:
            parts.append(f"{self.SYSTEM_TOKEN}\n{system_prompt}\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"{self.USER_TOKEN}\n{content}\n")
            elif role == "assistant":
                parts.append(f"{self.ASSISTANT_TOKEN}\n{content}\n")

        if add_generation_prompt:
            parts.append(f"{self.ASSISTANT_TOKEN}\n")

        return "".join(parts)

    def __repr__(self) -> str:
        return (
            f"EKATokenizer(vocab_size={self.vocab_size}, "
            f"bos_id={self.bos_id}, eos_id={self.eos_id})"
        )

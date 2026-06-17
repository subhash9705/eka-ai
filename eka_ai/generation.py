"""
High-level EKA generation API.

This module exposes the ``EKA`` class — the primary public interface of the
``eka_ai`` package.

Usage
-----
    from eka_ai import EKA

    model = EKA()

    # Plain text generation
    text = model.generate("The Roman Empire fell because", max_new_tokens=128)

    # Single-turn chat
    reply = model.chat("Who was Ashoka?")

    # Streaming generation
    for token in model.stream("Tell me a story"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import time
from typing import Generator, Iterator, List, Optional, Tuple

import torch

from eka_ai.config import EKAConfig
from eka_ai.model import EKA1Model
from eka_ai.tokenizer import EKATokenizer
from eka_ai.downloader import get_model_path, get_tokenizer_path
from eka_ai.utils import sample_token


# ── Default system prompt ─────────────────────────────────────────────────────

_DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant trained on historical texts from Project Gutenberg. "
    "Answer questions thoughtfully and accurately."
)


# ── Generation result ─────────────────────────────────────────────────────────

class GenerationResult:
    """
    Container for a completed generation.

    Attributes
    ----------
    text : str
        Decoded output text (not including the input prompt).
    tokens_generated : int
        Number of new tokens produced.
    tokens_per_second : float
        Generation throughput.
    device : str
        Device used for inference (e.g. ``"cpu"``, ``"cuda:0"``).
    """

    def __init__(
        self,
        text: str,
        tokens_generated: int,
        elapsed: float,
        device: str,
    ) -> None:
        self.text = text
        self.tokens_generated = tokens_generated
        self.elapsed = elapsed
        self.tokens_per_second: float = tokens_generated / max(elapsed, 1e-9)
        self.device = device

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return (
            f"GenerationResult(tokens={self.tokens_generated}, "
            f"tok/s={self.tokens_per_second:.1f}, device={self.device!r})"
        )


# ── Main class ────────────────────────────────────────────────────────────────

class EKA:
    """
    EKA-1: a 109M parameter historical language model.

    On first instantiation the model checkpoint and tokenizer are downloaded
    to ``~/.cache/eka_ai/`` automatically.

    Parameters
    ----------
    device : str or torch.device, optional
        Inference device.  Defaults to ``"cuda"`` if available, else ``"cpu"``.
    dtype : torch.dtype, optional
        Model dtype.  Defaults to ``torch.bfloat16`` on CPU and the checkpoint
        dtype (usually ``float32`` or ``bfloat16``) on CUDA.
    model_path : str, optional
        Override the default cached model checkpoint path.
    tokenizer_path : str, optional
        Override the default cached tokenizer path.
    auto_download : bool
        Download model files if not already cached (default ``True``).
    system_prompt : str, optional
        System prompt used by ``chat()``.  Override to customise the
        assistant persona.
    verbose : bool
        Print loading progress messages (default ``True``).

    Examples
    --------
    >>> from eka_ai import EKA
    >>> model = EKA()
    >>> print(model.generate("Julius Caesar was", max_new_tokens=64))
    """

    def __init__(
        self,
        device: Optional[str | torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        auto_download: bool = True,
        system_prompt: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        # ── Resolve device ────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Resolve dtype ─────────────────────────────────────────────────────
        if dtype is None:
            # bfloat16 on CPU saves memory and is fast; keep fp32/bf16 on CUDA
            self.dtype = torch.bfloat16 if self.device.type == "cpu" else None
        else:
            self.dtype = dtype

        self._verbose = verbose
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

        # ── Resolve file paths ────────────────────────────────────────────────
        resolved_model = (
            str(model_path) if model_path
            else str(get_model_path(auto_download=auto_download))
        )
        resolved_tok = (
            str(tokenizer_path) if tokenizer_path
            else str(get_tokenizer_path(auto_download=auto_download))
        )

        # ── Load tokenizer ────────────────────────────────────────────────────
        if self._verbose:
            print(f"[EKA] Loading tokenizer from {resolved_tok} …", flush=True)
        self.tokenizer = EKATokenizer(resolved_tok)
        if self._verbose:
            print(
                f"[EKA] Tokenizer ready — vocab size: {self.tokenizer.vocab_size}",
                flush=True,
            )

        # ── Load model ────────────────────────────────────────────────────────
        if self._verbose:
            print(
                f"[EKA] Loading model on {self.device}"
                + (f" ({self.dtype})" if self.dtype else "")
                + " …",
                flush=True,
            )
        t0 = time.perf_counter()
        self.model = EKA1Model.from_checkpoint(
            resolved_model,
            device=self.device,
            dtype=self.dtype,
        )
        elapsed = time.perf_counter() - t0

        n_params = self.model.num_parameters()
        if self._verbose:
            print(
                f"[EKA] Model ready — {n_params / 1e6:.1f}M parameters "
                f"loaded in {elapsed:.1f}s",
                flush=True,
            )

    # ── Internal generation loop ──────────────────────────────────────────────

    @torch.no_grad()
    def _generate_ids(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Generator[int, None, None]:
        """
        Core autoregressive loop.  Yields one token ID at a time.

        The context is truncated to ``context_length`` tokens at each step.
        """
        context_length = self.model.config.context_length
        context: List[int] = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate to model's context window
            window = context[-context_length:]
            idx = torch.tensor([window], dtype=torch.long, device=self.device)

            logits = self.model(idx)          # (1, T, vocab_size)
            next_logits = logits[0, -1, :]   # (vocab_size,)

            token_id = sample_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                input_ids=context,
                repetition_penalty=repetition_penalty,
            )

            # Stop at EOS
            if token_id == self.tokenizer.eos_id:
                break

            context.append(token_id)
            yield token_id

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        return_result: bool = False,
    ) -> str | GenerationResult:
        """
        Generate text continuation for a plain-text prompt.

        Parameters
        ----------
        prompt : str
            Input text.  The model will continue from this text.
        max_new_tokens : int
            Maximum number of new tokens to generate (default 256).
        temperature : float
            Sampling temperature (default 0.8).
        top_k : int
            Top-k sampling cutoff (default 50, 0 = disabled).
        top_p : float
            Nucleus sampling threshold (default 0.95, 1.0 = disabled).
        repetition_penalty : float
            Penalty for repeated tokens (default 1.1, 1.0 = disabled).
        return_result : bool
            If ``True``, return a :class:`GenerationResult` with metadata
            instead of a plain string.

        Returns
        -------
        str or GenerationResult
            Generated text (or result object if ``return_result=True``).

        Examples
        --------
        >>> text = model.generate("The fall of Rome began", max_new_tokens=64)
        >>> print(text)
        """
        prompt_ids = self.tokenizer.encode(prompt)
        # Ensure there is at least one token (BOS) so the model always has input
        if not prompt_ids:
            prompt_ids = [self.tokenizer.bos_id]
        generated_ids: List[int] = []

        t0 = time.perf_counter()
        for token_id in self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            generated_ids.append(token_id)
        elapsed = time.perf_counter() - t0

        text = self.tokenizer.decode(generated_ids)

        if return_result:
            return GenerationResult(
                text=text,
                tokens_generated=len(generated_ids),
                elapsed=elapsed,
                device=str(self.device),
            )
        return text

    def chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        Single-turn or multi-turn chat with the model.

        Formats the conversation using the EKA-1 chat template before passing
        it to the generation loop, and strips any trailing stop markers from
        the output.

        Parameters
        ----------
        message : str
            The latest user message.
        history : list of (str, str), optional
            Prior conversation as ``[(user_turn, assistant_turn), …]``.
        system_prompt : str, optional
            Override the instance-level system prompt for this call.
        max_new_tokens : int
            Maximum new tokens (default 256).
        temperature : float
            Sampling temperature (default 0.8).
        top_k : int
            Top-k cutoff (default 50).
        top_p : float
            Nucleus threshold (default 0.95).
        repetition_penalty : float
            Repetition penalty (default 1.1).

        Returns
        -------
        str
            Assistant reply text (stripped of any stop-marker fragments).

        Examples
        --------
        >>> reply = model.chat("Who was Julius Caesar?")
        >>> print(reply)
        """
        sys_prompt = system_prompt or self._system_prompt

        # Build chat messages list
        messages: List[dict] = []
        if history:
            for user_turn, assistant_turn in history:
                messages.append({"role": "user", "content": user_turn})
                messages.append({"role": "assistant", "content": assistant_turn})
        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            system_prompt=sys_prompt,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt)
        generated_ids: List[int] = []

        for token_id in self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            generated_ids.append(token_id)

        response = self.tokenizer.decode(generated_ids)

        # Strip any leaked stop markers
        _stop_markers = [
            self.tokenizer.USER_TOKEN,
            self.tokenizer.SYSTEM_TOKEN,
            self.tokenizer.ASSISTANT_TOKEN,
            "</s>",
            "<s>",
        ]
        for marker in _stop_markers:
            if marker in response:
                response = response[: response.index(marker)]

        return response.strip()

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """
        Streaming text generation.

        Yields one decoded token (piece) at a time, allowing the caller to
        print or process output incrementally without waiting for the full
        generation to complete.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        max_new_tokens : int
            Maximum new tokens (default 256).
        temperature : float
            Sampling temperature (default 0.8).
        top_k : int
            Top-k cutoff (default 50).
        top_p : float
            Nucleus threshold (default 0.95).
        repetition_penalty : float
            Repetition penalty (default 1.1).

        Yields
        ------
        str
            Decoded text for each new token.

        Examples
        --------
        >>> for token in model.stream("The ancient library of"):
        ...     print(token, end="", flush=True)
        """
        prompt_ids = self.tokenizer.encode(prompt)

        for token_id in self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            yield self.tokenizer.decode_token(token_id)

    def stream_chat(
        self,
        message: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> Iterator[str]:
        """
        Streaming chat generation.

        Like :meth:`stream` but uses the chat template for formatting.  Yields
        one token piece at a time.

        Parameters
        ----------
        message : str
            Latest user message.
        history : list of (str, str), optional
            Prior turns as ``[(user, assistant), …]``.
        system_prompt : str, optional
            Override the instance-level system prompt.
        max_new_tokens : int
            Maximum new tokens (default 256).
        temperature : float
            Sampling temperature (default 0.8).
        top_k : int
            Top-k cutoff (default 50).
        top_p : float
            Nucleus threshold (default 0.95).
        repetition_penalty : float
            Repetition penalty (default 1.1).

        Yields
        ------
        str
            Decoded text piece for each new token.
        """
        sys_prompt = system_prompt or self._system_prompt
        messages: List[dict] = []
        if history:
            for u, a in history:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            system_prompt=sys_prompt,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt)

        _stop_markers = [
            self.tokenizer.USER_TOKEN,
            self.tokenizer.SYSTEM_TOKEN,
            self.tokenizer.ASSISTANT_TOKEN,
            "</s>",
        ]

        buffer = ""
        for token_id in self._generate_ids(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            piece = self.tokenizer.decode_token(token_id)
            buffer += piece

            # Check for stop markers in accumulated buffer
            stop_early = False
            for marker in _stop_markers:
                if marker in buffer:
                    # Yield only the part before the marker
                    before_marker = buffer[: buffer.index(marker)]
                    if before_marker:
                        yield before_marker
                    stop_early = True
                    break

            if stop_early:
                break
            yield piece

    # ── Utility ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = self.model.num_parameters()
        return (
            f"EKA(params={n / 1e6:.1f}M, device={self.device!r}, "
            f"dtype={self.dtype})"
        )

    @property
    def config(self) -> EKAConfig:
        """The underlying model configuration."""
        return self.model.config

    def info(self) -> dict:
        """
        Return a summary dict with model metadata.

        Returns
        -------
        dict
            Keys: ``version``, ``parameters``, ``device``, ``dtype``,
            ``vocab_size``, ``n_layers``, ``d_model``, ``context_length``.
        """
        from eka_ai import __version__
        cfg = self.model.config
        return {
            "version": __version__,
            "parameters": self.model.num_parameters(),
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "vocab_size": cfg.vocab_size,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "context_length": cfg.context_length,
        }

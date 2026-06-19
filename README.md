<div align="center">

# EKA-1

**A 109M parameter historical language model trained from scratch on Project Gutenberg**

[![PyPI version](https://img.shields.io/pypi/v/eka-ai.svg)](https://pypi.org/project/eka-ai/)
[![Python](https://img.shields.io/pypi/pyversions/eka-ai.svg)](https://pypi.org/project/eka-ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://github.com/eka-ai/eka-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/eka-ai/eka-ai/actions/workflows/tests.yml)

</div>

---

EKA-1 is a **decoder-only Transformer** with **109,529,856 parameters**, trained entirely from scratch on classical texts from [Project Gutenberg](https://www.gutenberg.org/). It features a modern architecture — RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU activation, multi-head causal attention via PyTorch SDPA, and weight tying — packed into a library with a clean, one-line API.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Model Architecture](#model-architecture)
- [Model Configuration](#model-configuration)
- [Examples](#examples)
- [Development](#development)
- [Publishing to PyPI](#publishing-to-pypi)
- [License](#license)

---

## Features

| Feature | Status |
|---|---|
| Automatic model & tokenizer download | ✅ |
| CPU inference | ✅ |
| CUDA inference | ✅ |
| Text generation | ✅ |
| Chat interface (single & multi-turn) | ✅ |
| Streaming generation | ✅ |
| Temperature sampling | ✅ |
| Top-k sampling | ✅ |
| Top-p (nucleus) sampling | ✅ |
| Repetition penalty | ✅ |
| Context truncation | ✅ |
| PEP 561 typed package | ✅ |

---

## Installation

**From PyPI (recommended):**

```bash
pip install eka-ai
```

**From source:**

```bash
git clone https://github.com/eka-ai/eka-ai.git
cd eka-ai
pip install -e ".[dev]"
```

**Requirements:**

- Python ≥ 3.9
- PyTorch ≥ 2.1
- sentencepiece ≥ 0.1.99
- gdown ≥ 4.7.3

> [!IMPORTANT]
> **System Requirements & Recommendations:**
> * **Memory (RAM)**: We recommend a minimum of **8 GB RAM** overall, with at least **3 GB of free RAM** dedicated to model execution.
> * **Hardware Acceleration**: For faster response times and better usage, it is highly recommended to run this package on a **GPU**.
> * **Cloud Notebooks**: We strongly recommend running this package in **Google Colab** or **Kaggle Notebooks** with GPU acceleration enabled for the fastest speeds.

On first use, `EKA()` automatically downloads `eka_model.pt` (~424 MB) and `tokenizer.model` (~768 KB) to `~/.cache/eka_ai/`.

> **Custom cache location:** Set the `EKA_CACHE_DIR` environment variable to override the default cache directory.

---


## Quick Start

```python
from eka_ai import EKA

# Loads model on first call — downloads files automatically
model = EKA()

# ── Text generation ────────────────────────────────────────
print(
    model.generate(
        "Tell me about the Roman Empire",
        max_new_tokens=128,
    )
)

# ── Single-turn chat ───────────────────────────────────────
print(
    model.chat(
        "Who was Ashoka?"
    )
)

# ── Streaming generation ───────────────────────────────────
for token in model.stream(
    "Tell me a story"
):
    print(token, end="", flush=True)
```

---

## API Reference

### `EKA(...)` — constructor

```python
model = EKA(
    device=None,           # "cpu" | "cuda" | torch.device — auto-detected
    dtype=None,            # torch.bfloat16 on CPU by default
    model_path=None,       # override cached model path
    tokenizer_path=None,   # override cached tokenizer path
    auto_download=True,    # download files if not cached
    system_prompt=None,    # custom system prompt for chat()
    verbose=True,          # print loading progress
)
```

---

### `model.generate(prompt, ...)` → `str`

Generate a text continuation for a plain prompt.

```python
text = model.generate(
    prompt,
    max_new_tokens=256,        # int   — max tokens to generate
    temperature=0.8,           # float — sampling temperature (> 0)
    top_k=50,                  # int   — top-k cutoff (0 = disabled)
    top_p=0.95,                # float — nucleus threshold (1.0 = disabled)
    repetition_penalty=1.1,    # float — penalty for repeated tokens (1.0 = disabled)
    return_result=False,       # bool  — return GenerationResult with metadata
)
```

When `return_result=True`, returns a `GenerationResult` with attributes:
- `.text` — the generated text
- `.tokens_generated` — number of new tokens
- `.tokens_per_second` — generation throughput
- `.device` — device used

---

### `model.chat(message, ...)` → `str`

Single-turn or multi-turn chat using the built-in chat template.

```python
reply = model.chat(
    message,
    history=None,              # list of (user, assistant) tuples
    system_prompt=None,        # override instance system prompt
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.1,
)
```

**Multi-turn example:**

```python
history = []
reply1 = model.chat("Who was Julius Caesar?")
history.append(("Who was Julius Caesar?", reply1))

reply2 = model.chat("What were his greatest conquests?", history=history)
```

---

### `model.stream(prompt, ...)` → `Iterator[str]`

Streaming generation — yields one decoded token piece at a time.

```python
for piece in model.stream(
    "The ancient library of",
    max_new_tokens=128,
    temperature=0.8,
):
    print(piece, end="", flush=True)
```

---

### `model.stream_chat(message, ...)` → `Iterator[str]`

Like `stream()` but uses the chat template.

```python
for piece in model.stream_chat("Tell me about Socrates"):
    print(piece, end="", flush=True)
```

---

### `model.info()` → `dict`

```python
info = model.info()
# {
#   "version": "1.0.0",
#   "parameters": 109529856,
#   "device": "cpu",
#   "dtype": "torch.bfloat16",
#   "vocab_size": 32000,
#   "n_layers": 12,
#   "n_heads": 12,
#   "d_model": 768,
#   "context_length": 512,
# }
```

---

## Model Architecture

EKA-1 is a decoder-only Transformer with the following design choices:

| Component | Implementation |
|---|---|
| Normalisation | **RMSNorm** (no bias, no mean subtraction) |
| Position encoding | **Rotary Position Embeddings (RoPE)** — no learned position table |
| Feed-forward | **SwiGLU** — `down(silu(gate(x)) ⊙ up(x))` |
| Attention | **Multi-Head Causal Attention** via PyTorch `scaled_dot_product_attention` |
| Weight sharing | **Weight tying** — embedding and LM-head share parameters |
| Architecture | Pre-norm residual blocks |
| Bias | None (all linear layers are bias-free) |

The attention implementation uses PyTorch's built-in SDPA which automatically selects the optimal kernel: FlashAttention 2 (when available), memory-efficient attention, or the reference implementation.

---

## Model Configuration

```json
{
  "vocab_size": 32000,
  "n_layers": 12,
  "n_heads": 12,
  "n_kv_heads": 12,
  "d_model": 768,
  "d_ffn": 3072,
  "context_length": 512,
  "dropout": 0.0,
  "bias": false
}
```

| Parameter | Value |
|---|---|
| Total parameters | **109,529,856** |
| Embedding dimension | 768 |
| Attention heads | 12 × 64-dim heads |
| Transformer layers | 12 |
| FFN hidden dim | 2048 (after SwiGLU 2/3 ratio) |
| Vocabulary | 32,000 BPE tokens (SentencePiece) |
| Context length | 512 tokens |
| Training data | Project Gutenberg |

---

## Examples

Run the provided example scripts from the repository root:

```bash
# Interactive text generation
python examples/generation.py --prompt "The fall of Rome" --max_tokens 200

# Interactive chat REPL
python examples/chat.py
python examples/chat.py --stream       # streaming mode
python examples/chat.py --device cuda  # GPU

# Throughput benchmark
python examples/benchmark.py --runs 10 --max_tokens 256
python examples/benchmark.py --device cuda
```

---


## Citation

If you use EKA-1 in your research, please cite:

```bibtex
@misc{eka1_2024,
  title  = {{EKA-1}: A 109M Parameter Historical Language Model},
  author = {chvkrsubhash},
  year   = {2026},
  url    = {https://github.com/eka-ai/eka-ai},
  note   = {Trained from scratch on Project Gutenberg}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

The training data is sourced from [Project Gutenberg](https://www.gutenberg.org/), which consists of public-domain works.

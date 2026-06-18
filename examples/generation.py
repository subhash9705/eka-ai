"""
examples/generation.py — Basic text generation with EKA-1.

Demonstrates the model.generate() API with various prompts and sampling settings.

Usage:
    python examples/generation.py
    python examples/generation.py --prompt "The Byzantine Empire" --max_tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from eka_ai import EKA

DEMO_PROMPTS = [
    "The Roman Empire fell because",
    "In the ancient city of Athens, the philosophers would gather",
    "The Silk Road connected the civilisations of",
    "Ashoka the Great, after the Kalinga War,",
    "The library of Alexandria held",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EKA-1 text generation demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--rep_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu / cuda)")
    args = parser.parse_args()

    print("=" * 64)
    print("  EKA-1 — Text Generation Demo")
    print("=" * 64)

    model = EKA(device=args.device)

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS

    for prompt in prompts:
        print(f"\nPrompt : {prompt!r}")
        print("-" * 56)

        result = model.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            return_result=True,
        )

        print(f"{prompt}{result.text}")
        print(
            f"\n[{result.tokens_generated} tokens | "
            f"{result.tokens_per_second:.1f} tok/s | "
            f"device={result.device}]"
        )
        print("=" * 64)


if __name__ == "__main__":
    main()

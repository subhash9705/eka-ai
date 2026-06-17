"""
examples/chat.py — Interactive multi-turn chat with EKA-1.

Provides a simple REPL (Read-Eval-Print Loop) for chatting with the model.

Usage:
    python examples/chat.py
    python examples/chat.py --stream           # stream tokens as they are generated
    python examples/chat.py --device cpu       # force CPU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from eka_ai import EKA

BANNER = r"""
  ███████╗██╗  ██╗ █████╗       ██████╗██╗  ██╗ █████╗ ████████╗
  ██╔════╝██║ ██╔╝██╔══██╗     ██╔════╝██║  ██║██╔══██╗╚══██╔══╝
  █████╗  █████╔╝ ███████║     ██║     ███████║███████║   ██║   
  ██╔══╝  ██╔═██╗ ██╔══██║     ██║     ██╔══██║██╔══██║   ██║   
  ███████╗██║  ██╗██║  ██║     ╚██████╗██║  ██║██║  ██║   ██║   
  ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝      ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
                 109M · Historical Language Model
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EKA-1 interactive chat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stream", action="store_true", help="Stream tokens as generated")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k cutoff")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus threshold")
    parser.add_argument("--rep_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Custom system prompt",
    )
    args = parser.parse_args()

    print(BANNER)
    model = EKA(device=args.device, system_prompt=args.system)
    print("\nType 'quit' or 'exit' to leave. Type 'clear' to reset conversation history.")
    print("─" * 64)

    history: List[Tuple[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[EKA] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("[EKA] Goodbye!")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[EKA] Conversation history cleared.")
            continue

        print("\nEKA: ", end="", flush=True)

        if args.stream:
            response_parts: List[str] = []
            for piece in model.stream_chat(
                user_input,
                history=history,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
            ):
                print(piece, end="", flush=True)
                response_parts.append(piece)
            response = "".join(response_parts)
        else:
            response = model.chat(
                user_input,
                history=history,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.rep_penalty,
            )
            print(response)

        print()  # newline after streamed output
        history.append((user_input, response))


if __name__ == "__main__":
    main()

"""
EKA-AI: A 109M parameter historical language model trained from scratch on Project Gutenberg.

Usage:
    from eka_ai import EKA

    model = EKA()
    print(model.generate("Tell me about the Roman Empire", max_new_tokens=128))
    print(model.chat("Who was Ashoka?"))
    for token in model.stream("Tell me a story"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

from eka_ai.config import EKAConfig
from eka_ai.generation import EKA

__all__ = ["EKA", "EKAConfig"]
__version__ = "1.0.1"
__author__ = "EKA-AI Contributors"
__license__ = "Apache-2.0"

"""
examples/benchmark.py — Throughput and latency benchmark for EKA-1.

Measures tokens-per-second across different batch configurations and reports
statistics useful for comparing CPU vs CUDA performance.

Usage:
    python examples/benchmark.py
    python examples/benchmark.py --runs 10 --max_tokens 256 --device cuda
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eka_ai import EKA


BENCHMARK_PROMPT = (
    "The ancient historians recorded that the great empires of the world "
    "rose and fell in cycles, each leaving behind a legacy of culture, "
    "science, and philosophy that would shape the generations to come."
)


def run_benchmark(
    model: EKA,
    prompt: str,
    max_new_tokens: int,
    runs: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict:
    """Run ``runs`` generation passes and collect timing statistics."""
    latencies: list[float] = []
    token_counts: list[int] = []

    print(f"Running {runs} benchmark iterations …", flush=True)
    for i in range(runs):
        t0 = time.perf_counter()
        result = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_result=True,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        token_counts.append(result.tokens_generated)
        tps = result.tokens_generated / max(elapsed, 1e-9)
        print(f"  Run {i + 1:2d}: {result.tokens_generated:4d} tokens  "
              f"{tps:7.1f} tok/s  {elapsed:.2f}s", flush=True)

    throughputs = [t / max(l, 1e-9) for t, l in zip(token_counts, latencies)]

    return {
        "runs": runs,
        "prompt_tokens": len(model.tokenizer.encode(prompt)),
        "avg_tokens_generated": statistics.mean(token_counts),
        "avg_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "avg_throughput_tps": statistics.mean(throughputs),
        "median_throughput_tps": statistics.median(throughputs),
        "peak_throughput_tps": max(throughputs),
    }


def print_report(stats: dict, model: EKA) -> None:
    info = model.info()
    print("\n" + "=" * 64)
    print("  EKA-1 Benchmark Report")
    print("=" * 64)
    print(f"  Model parameters : {info['parameters'] / 1e6:.1f} M")
    print(f"  Device           : {info['device']}")
    print(f"  Dtype            : {info['dtype']}")
    print(f"  Prompt tokens    : {stats['prompt_tokens']}")
    print(f"  Runs             : {stats['runs']}")
    print(f"  Avg tokens gen.  : {stats['avg_tokens_generated']:.0f}")
    print()
    print("  Latency (seconds)")
    print(f"    Mean   : {stats['avg_latency_s']:.3f}s")
    print(f"    Median : {stats['median_latency_s']:.3f}s")
    print(f"    Min    : {stats['min_latency_s']:.3f}s")
    print(f"    Max    : {stats['max_latency_s']:.3f}s")
    print()
    print("  Throughput (tokens/second)")
    print(f"    Mean   : {stats['avg_throughput_tps']:.1f} tok/s")
    print(f"    Median : {stats['median_throughput_tps']:.1f} tok/s")
    print(f"    Peak   : {stats['peak_throughput_tps']:.1f} tok/s")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EKA-1 throughput benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument("--max_tokens", type=int, default=128, help="Tokens per run")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    args = parser.parse_args()

    model = EKA(device=args.device)
    prompt = args.prompt or BENCHMARK_PROMPT

    # Warmup pass (not included in stats)
    print("Warming up …", flush=True)
    model.generate(prompt, max_new_tokens=16, temperature=args.temperature)
    print("Warmup complete.\n", flush=True)

    stats = run_benchmark(
        model,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        runs=args.runs,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print_report(stats, model)


if __name__ == "__main__":
    main()

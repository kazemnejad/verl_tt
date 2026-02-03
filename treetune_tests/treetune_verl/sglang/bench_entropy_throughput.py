# Copyright 2025 Individual Contributor: Amirhossein Kazemnejad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark comparing sglang throughput with and without entropy patches.

Usage:
    # Quick test with defaults
    python bench_entropy_throughput.py

    # Full benchmark
    python bench_entropy_throughput.py --model-path meta-llama/Llama-3.1-8B-Instruct \
        --num-prompts 500 --input-len 512 --output-len 256 --runs 5

    # Save results
    python bench_entropy_throughput.py --output-json results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from typing import Any

import numpy as np
import torch

# Check GPU availability early
if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires CUDA GPUs")


@dataclasses.dataclass
class BenchResult:
    """Single benchmark run result."""

    output_throughput: float  # tok/s
    total_latency: float  # seconds
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float  # req/s


@dataclasses.dataclass
class AggregatedResult:
    """Aggregated results across multiple runs."""

    output_throughput_mean: float
    output_throughput_std: float
    total_latency_mean: float
    total_latency_std: float
    request_throughput_mean: float
    request_throughput_std: float
    total_input_tokens: int
    total_output_tokens_mean: float
    runs: int

    @classmethod
    def from_results(cls, results: list[BenchResult]) -> AggregatedResult:
        out_tps = [r.output_throughput for r in results]
        lats = [r.total_latency for r in results]
        req_tps = [r.request_throughput for r in results]
        out_toks = [r.total_output_tokens for r in results]
        return cls(
            output_throughput_mean=float(np.mean(out_tps)),
            output_throughput_std=float(np.std(out_tps)),
            total_latency_mean=float(np.mean(lats)),
            total_latency_std=float(np.std(lats)),
            request_throughput_mean=float(np.mean(req_tps)),
            request_throughput_std=float(np.std(req_tps)),
            total_input_tokens=results[0].total_input_tokens,
            total_output_tokens_mean=float(np.mean(out_toks)),
            runs=len(results),
        )


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def create_random_requests(
    tokenizer,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create random token sequences for benchmarking.

    Returns list of dicts with 'prompt_ids' and 'output_len'.
    """
    np_rng = np.random.RandomState(seed)

    vocab_size = tokenizer.vocab_size
    requests = []
    for _ in range(num_prompts):
        # Generate random token IDs (avoid special tokens by using range 100 to vocab_size-100)
        safe_start = min(100, vocab_size - 1)
        safe_end = max(safe_start + 1, vocab_size - 100)
        prompt_ids = np_rng.randint(safe_start, safe_end, size=input_len).tolist()
        requests.append(
            {
                "prompt_ids": prompt_ids,
                "output_len": output_len,
            }
        )
    return requests


def run_baseline_benchmark(
    model_path: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
    skip_warmup: bool = False,
) -> BenchResult:
    """Run benchmark with vanilla sglang (no entropy patches)."""
    from sglang.bench_serving import get_tokenizer
    from sglang.srt.entrypoints.engine import Engine

    print("\n[Baseline] Launching vanilla sglang Engine...")
    engine = Engine(model_path=model_path, tp_size=1)

    try:
        tokenizer = get_tokenizer(model_path)  # type: ignore[attr-defined]
        requests = create_random_requests(tokenizer, num_prompts, input_len, output_len, seed)

        # Warmup
        if not skip_warmup:
            print("[Baseline] Warmup run...")
            warmup_reqs = requests[: min(16, num_prompts)]
            _ = engine.generate(
                input_ids=[req["prompt_ids"] for req in warmup_reqs],
                sampling_params=[{"max_new_tokens": 16, "temperature": 0.8} for _ in warmup_reqs],
                return_logprob=True,  # Match benchmark settings
            )
            time.sleep(0.5)

        # Benchmark
        print(f"[Baseline] Running benchmark with {num_prompts} prompts...")
        input_ids_list = [req["prompt_ids"] for req in requests]
        sampling_params = [
            {"max_new_tokens": req["output_len"], "temperature": 0.8, "ignore_eos": True} for req in requests
        ]

        start = time.perf_counter()
        outputs = engine.generate(
            input_ids=input_ids_list,
            sampling_params=sampling_params,
            # Use return_logprob=True to match entropy benchmark (fair comparison)
            return_logprob=True,
        )
        latency = time.perf_counter() - start

        total_input = sum(len(ids) for ids in input_ids_list)
        total_output = sum(o["meta_info"]["completion_tokens"] for o in outputs)  # type: ignore[index]

        return BenchResult(
            output_throughput=total_output / latency,
            total_latency=latency,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            request_throughput=num_prompts / latency,
        )
    finally:
        engine.shutdown()
        clear_cuda_cache()


def run_entropy_benchmark(
    model_path: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
    skip_warmup: bool = False,
) -> BenchResult:
    """Run benchmark with entropy patches enabled and return_logprob=True."""
    # Must apply patches BEFORE importing Engine for this run
    import sglang.srt.entrypoints.engine as engine_mod
    from sglang.bench_serving import get_tokenizer
    from sglang.srt.entrypoints.engine import Engine
    from treetune_verl.sglang.entropy import (
        apply_parent_patches,
        custom_run_scheduler_process,
    )

    # Set env var for full-vocab entropy (matches typical training use)
    os.environ["TREETUNE_ENTROPY_TOP_K"] = "0"

    # Monkey-patch scheduler entry-point
    engine_mod.run_scheduler_process = custom_run_scheduler_process

    # Apply parent-process patches (TokenizerManager)
    apply_parent_patches()

    print("\n[Entropy] Launching entropy-patched sglang Engine...")
    engine = Engine(model_path=model_path, tp_size=1)

    try:
        tokenizer = get_tokenizer(model_path)  # type: ignore[attr-defined]
        requests = create_random_requests(tokenizer, num_prompts, input_len, output_len, seed)

        # Warmup
        if not skip_warmup:
            print("[Entropy] Warmup run...")
            warmup_reqs = requests[: min(16, num_prompts)]
            _ = engine.generate(
                input_ids=[req["prompt_ids"] for req in warmup_reqs],
                sampling_params=[{"max_new_tokens": 16, "temperature": 0.8} for _ in warmup_reqs],
                return_logprob=True,
            )
            time.sleep(0.5)

        # Benchmark
        print(f"[Entropy] Running benchmark with {num_prompts} prompts...")
        input_ids_list = [req["prompt_ids"] for req in requests]
        sampling_params = [
            {"max_new_tokens": req["output_len"], "temperature": 0.8, "ignore_eos": True} for req in requests
        ]

        start = time.perf_counter()
        outputs = engine.generate(
            input_ids=input_ids_list,
            sampling_params=sampling_params,
            return_logprob=True,  # Required for entropy extraction
        )
        latency = time.perf_counter() - start

        total_input = sum(len(ids) for ids in input_ids_list)
        total_output = sum(o["meta_info"]["completion_tokens"] for o in outputs)  # type: ignore[index]

        # Verify entropy was actually computed
        sample_entropy = outputs[0]["meta_info"].get("output_token_entropy")  # type: ignore[index]
        if sample_entropy is None or len(sample_entropy) == 0:
            print("[WARNING] Entropy values not found in output! Patches may not be applied correctly.")

        return BenchResult(
            output_throughput=total_output / latency,
            total_latency=latency,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            request_throughput=num_prompts / latency,
        )
    finally:
        engine.shutdown()
        clear_cuda_cache()


def print_comparison_table(
    baseline: AggregatedResult,
    entropy: AggregatedResult,
):
    """Print formatted comparison table."""
    delta_throughput = (
        (entropy.output_throughput_mean - baseline.output_throughput_mean) / baseline.output_throughput_mean * 100
    )
    delta_latency = (entropy.total_latency_mean - baseline.total_latency_mean) / baseline.total_latency_mean * 100

    print("\n" + "=" * 70)
    print(" Entropy Throughput Benchmark Results ".center(70, "="))
    print("=" * 70)
    print(f"{'Metric':<35} {'Baseline':>15} {'Entropy':>15}")
    print("-" * 70)

    # Output throughput
    b_out = f"{baseline.output_throughput_mean:.1f}"
    e_out = f"{entropy.output_throughput_mean:.1f}"
    if baseline.runs > 1:
        b_out = f"{baseline.output_throughput_mean:.1f} +/- {baseline.output_throughput_std:.1f}"
        e_out = f"{entropy.output_throughput_mean:.1f} +/- {entropy.output_throughput_std:.1f}"
    print(f"{'Output throughput (tok/s)':<35} {b_out:>18} {e_out:>18}")

    # Total latency
    b_lat = f"{baseline.total_latency_mean:.2f}"
    e_lat = f"{entropy.total_latency_mean:.2f}"
    if baseline.runs > 1:
        b_lat = f"{baseline.total_latency_mean:.2f} +/- {baseline.total_latency_std:.2f}"
        e_lat = f"{entropy.total_latency_mean:.2f} +/- {entropy.total_latency_std:.2f}"
    print(f"{'Total latency (s)':<35} {b_lat:>18} {e_lat:>18}")

    # Request throughput
    b_req = f"{baseline.request_throughput_mean:.2f}"
    e_req = f"{entropy.request_throughput_mean:.2f}"
    if baseline.runs > 1:
        b_req = f"{baseline.request_throughput_mean:.2f} +/- {baseline.request_throughput_std:.2f}"
        e_req = f"{entropy.request_throughput_mean:.2f} +/- {entropy.request_throughput_std:.2f}"
    print(f"{'Request throughput (req/s)':<35} {b_req:>18} {e_req:>18}")

    print("-" * 70)
    print(f"{'Input tokens':<35} {baseline.total_input_tokens:>18,} {entropy.total_input_tokens:>18,}")
    b_out_tok = f"{baseline.total_output_tokens_mean:,.0f}"
    e_out_tok = f"{entropy.total_output_tokens_mean:,.0f}"
    print(f"{'Output tokens (mean)':<35} {b_out_tok:>18} {e_out_tok:>18}")
    print(f"{'Runs':<35} {baseline.runs:>15} {entropy.runs:>15}")
    print("-" * 70)
    print(f"{'Throughput delta':<35} {delta_throughput:>15.2f}%")
    print(f"{'Latency delta':<35} {delta_latency:>15.2f}%")
    print("=" * 70)

    if delta_throughput < -10:
        print("\n[!] Significant throughput regression detected (>10%)")
    elif delta_throughput < -5:
        print("\n[i] Moderate throughput regression detected (5-10%)")
    else:
        print("\n[+] Throughput overhead acceptable (<5%)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sglang throughput with and without entropy patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model path (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to benchmark",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=256,
        help="Input sequence length (tokens)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Output sequence length (tokens)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark runs for averaging",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save JSON results",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup run before timing",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run baseline benchmark only (no entropy)",
    )
    parser.add_argument(
        "--entropy-only",
        action="store_true",
        help="Run entropy benchmark only (no baseline)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(" Entropy Throughput Benchmark ".center(70))
    print("=" * 70)
    print(f"Model:       {args.model_path}")
    print(f"Prompts:     {args.num_prompts}")
    print(f"Input len:   {args.input_len}")
    print(f"Output len:  {args.output_len}")
    print(f"Seed:        {args.seed}")
    print(f"Runs:        {args.runs}")
    print("=" * 70)

    baseline_results: list[BenchResult] = []
    entropy_results: list[BenchResult] = []

    # Run baseline benchmarks
    if not args.entropy_only:
        print("\n" + "-" * 40)
        print(" BASELINE BENCHMARKS ".center(40, "-"))
        print("-" * 40)
        for i in range(args.runs):
            print(f"\n--- Baseline Run {i + 1}/{args.runs} ---")
            result = run_baseline_benchmark(
                model_path=args.model_path,
                num_prompts=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                seed=args.seed,
                skip_warmup=args.skip_warmup,
            )
            baseline_results.append(result)
            print(f"  Output throughput: {result.output_throughput:.1f} tok/s")
            print(f"  Total latency:     {result.total_latency:.2f} s")

    # Run entropy benchmarks
    if not args.baseline_only:
        print("\n" + "-" * 40)
        print(" ENTROPY BENCHMARKS ".center(40, "-"))
        print("-" * 40)
        for i in range(args.runs):
            print(f"\n--- Entropy Run {i + 1}/{args.runs} ---")
            result = run_entropy_benchmark(
                model_path=args.model_path,
                num_prompts=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                seed=args.seed,
                skip_warmup=args.skip_warmup,
            )
            entropy_results.append(result)
            print(f"  Output throughput: {result.output_throughput:.1f} tok/s")
            print(f"  Total latency:     {result.total_latency:.2f} s")

    # Aggregate and report
    baseline_agg = AggregatedResult.from_results(baseline_results) if baseline_results else None
    entropy_agg = AggregatedResult.from_results(entropy_results) if entropy_results else None

    if baseline_agg and entropy_agg:
        print_comparison_table(baseline_agg, entropy_agg)
    elif baseline_agg:
        print("\n[Baseline Only Results]")
        out_str = f"{baseline_agg.output_throughput_mean:.1f} +/- {baseline_agg.output_throughput_std:.1f}"
        lat_str = f"{baseline_agg.total_latency_mean:.2f} +/- {baseline_agg.total_latency_std:.2f}"
        print(f"  Output throughput: {out_str} tok/s")
        print(f"  Total latency:     {lat_str} s")
    elif entropy_agg:
        print("\n[Entropy Only Results]")
        out_str = f"{entropy_agg.output_throughput_mean:.1f} +/- {entropy_agg.output_throughput_std:.1f}"
        lat_str = f"{entropy_agg.total_latency_mean:.2f} +/- {entropy_agg.total_latency_std:.2f}"
        print(f"  Output throughput: {out_str} tok/s")
        print(f"  Total latency:     {lat_str} s")

    # Save JSON results
    if args.output_json:
        output = {
            "config": {
                "model_path": args.model_path,
                "num_prompts": args.num_prompts,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "seed": args.seed,
                "runs": args.runs,
            },
        }
        if baseline_agg:
            output["baseline"] = dataclasses.asdict(baseline_agg)
        if entropy_agg:
            output["entropy"] = dataclasses.asdict(entropy_agg)
        if baseline_agg and entropy_agg:
            output["delta"] = {
                "throughput_pct": (entropy_agg.output_throughput_mean - baseline_agg.output_throughput_mean)
                / baseline_agg.output_throughput_mean
                * 100,
                "latency_pct": (entropy_agg.total_latency_mean - baseline_agg.total_latency_mean)
                / baseline_agg.total_latency_mean
                * 100,
            }

        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[i] Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()

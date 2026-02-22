#!/usr/bin/env python3
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

"""Analyze per-token entropy distribution from entropy generation trajectories.

Reproduces the analysis in Section 3 / Figure 2(a) of:
  "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
   Reinforcement Learning for LLM Reasoning" (Wang et al., 2025)

Usage:
    python -m treetune_recipe.entropy_gen.analyze_entropy \
        --trajectories path/to/trajectories.pkl [path2.pkl ...] \
        --output-dir analysis_output/

    # With token-level stats (needs tokenizer):
    python -m treetune_recipe.entropy_gen.analyze_entropy \
        --trajectories path/to/trajectories.pkl \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --output-dir analysis_output/ \
        --min-token-freq 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_PALETTE = {
    "hist": "#4C72B0",
    "percentile": "#C44E52",
    "median": "#DD8452",
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
    "text": "#333333",
}


def _apply_style():
    """Publication-quality matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": _PALETTE["bg"],
            "axes.edgecolor": "#CCCCCC",
            "axes.labelcolor": _PALETTE["text"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": _PALETTE["grid"],
            "grid.alpha": 0.6,
            "grid.linewidth": 0.5,
            "xtick.color": _PALETTE["text"],
            "ytick.color": _PALETTE["text"],
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_entropies(trajectory_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate valid (masked) entropies and token ids.

    Returns:
        (all_entropies, all_token_ids) — 1-D arrays of active tokens only.
    """
    from verl.protocol import DataProto

    all_entropies = []
    all_token_ids = []
    total_trajectories = 0

    for path in trajectory_paths:
        print(f"  Loading {path} ...")
        merged = DataProto.load_from_disk(path)
        n = len(merged)
        total_trajectories += n

        entropy = merged.batch["rollout_entropy"]  # [B, response_len]
        mask = merged.batch["response_mask"]  # [B, response_len]
        responses = merged.batch["responses"]  # [B, response_len]

        valid = mask.bool()
        all_entropies.append(entropy[valid].numpy())
        all_token_ids.append(responses[valid].numpy())

        print(f"    {n} trajectories, {valid.sum().item():,} active tokens")

    all_entropies = np.concatenate(all_entropies)
    all_token_ids = np.concatenate(all_token_ids)
    print(f"  Total: {total_trajectories} trajectories, {len(all_entropies):,} active tokens")
    return all_entropies, all_token_ids


# ---------------------------------------------------------------------------
# Percentile analysis
# ---------------------------------------------------------------------------

_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


def compute_percentiles(entropies: np.ndarray) -> dict:
    """Compute summary statistics and percentiles."""
    vals = np.percentile(entropies, _PERCENTILES)
    stats = {
        "count": int(len(entropies)),
        "mean": float(np.mean(entropies)),
        "std": float(np.std(entropies)),
        "min": float(np.min(entropies)),
        "max": float(np.max(entropies)),
        "percentiles": {str(p): float(v) for p, v in zip(_PERCENTILES, vals, strict=False)},
        "frac_below_0.01": float((entropies < 0.01).mean()),
        "frac_below_0.1": float((entropies < 0.1).mean()),
        "frac_above_80th": 0.2,  # by definition
        "80th_threshold": float(vals[_PERCENTILES.index(80)]),
    }
    return stats


# ---------------------------------------------------------------------------
# Entropy distribution plot (Figure 2a)
# ---------------------------------------------------------------------------


def plot_entropy_distribution(
    entropies: np.ndarray,
    stats: dict,
    output_path: Path,
    num_bins: int = 200,
):
    """Histogram of token entropies with log-scale y-axis and percentile markers."""
    _apply_style()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Filter out exact zeros (padding artifacts)
    nonzero = entropies[entropies > 1e-8]

    ax.hist(
        nonzero,
        bins=num_bins,
        color=_PALETTE["hist"],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
        log=True,
    )

    # Percentile lines
    p80 = stats["percentiles"]["80"]
    ymin, ymax = ax.get_ylim()

    ax.axvline(
        p80,
        color=_PALETTE["percentile"],
        linestyle="--",
        linewidth=1.5,
        label=f"80th pctl = {p80:.3f}",
    )

    # Shaded region for top-20% forking tokens
    ax.fill_betweenx(
        [ymin, ymax],
        p80,
        nonzero.max() + 0.1,
        alpha=0.06,
        color=_PALETTE["percentile"],
    )
    ax.text(
        p80 + (nonzero.max() - p80) * 0.4,
        ymax * 0.3,
        "top 20%\n(forking tokens)",
        ha="center",
        va="center",
        fontsize=10,
        color=_PALETTE["percentile"],
        fontstyle="italic",
    )

    ax.set_xlabel("Token Entropy (nats)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Distribution of Per-Token Generation Entropy")
    ax.legend(loc="center right")

    # Summary text box — top-left to avoid legend overlap
    box_text = (
        f"N = {stats['count']:,} tokens\n"
        f"mean = {stats['mean']:.3f},  std = {stats['std']:.3f}\n"
        f"{stats['frac_below_0.01']:.1%} below 0.01\n"
        f"{stats['frac_below_0.1']:.1%} below 0.1"
    )
    ax.text(
        0.98,
        0.98,
        box_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.9,
        ),
    )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Token-level entropy stats
# ---------------------------------------------------------------------------


def compute_token_entropy_stats(
    entropies: np.ndarray,
    token_ids: np.ndarray,
    tokenizer,
    min_freq: int = 5,
    top_k: int = 100,
) -> tuple[list[dict], list[dict]]:
    """Per-token average entropy. Returns (top_high, top_low) lists."""
    unique_ids, inverse, counts = np.unique(token_ids, return_inverse=True, return_counts=True)

    # Sum entropies per unique token
    sum_entropy = np.zeros(len(unique_ids), dtype=np.float64)
    np.add.at(sum_entropy, inverse, entropies)
    avg_entropy = sum_entropy / counts

    # Filter by frequency
    freq_mask = counts >= min_freq
    filtered_ids = unique_ids[freq_mask]
    filtered_avg = avg_entropy[freq_mask]
    filtered_counts = counts[freq_mask]

    if len(filtered_ids) == 0:
        print(f"  Warning: no tokens with frequency >= {min_freq}")
        return [], []

    # Sort by average entropy
    sorted_idx = np.argsort(filtered_avg)

    def _make_records(indices):
        records = []
        for idx in indices:
            tid = int(filtered_ids[idx])
            text = tokenizer.decode([tid])
            records.append(
                {
                    "token_id": tid,
                    "token_text": text,
                    "avg_entropy": float(filtered_avg[idx]),
                    "count": int(filtered_counts[idx]),
                }
            )
        return records

    top_low = _make_records(sorted_idx[:top_k])
    top_high = _make_records(sorted_idx[-top_k:][::-1])

    return top_high, top_low


def plot_token_entropy_bars(
    records: list[dict],
    title: str,
    output_path: Path,
    max_tokens: int = 40,
    color: str = _PALETTE["hist"],
):
    """Horizontal bar chart of per-token average entropy."""
    _apply_style()

    records = records[:max_tokens]
    labels = [f"{r['token_text']!r}  (n={r['count']})" for r in reversed(records)]
    values = [r["avg_entropy"] for r in reversed(records)]

    fig_height = max(4, len(records) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    bars = ax.barh(
        range(len(labels)),
        values,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Average Entropy (nats)")
    ax.set_title(title)

    # Value labels on bars
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
            color=_PALETTE["text"],
        )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-token entropy distribution from generation trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Path(s) to trajectories.pkl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="entropy_analysis",
        help="Directory for output plots and stats (default: entropy_analysis)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model path for tokenizer (enables per-token analysis)",
    )
    parser.add_argument(
        "--min-token-freq",
        type=int,
        default=5,
        help="Min token frequency for per-token stats (default: 5)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=200,
        help="Number of histogram bins (default: 200)",
    )
    parser.add_argument(
        "--top-k-tokens",
        type=int,
        default=100,
        help="Number of top/bottom tokens to report (default: 100)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print("Loading trajectories...")
    all_entropies, all_token_ids = load_entropies(args.trajectories)

    # --- Percentiles ---
    print("Computing statistics...")
    stats = compute_percentiles(all_entropies)

    stats_path = out_dir / "entropy_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {stats_path}")

    # Print summary
    print("\n--- Entropy Distribution Summary ---")
    print(f"  Tokens:          {stats['count']:,}")
    print(f"  Mean entropy:    {stats['mean']:.4f}")
    print(f"  Std entropy:     {stats['std']:.4f}")
    print(f"  Below 0.01:      {stats['frac_below_0.01']:.1%}")
    print(f"  Below 0.1:       {stats['frac_below_0.1']:.1%}")
    print(f"  80th percentile: {stats['percentiles']['80']:.4f}")
    for p in _PERCENTILES:
        print(f"  P{p:>2d}: {stats['percentiles'][str(p)]:.4f}")

    # --- Distribution plot ---
    print("\nPlotting entropy distribution...")
    plot_entropy_distribution(
        all_entropies,
        stats,
        out_dir / "entropy_distribution.png",
        num_bins=args.num_bins,
    )

    # --- Per-token analysis (optional, needs tokenizer) ---
    if args.model:
        print(f"\nLoading tokenizer from {args.model}...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        print("Computing per-token entropy stats...")
        top_high, top_low = compute_token_entropy_stats(
            all_entropies,
            all_token_ids,
            tokenizer,
            min_freq=args.min_token_freq,
            top_k=args.top_k_tokens,
        )

        # Save JSON
        for name, records in [
            ("top_tokens_high_entropy", top_high),
            ("top_tokens_low_entropy", top_low),
        ]:
            path = out_dir / f"{name}.json"
            with open(path, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {path}")

        # Print top-20
        if top_high:
            print("\n--- Top 20 Highest Avg Entropy Tokens ---")
            for r in top_high[:20]:
                print(f"  {r['token_text']!r:>20s}  avg_H={r['avg_entropy']:.4f}  n={r['count']}")

        if top_low:
            print("\n--- Top 20 Lowest Avg Entropy Tokens ---")
            for r in top_low[:20]:
                print(f"  {r['token_text']!r:>20s}  avg_H={r['avg_entropy']:.6f}  n={r['count']}")

        # Bar plots
        if top_high:
            plot_token_entropy_bars(
                top_high,
                "Tokens with Highest Average Entropy",
                out_dir / "top_tokens_high_entropy.png",
                color="#C44E52",
            )
        if top_low:
            plot_token_entropy_bars(
                top_low,
                "Tokens with Lowest Average Entropy",
                out_dir / "top_tokens_low_entropy.png",
                color="#4C72B0",
            )
    else:
        print("\n  Skipping per-token analysis (pass --model to enable)")

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

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

"""Generate word clouds of tokens with highest/lowest average entropy.

Reproduces Figure 2(b) and 2(c) of:
  "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
   Reinforcement Learning for LLM Reasoning" (Wang et al., 2025)

Usage:
    python -m treetune_recipe.entropy_gen.plot_word_cloud \
        --trajectories path/to/trajectories.pkl [path2.pkl ...] \
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


def _apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
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
    """Load and concatenate valid (masked) entropies and token ids."""
    from verl.protocol import DataProto

    all_entropies = []
    all_token_ids = []

    for path in trajectory_paths:
        print(f"  Loading {path} ...")
        merged = DataProto.load_from_disk(path)
        entropy = merged.batch["rollout_entropy"]
        mask = merged.batch["response_mask"]
        responses = merged.batch["responses"]
        valid = mask.bool()
        all_entropies.append(entropy[valid].numpy())
        all_token_ids.append(responses[valid].numpy())
        print(f"    {len(merged)} trajectories, {valid.sum().item():,} active tokens")

    return np.concatenate(all_entropies), np.concatenate(all_token_ids)


# ---------------------------------------------------------------------------
# Per-token entropy stats
# ---------------------------------------------------------------------------


def compute_token_entropy_stats(
    entropies: np.ndarray,
    token_ids: np.ndarray,
    tokenizer,
    min_freq: int,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    """Per-token average entropy. Returns (top_high, top_low) lists."""
    unique_ids, inverse, counts = np.unique(token_ids, return_inverse=True, return_counts=True)
    sum_entropy = np.zeros(len(unique_ids), dtype=np.float64)
    np.add.at(sum_entropy, inverse, entropies)
    avg_entropy = sum_entropy / counts

    freq_mask = counts >= min_freq
    filtered_ids = unique_ids[freq_mask]
    filtered_avg = avg_entropy[freq_mask]
    filtered_counts = counts[freq_mask]

    if len(filtered_ids) == 0:
        print(f"  Warning: no tokens with frequency >= {min_freq}")
        return [], []

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


# ---------------------------------------------------------------------------
# Word cloud generation
# ---------------------------------------------------------------------------


def _make_word_cloud(
    records: list[dict],
    title: str,
    output_path: Path,
    colormap: str,
    use_avg_entropy_as_weight: bool = True,
):
    """Render a word cloud from token records."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print(f"  wordcloud not installed, skipping {output_path.name}")
        print("  Install with: pip install wordcloud")
        return

    _apply_style()

    # Build frequencies dict
    # Deduplicate: same text from different token IDs → sum weights
    freq: dict[str, float] = {}
    for r in records:
        text = r["token_text"].strip()
        if not text or text in ("\n", "\t", "\r"):
            text = repr(r["token_text"])
        if use_avg_entropy_as_weight:
            weight = r["avg_entropy"]
        else:
            # For low-entropy cloud: use inverse rank so lowest = largest
            rank = records.index(r)
            weight = 1.0 / (rank + 1)
        freq[text] = freq.get(text, 0) + weight

    if not freq:
        print(f"  No valid tokens for {title}, skipping")
        return

    wc = WordCloud(
        width=1200,
        height=600,
        max_words=len(records),
        background_color="white",
        colormap=colormap,
        prefer_horizontal=0.7,
        relative_scaling=0.5,  # type: ignore[arg-type]
        min_font_size=8,
        max_font_size=120,
        margin=4,
    )
    wc.generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=14, pad=12, fontweight="600", color="#333")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate word clouds of tokens with highest/lowest average entropy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Path(s) to trajectories.pkl files",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model path for tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="entropy_analysis",
        help="Directory for output files (default: entropy_analysis)",
    )
    parser.add_argument(
        "--min-token-freq",
        type=int,
        default=5,
        help="Min token frequency for inclusion (default: 5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top/bottom tokens (default: 100)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading trajectories...")
    all_entropies, all_token_ids = load_entropies(args.trajectories)

    # Tokenizer
    print(f"Loading tokenizer from {args.model}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Compute stats
    print("Computing per-token entropy stats...")
    top_high, top_low = compute_token_entropy_stats(
        all_entropies,
        all_token_ids,
        tokenizer,
        args.min_token_freq,
        args.top_k,
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

    # Print summary
    if top_high:
        print("\n--- Top 15 Highest Avg Entropy Tokens ---")
        for r in top_high[:15]:
            print(f"  {r['token_text']!r:>20s}  avg_H={r['avg_entropy']:.4f}  n={r['count']}")
    if top_low:
        print("\n--- Top 15 Lowest Avg Entropy Tokens ---")
        for r in top_low[:15]:
            print(f"  {r['token_text']!r:>20s}  avg_H={r['avg_entropy']:.6f}  n={r['count']}")

    # Word clouds
    print("\nGenerating word clouds...")
    if top_high:
        _make_word_cloud(
            top_high,
            "Tokens with Highest Average Entropy (Forking Tokens)",
            out_dir / "word_cloud_high_entropy.png",
            colormap="Reds",
            use_avg_entropy_as_weight=True,
        )
    if top_low:
        _make_word_cloud(
            top_low,
            "Tokens with Lowest Average Entropy",
            out_dir / "word_cloud_low_entropy.png",
            colormap="Blues",
            use_avg_entropy_as_weight=False,
        )

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

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

"""Visualize per-token entropy along sequence position for individual trajectories.

Reproduces the style of Figures 12-17 (Appendix) of:
  "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
   Reinforcement Learning for LLM Reasoning" (Wang et al., 2025)

Each trajectory gets a line plot of entropy vs. token position, with the 80th
percentile threshold shown as a horizontal line and forking tokens highlighted.

Usage:
    # Plot first 5 trajectories
    python -m treetune_recipe.entropy_gen.plot_trajectory_entropy \
        --trajectories path/to/trajectories.pkl \
        --output-dir analysis_output/ \
        --num-trajectories 5

    # Specific trajectory indices
    python -m treetune_recipe.entropy_gen.plot_trajectory_entropy \
        --trajectories path/to/trajectories.pkl \
        --output-dir analysis_output/ \
        --indices 0 3 7

    # With token text annotations on peaks
    python -m treetune_recipe.entropy_gen.plot_trajectory_entropy \
        --trajectories path/to/trajectories.pkl \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --output-dir analysis_output/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_COLORS = {
    "entropy_line": "#4C72B0",
    "entropy_fill": "#4C72B0",
    "threshold": "#C44E52",
    "fork_dot": "#C44E52",
    "annotation": "#333333",
    "grid": "#E0E0E0",
    "bg": "#FAFAFA",
    "text": "#333333",
}


def _apply_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": _COLORS["bg"],
            "axes.edgecolor": "#CCCCCC",
            "axes.labelcolor": _COLORS["text"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": _COLORS["grid"],
            "grid.alpha": 0.5,
            "grid.linewidth": 0.4,
            "xtick.color": _COLORS["text"],
            "ytick.color": _COLORS["text"],
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trajectory_data(
    trajectory_paths: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and concatenate trajectories.

    Returns:
        (entropies, masks, responses) — all [N, response_len] tensors.
    """
    from verl.protocol import DataProto

    all_entropy = []
    all_mask = []
    all_responses = []

    for path in trajectory_paths:
        print(f"  Loading {path} ...")
        merged = DataProto.load_from_disk(path)
        all_entropy.append(merged.batch["rollout_entropy"])
        all_mask.append(merged.batch["response_mask"])
        all_responses.append(merged.batch["responses"])
        print(f"    {len(merged)} trajectories")

    return (
        torch.cat(all_entropy, dim=0),
        torch.cat(all_mask, dim=0),
        torch.cat(all_responses, dim=0),
    )


# ---------------------------------------------------------------------------
# Single trajectory plot
# ---------------------------------------------------------------------------


def plot_single_trajectory(
    entropy: np.ndarray,
    mask: np.ndarray,
    threshold_80: float,
    traj_idx: int,
    output_path: Path,
    tokenizer=None,
    response_ids: np.ndarray | None = None,
    annotate_top_n: int = 8,
    window_size: int = 20,
):
    """Plot entropy vs. position for one trajectory.

    Args:
        entropy: [seq_len] array of entropy values.
        mask: [seq_len] binary mask (1=active token).
        threshold_80: Global 80th percentile entropy threshold.
        traj_idx: Trajectory index (for title).
        output_path: Where to save the plot.
        tokenizer: Optional, for annotating peak tokens.
        response_ids: Optional [seq_len] token ids.
        annotate_top_n: Number of top-entropy tokens to annotate.
        window_size: Smoothing window for the trend line.
    """
    _apply_style()

    # Extract active region only
    active = mask.astype(bool)
    ent = entropy[active]
    positions = np.arange(len(ent))

    if len(ent) == 0:
        print(f"  Trajectory {traj_idx}: no active tokens, skipping")
        return

    # Smoothed trend
    if len(ent) > window_size:
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(ent, kernel, mode="same")
    else:
        smoothed = ent

    fig, ax = plt.subplots(figsize=(16, 4.5))

    # Raw entropy as thin line + translucent fill
    ax.fill_between(
        positions,
        0,
        ent,
        alpha=0.15,
        color=_COLORS["entropy_fill"],
    )
    ax.plot(
        positions,
        ent,
        color=_COLORS["entropy_line"],
        alpha=0.4,
        linewidth=0.4,
        rasterized=True,
    )

    # Smoothed trend
    ax.plot(
        positions,
        smoothed,
        color=_COLORS["entropy_line"],
        linewidth=1.5,
        alpha=0.9,
        label=f"Smoothed (w={window_size})",
    )

    # 80th percentile threshold
    ax.axhline(
        threshold_80,
        color=_COLORS["threshold"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"80th pctl = {threshold_80:.3f}",
    )

    # Highlight forking tokens (above threshold)
    forking = ent >= threshold_80
    if forking.any():
        ax.scatter(
            positions[forking],
            ent[forking],
            s=6,
            color=_COLORS["fork_dot"],
            alpha=0.5,
            zorder=3,
            label=f"Forking tokens ({forking.sum()}/{len(ent)})",
        )

    # Annotate top-N peak tokens
    if tokenizer is not None and response_ids is not None and annotate_top_n > 0:
        active_ids = response_ids[active]
        top_indices = np.argsort(ent)[-annotate_top_n:][::-1]

        # Deduplicate annotations that are too close
        annotated_positions = []
        min_gap = len(ent) * 0.03  # min 3% of sequence apart

        for idx in top_indices:
            if any(abs(idx - p) < min_gap for p in annotated_positions):
                continue
            annotated_positions.append(idx)

            token_text = tokenizer.decode([int(active_ids[idx])])
            token_text = repr(token_text) if len(token_text.strip()) < 2 else token_text.strip()
            if len(token_text) > 15:
                token_text = token_text[:12] + "..."

            ax.annotate(
                token_text,
                xy=(float(idx), float(ent[idx])),
                xytext=(0, 12),
                textcoords="offset points",
                fontsize=7.5,
                fontfamily="monospace",
                color=_COLORS["annotation"],
                ha="center",
                va="bottom",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#999999",
                    linewidth=0.5,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="#DDDDDD",
                    alpha=0.85,
                ),
            )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title(f"Per-Token Entropy — Trajectory {traj_idx}  ({len(ent):,} tokens)")
    ax.set_xlim(0, len(ent))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Grid overview plot
# ---------------------------------------------------------------------------


def plot_trajectory_grid(
    all_entropy: torch.Tensor,
    all_mask: torch.Tensor,
    indices: list[int],
    threshold_80: float,
    output_path: Path,
):
    """Compact multi-trajectory grid: one row per trajectory, entropy as heatmap."""
    _apply_style()

    n = len(indices)
    fig, axes = plt.subplots(n, 1, figsize=(16, 1.5 * n + 1.0), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, traj_idx in zip(axes, indices, strict=False):
        mask = all_mask[traj_idx].numpy().astype(bool)
        ent = all_entropy[traj_idx].numpy()
        ent_active = ent[mask]
        seq_len = len(ent_active)

        if seq_len == 0:
            ax.text(0.5, 0.5, "empty", transform=ax.transAxes, ha="center")
            continue

        # Show as an image (1 pixel tall, seq_len wide)
        img = ent_active.reshape(1, -1)
        ax.imshow(
            img,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            vmin=0,
            vmax=max(threshold_80 * 2, ent_active.max()),
        )
        ax.set_ylabel(f"#{traj_idx}", rotation=0, labelpad=30, fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(0, seq_len)

        # Tick every ~1000 tokens
        step = max(1, seq_len // 6)
        ax.set_xticks(range(0, seq_len, step))

    axes[-1].set_xlabel("Token Position")
    fig.suptitle(
        f"Entropy Heatmap Overview (80th pctl = {threshold_80:.3f})",
        fontsize=13,
        y=1.01,
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
        description="Plot per-token entropy for individual trajectories.",
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
        help="Directory for output plots (default: entropy_analysis)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model path for tokenizer (enables token annotations on peaks)",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=5,
        help="Number of trajectories to plot (default: 5, ignored if --indices set)",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Specific trajectory indices to plot",
    )
    parser.add_argument(
        "--annotate-top-n",
        type=int,
        default=8,
        help="Number of peak tokens to annotate per trajectory (default: 8)",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=20,
        help="Moving average window for trend line (default: 20)",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Skip the grid overview plot",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print("Loading trajectories...")
    all_entropy, all_mask, all_responses = load_trajectory_data(args.trajectories)
    total = all_entropy.shape[0]
    print(f"  Total trajectories available: {total}")

    # --- Select indices ---
    if args.indices is not None:
        indices = [i for i in args.indices if i < total]
        if len(indices) < len(args.indices):
            print(f"  Warning: some indices out of range (max={total - 1}), using {indices}")
    else:
        indices = list(range(min(args.num_trajectories, total)))

    if not indices:
        print("No valid trajectory indices. Exiting.")
        return

    # --- Compute global 80th percentile threshold ---
    all_valid = all_mask.bool()
    all_ent_flat = all_entropy[all_valid].numpy()
    threshold_80 = float(np.percentile(all_ent_flat, 80))
    print(f"  Global 80th percentile entropy: {threshold_80:.4f}")

    # --- Tokenizer ---
    tokenizer = None
    if args.model:
        print(f"Loading tokenizer from {args.model}...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- Individual trajectory plots ---
    print(f"\nPlotting {len(indices)} trajectories...")
    for traj_idx in indices:
        plot_single_trajectory(
            entropy=all_entropy[traj_idx].numpy(),
            mask=all_mask[traj_idx].numpy(),
            threshold_80=threshold_80,
            traj_idx=traj_idx,
            output_path=out_dir / f"trajectory_{traj_idx:04d}.png",
            tokenizer=tokenizer,
            response_ids=all_responses[traj_idx].numpy() if tokenizer else None,
            annotate_top_n=args.annotate_top_n,
            window_size=args.smoothing_window,
        )

    # --- Grid overview ---
    if not args.no_grid and len(indices) > 1:
        print("\nPlotting grid overview...")
        plot_trajectory_grid(
            all_entropy,
            all_mask,
            indices,
            threshold_80,
            out_dir / "trajectory_grid.png",
        )

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()

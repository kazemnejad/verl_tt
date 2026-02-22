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

"""Render CoT tokens as colored text, with background color = entropy.

Reproduces Figure 12 of:
  "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
   Reinforcement Learning for LLM Reasoning" (Wang et al., 2025)

Each token is shown inline as readable text with background color on a
blue (low entropy) → white → red (high entropy) diverging scale.

Outputs self-contained HTML files (one per trajectory).

Usage:
    python -m treetune_recipe.entropy_gen.plot_token_entropy_text \
        --trajectories path/to/trajectories.pkl \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --output-dir analysis_output/ \
        --num-trajectories 3

    # Specific indices
    python -m treetune_recipe.entropy_gen.plot_token_entropy_text \
        --trajectories path/to/trajectories.pkl \
        --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
        --output-dir analysis_output/ \
        --indices 0 5 12
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Color mapping: blue → white → red (diverging)
# ---------------------------------------------------------------------------


def _entropy_to_rgb(entropy: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    """Map entropy value to RGB using a blue→white→red diverging scale.

    vmin maps to blue (low entropy), vmax maps to red (high entropy),
    midpoint maps to white.
    """
    if vmax <= vmin:
        return (255, 255, 255)

    t = (entropy - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))

    # Blue (0,0) → White (0.5) → Red (1.0)
    if t <= 0.5:
        # Blue to white
        s = t / 0.5  # 0→1
        r = int(59 + s * (255 - 59))
        g = int(76 + s * (255 - 76))
        b = int(192 + s * (255 - 192))
    else:
        # White to red
        s = (t - 0.5) / 0.5  # 0→1
        r = int(255 - s * (255 - 197))
        g = int(255 - s * (255 - 58))
        b = int(255 - s * (255 - 50))

    return (r, g, b)


def _text_color_for_bg(r: int, g: int, b: int) -> str:
    """Return black or white text depending on background luminance."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111" if lum > 140 else "#fff"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Token Entropy — Trajectory {traj_idx}</title>
<style>
  body {{
    font-family: 'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace;
    font-size: 13px;
    line-height: 1.9;
    background: #fff;
    color: #222;
    max-width: 1200px;
    margin: 20px auto;
    padding: 0 24px;
  }}
  h2 {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-weight: 600;
    color: #333;
    margin-bottom: 4px;
  }}
  .meta {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 13px;
    color: #777;
    margin-bottom: 18px;
  }}
  .tokens {{
    word-wrap: break-word;
    overflow-wrap: break-word;
  }}
  .tok {{
    display: inline;
    padding: 1px 0px;
    border-radius: 2px;
    white-space: pre-wrap;
  }}
  .colorbar {{
    margin-top: 28px;
    padding-top: 14px;
    border-top: 1px solid #e0e0e0;
  }}
  .colorbar-title {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 12px;
    color: #666;
    margin-bottom: 6px;
  }}
  .colorbar-gradient {{
    height: 18px;
    border-radius: 3px;
    border: 1px solid #ddd;
  }}
  .colorbar-labels {{
    display: flex;
    justify-content: space-between;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 11px;
    color: #888;
    margin-top: 3px;
  }}
</style>
</head>
<body>
<h2>Token Entropy Visualization — Trajectory {traj_idx}</h2>
<div class="meta">{meta_text}</div>
<div class="tokens">{token_spans}</div>
<div class="colorbar">
  <div class="colorbar-title">Token Entropy (nats)</div>
  <div class="colorbar-gradient" style="background: linear-gradient(
    to right, rgb(59,76,192), rgb(255,255,255) 50%, rgb(197,58,50));"></div>
  <div class="colorbar-labels">
    <span>{vmin:.3f}</span>
    <span>{vmid:.3f}</span>
    <span>{vmax:.3f}</span>
  </div>
</div>
</body>
</html>
"""


def render_trajectory_html(
    token_texts: list[str],
    entropies: list[float],
    traj_idx: int,
    vmin: float,
    vmax: float,
    meta_text: str = "",
) -> str:
    """Render a single trajectory as an HTML string."""
    spans = []
    for text, ent in zip(token_texts, entropies, strict=False):
        r, g, b = _entropy_to_rgb(ent, vmin, vmax)
        fg = _text_color_for_bg(r, g, b)
        escaped = html.escape(text)
        # Preserve newlines as <br>
        escaped = escaped.replace("\n", "<br>")
        spans.append(
            f'<span class="tok" style="background:rgb({r},{g},{b});color:{fg}" title="H={ent:.4f}">{escaped}</span>'
        )

    return _HTML_TEMPLATE.format(
        traj_idx=traj_idx,
        meta_text=html.escape(meta_text),
        token_spans="".join(spans),
        vmin=vmin,
        vmid=(vmin + vmax) / 2,
        vmax=vmax,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trajectory_data(
    trajectory_paths: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and concatenate trajectories."""
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Render CoT tokens as colored text with entropy background colors.",
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
        help="Directory for output HTML files (default: entropy_analysis)",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
        help="Number of trajectories to render (default: 3, ignored if --indices set)",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="Specific trajectory indices to render",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Max entropy for color scale (default: 95th percentile of all active tokens)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print("Loading trajectories...")
    all_entropy, all_mask, all_responses = load_trajectory_data(args.trajectories)
    total = all_entropy.shape[0]
    print(f"  Total trajectories: {total}")

    # --- Tokenizer ---
    print(f"Loading tokenizer from {args.model}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- Select indices ---
    if args.indices is not None:
        indices = [i for i in args.indices if i < total]
    else:
        indices = list(range(min(args.num_trajectories, total)))

    if not indices:
        print("No valid trajectory indices. Exiting.")
        return

    # --- Global color scale ---
    all_valid = all_mask.bool()
    all_ent_flat = all_entropy[all_valid].numpy()
    vmin = 0.0
    if args.vmax is not None:
        vmax = args.vmax
    else:
        vmax = float(np.percentile(all_ent_flat, 95))
    p80 = float(np.percentile(all_ent_flat, 80))
    print(f"  Color scale: [{vmin:.3f}, {vmax:.3f}]  (80th pctl = {p80:.3f})")

    # --- Render ---
    print(f"\nRendering {len(indices)} trajectories...")
    for traj_idx in indices:
        mask = all_mask[traj_idx].numpy().astype(bool)
        ent = all_entropy[traj_idx].numpy()
        resp = all_responses[traj_idx].numpy()

        # Active tokens only
        active_ent = ent[mask]
        active_ids = resp[mask]

        if len(active_ids) == 0:
            print(f"  Trajectory {traj_idx}: no active tokens, skipping")
            continue

        # Decode each token individually
        token_texts = [tokenizer.decode([int(tid)]) for tid in active_ids]
        entropies = [float(e) for e in active_ent]

        n_tokens = len(token_texts)
        n_forking = sum(1 for e in entropies if e >= p80)
        mean_ent = np.mean(active_ent)
        meta_text = (
            f"{n_tokens:,} tokens | "
            f"mean entropy = {mean_ent:.3f} | "
            f"{n_forking:,} forking tokens ({n_forking / n_tokens:.1%} above 80th pctl)"
        )

        html_content = render_trajectory_html(token_texts, entropies, traj_idx, vmin, vmax, meta_text)

        out_path = out_dir / f"token_entropy_{traj_idx:04d}.html"
        out_path.write_text(html_content, encoding="utf-8")
        print(f"  Saved: {out_path}  ({n_tokens:,} tokens)")

    print(f"\nDone. Open HTML files in a browser to view. Results in: {out_dir}")


if __name__ == "__main__":
    main()

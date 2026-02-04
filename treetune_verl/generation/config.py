# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""GenerationConfig - Configuration dataclass for generation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationConfig:
    """Configuration for GenerationRunner.

    Attributes:
        output_dir: Directory to save trajectories and checkpoints.
        save_batch_size: Number of samples per batch file.
        pull_timeout: Seconds to wait for batch from queue.
        final_merge: Whether to merge batches into single file at end.
        checkpoint_interval: Save checkpoint every N batches.
        wandb_upload: Whether to upload trajectories as WandB artifact.
        wandb_project: WandB project name (required if wandb_upload and no active run).
        wandb_run_name: WandB run name (auto-generated if None).
        show_progress: Whether to display tqdm progress bar.
    """

    output_dir: str
    save_batch_size: int = 1000
    pull_timeout: float = 30.0
    final_merge: bool = True
    checkpoint_interval: int = 1
    wandb_upload: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    show_progress: bool = True

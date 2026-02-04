# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""CheckpointManager - Manages checkpoint state for generation runner."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint state for resumable generation.

    Tracks completed sample indices, saved batch files, and config snapshot.
    Enables resume from interruption by filtering already-completed samples.

    Attributes:
        output_dir: Directory containing checkpoint and batch files.
        total_samples: Total number of samples in dataset.
        config_snapshot: Generation config for validation on resume.
        completed_indices: Set of sample indices that have been saved.
        saved_batches: List of saved batch file names.
    """

    CHECKPOINT_FILE = "checkpoint.json"

    def __init__(
        self,
        output_dir: str,
        total_samples: int,
        config_snapshot: dict[str, Any],
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoint and batch files.
            total_samples: Total number of samples in dataset.
            config_snapshot: Generation config snapshot for validation.
        """
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.config_snapshot = config_snapshot
        self.completed_indices: set[int] = set()
        self.saved_batches: list[str] = []

    def load(self) -> bool:
        """Load checkpoint from file if it exists.

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists.
        """
        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            return False

        with open(checkpoint_path) as f:
            data = json.load(f)

        self.completed_indices = set(data.get("completed_indices", []))
        self.saved_batches = data.get("saved_batches", [])
        self.total_samples = data.get("total_samples", self.total_samples)

        logger.info(f"Loaded checkpoint: {len(self.completed_indices)} completed, {len(self.saved_batches)} batches")
        return True

    def save(self) -> None:
        """Save checkpoint to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = self.output_dir / self.CHECKPOINT_FILE
        data = {
            "completed_indices": sorted(self.completed_indices),
            "saved_batches": self.saved_batches,
            "total_samples": self.total_samples,
            "config_snapshot": self.config_snapshot,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_completed(self, indices: list[int]) -> None:
        """Add completed sample indices.

        Args:
            indices: List of sample indices that have been saved.
        """
        self.completed_indices.update(indices)

    def add_batch(self, batch_name: str) -> None:
        """Add a saved batch file name.

        Args:
            batch_name: Name of the batch file (without extension).
        """
        self.saved_batches.append(batch_name)

    def get_pending_indices(self, all_indices: list[int]) -> list[int]:
        """Get indices that have not been completed.

        Args:
            all_indices: List of all sample indices.

        Returns:
            List of indices not in completed_indices.
        """
        return [i for i in all_indices if i not in self.completed_indices]

    def validate_batches(self) -> tuple[list[str], list[str]]:
        """Validate that all saved batch files exist and are readable.

        Returns:
            Tuple of (valid_batches, invalid_batches).
        """
        valid = []
        invalid = []

        for batch_name in self.saved_batches:
            batch_path = self.output_dir / f"{batch_name}.pkl"
            if not batch_path.exists():
                invalid.append(batch_name)
                continue

            try:
                with open(batch_path, "rb") as f:
                    pickle.load(f)
                valid.append(batch_name)
            except Exception as e:
                logger.warning(f"Invalid batch file {batch_name}: {e}")
                invalid.append(batch_name)

        return valid, invalid

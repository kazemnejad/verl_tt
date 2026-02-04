# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Tests for CheckpointManager."""

import pickle


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoint."""
        from treetune_verl.generation.checkpoint import CheckpointManager

        config_snapshot = {"model": "test-model", "batch_size": 100}
        manager = CheckpointManager(
            output_dir=str(tmp_path),
            total_samples=1000,
            config_snapshot=config_snapshot,
        )

        # Add some completed indices
        manager.add_completed([0, 1, 2, 3, 4])
        manager.add_batch("batch_0000")
        manager.save()

        # Load in new manager
        manager2 = CheckpointManager(
            output_dir=str(tmp_path),
            total_samples=1000,
            config_snapshot=config_snapshot,
        )
        manager2.load()

        assert manager2.completed_indices == {0, 1, 2, 3, 4}
        assert manager2.saved_batches == ["batch_0000"]
        assert manager2.total_samples == 1000

    def test_checkpoint_tracks_completed_indices(self, tmp_path):
        """Test that completed indices are tracked correctly."""
        from treetune_verl.generation.checkpoint import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path), total_samples=100, config_snapshot={})

        manager.add_completed([0, 1, 2])
        assert manager.completed_indices == {0, 1, 2}

        manager.add_completed([3, 4, 5])
        assert manager.completed_indices == {0, 1, 2, 3, 4, 5}

        # Adding duplicates should not change set
        manager.add_completed([0, 1])
        assert len(manager.completed_indices) == 6

    def test_checkpoint_tracks_saved_batches(self, tmp_path):
        """Test that saved batch names are tracked."""
        from treetune_verl.generation.checkpoint import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path), total_samples=100, config_snapshot={})

        manager.add_batch("batch_0000")
        manager.add_batch("batch_0001")
        manager.add_batch("batch_0002")

        assert manager.saved_batches == ["batch_0000", "batch_0001", "batch_0002"]

    def test_validate_batch_files(self, tmp_path):
        """Test validation of batch files."""
        from treetune_verl.generation.checkpoint import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path), total_samples=100, config_snapshot={})

        # Create valid batch file
        valid_batch = {"data": [1, 2, 3]}
        with open(tmp_path / "batch_0000.pkl", "wb") as f:
            pickle.dump(valid_batch, f)
        manager.add_batch("batch_0000")

        # Validation should pass
        valid, invalid = manager.validate_batches()
        assert valid == ["batch_0000"]
        assert invalid == []

        # Add non-existent batch
        manager.add_batch("batch_0001")
        valid, invalid = manager.validate_batches()
        assert valid == ["batch_0000"]
        assert invalid == ["batch_0001"]

    def test_resume_filters_completed(self, tmp_path):
        """Test that get_pending_indices filters out completed indices."""
        from treetune_verl.generation.checkpoint import CheckpointManager

        manager = CheckpointManager(output_dir=str(tmp_path), total_samples=100, config_snapshot={})

        # Mark some as completed
        manager.add_completed([0, 2, 4, 6, 8])

        # Get pending from a range
        all_indices = list(range(10))
        pending = manager.get_pending_indices(all_indices)

        assert pending == [1, 3, 5, 7, 9]

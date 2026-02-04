# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Tests for GenerationRunner."""

import pickle
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


class TestGenerationRunner:
    """Tests for GenerationRunner."""

    def test_runner_adapts_config_for_manager(self, tmp_path):
        """Test config adapter produces expected structure."""
        from treetune_verl.generation.runner import GenerationRunner

        flat_config = OmegaConf.create(
            {
                "n_gpus_per_node": 8,
                "nnodes": 1,
                "model": {"path": "test/model", "dtype": "bfloat16"},
                "rollout": {
                    "name": "sglang",
                    "tensor_model_parallel_size": 1,
                    "temperature": 0.7,
                },
                "data": {"files": ["/tmp/test.parquet"], "prompt_key": "prompt"},
                "generation": {
                    "output_dir": str(tmp_path),
                    "save_batch_size": 100,
                    "pull_timeout": 10.0,
                    "final_merge": True,
                    "checkpoint_interval": 1,
                    "wandb_upload": False,
                    "wandb_project": None,
                    "wandb_run_name": None,
                    "show_progress": False,
                },
                "tasks": None,
            }
        )

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        assert adapted.trainer.n_gpus_per_node == 8
        assert adapted.trainer.nnodes == 1
        assert adapted.actor_rollout_ref.model.path == "test/model"
        assert adapted.actor_rollout_ref.rollout.name == "sglang"
        assert adapted.reward_model.enable is False

    def test_runner_saves_batch_as_dataproto(self, tmp_path):
        """Test that batches are saved as pickle files."""
        from treetune_verl.generation.runner import GenerationRunner

        # Mock DataProto
        mock_dataproto = MagicMock()
        mock_dataproto.to_dict.return_value = {"batch": {"data": [1, 2, 3]}}

        # Create minimal config
        config = OmegaConf.create(
            {
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "model": {"path": "test/model"},
                "rollout": {"name": "sglang", "tensor_model_parallel_size": 1},
                "data": {"files": [], "prompt_key": "prompt"},
                "generation": {
                    "output_dir": str(tmp_path),
                    "save_batch_size": 100,
                    "pull_timeout": 10.0,
                    "final_merge": True,
                    "checkpoint_interval": 1,
                    "wandb_upload": False,
                    "wandb_project": None,
                    "wandb_run_name": None,
                    "show_progress": False,
                },
                "tasks": None,
            }
        )

        # Create runner without initializing manager
        with patch.object(GenerationRunner, "_initialize"):
            runner = GenerationRunner(config)
            runner.output_dir = tmp_path
            runner.checkpoint_manager = MagicMock()

        # Save a batch
        batch_items = [(0, {"data": "item0"}), (1, {"data": "item1"})]
        batch_name = runner._save_batch(batch_items, batch_idx=0)

        # Verify file was created
        batch_path = tmp_path / f"{batch_name}.pkl"
        assert batch_path.exists()

        # Verify content
        with open(batch_path, "rb") as f:
            saved_data = pickle.load(f)
        assert len(saved_data) == 2

    def test_runner_updates_checkpoint_after_batch(self, tmp_path):
        """Test checkpoint is updated after saving batch."""
        from treetune_verl.generation.runner import GenerationRunner

        config = OmegaConf.create(
            {
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "model": {"path": "test/model"},
                "rollout": {"name": "sglang", "tensor_model_parallel_size": 1},
                "data": {"files": [], "prompt_key": "prompt"},
                "generation": {
                    "output_dir": str(tmp_path),
                    "save_batch_size": 100,
                    "pull_timeout": 10.0,
                    "final_merge": True,
                    "checkpoint_interval": 1,
                    "wandb_upload": False,
                    "wandb_project": None,
                    "wandb_run_name": None,
                    "show_progress": False,
                },
                "tasks": None,
            }
        )

        mock_checkpoint = MagicMock()

        with patch.object(GenerationRunner, "_initialize"):
            runner = GenerationRunner(config)
            runner.output_dir = tmp_path
            runner.checkpoint_manager = mock_checkpoint
            runner.gen_config = config.generation

        batch_items = [(5, {"data": "item5"}), (10, {"data": "item10"})]
        runner._save_batch(batch_items, batch_idx=0)

        # Verify checkpoint was updated
        mock_checkpoint.add_completed.assert_called_once_with([5, 10])
        mock_checkpoint.add_batch.assert_called_once()
        mock_checkpoint.save.assert_called_once()

    def test_runner_final_merge(self, tmp_path):
        """Test final merge combines batch files."""
        from treetune_verl.generation.runner import GenerationRunner

        config = OmegaConf.create(
            {
                "n_gpus_per_node": 1,
                "nnodes": 1,
                "model": {"path": "test/model"},
                "rollout": {"name": "sglang", "tensor_model_parallel_size": 1},
                "data": {"files": [], "prompt_key": "prompt"},
                "generation": {
                    "output_dir": str(tmp_path),
                    "save_batch_size": 100,
                    "pull_timeout": 10.0,
                    "final_merge": True,
                    "checkpoint_interval": 1,
                    "wandb_upload": False,
                    "wandb_project": None,
                    "wandb_run_name": None,
                    "show_progress": False,
                },
                "tasks": None,
            }
        )

        # Create mock batch files
        batch0 = [(0, {"data": "a"}), (1, {"data": "b"})]
        batch1 = [(2, {"data": "c"}), (3, {"data": "d"})]

        with open(tmp_path / "batch_0000.pkl", "wb") as f:
            pickle.dump(batch0, f)
        with open(tmp_path / "batch_0001.pkl", "wb") as f:
            pickle.dump(batch1, f)

        mock_checkpoint = MagicMock()
        mock_checkpoint.saved_batches = ["batch_0000", "batch_0001"]

        with patch.object(GenerationRunner, "_initialize"):
            runner = GenerationRunner(config)
            runner.output_dir = tmp_path
            runner.checkpoint_manager = mock_checkpoint
            runner.gen_config = config.generation

        runner._merge_batches()

        # Verify merged file exists
        merged_path = tmp_path / "trajectories.pkl"
        assert merged_path.exists()

        with open(merged_path, "rb") as f:
            merged = pickle.load(f)
        assert len(merged) == 4

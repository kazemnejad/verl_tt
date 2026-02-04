# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Tests for GenerationConfig dataclass."""

import pytest


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_config_defaults(self):
        """Test default values are set correctly."""
        from treetune_verl.generation.config import GenerationConfig

        config = GenerationConfig(output_dir="/tmp/test")

        assert config.output_dir == "/tmp/test"
        assert config.save_batch_size == 1000
        assert config.pull_timeout == 30.0
        assert config.final_merge is True
        assert config.checkpoint_interval == 1
        assert config.wandb_upload is False
        assert config.wandb_project is None
        assert config.wandb_run_name is None
        assert config.show_progress is True

    def test_config_from_dict(self):
        """Test creating config from dict."""
        from treetune_verl.generation.config import GenerationConfig

        config = GenerationConfig(
            output_dir="/path/to/output",
            save_batch_size=500,
            pull_timeout=60.0,
            final_merge=False,
            checkpoint_interval=2,
            wandb_upload=True,
            wandb_project="my-project",
            wandb_run_name="run-001",
            show_progress=False,
        )

        assert config.output_dir == "/path/to/output"
        assert config.save_batch_size == 500
        assert config.pull_timeout == 60.0
        assert config.final_merge is False
        assert config.checkpoint_interval == 2
        assert config.wandb_upload is True
        assert config.wandb_project == "my-project"
        assert config.wandb_run_name == "run-001"
        assert config.show_progress is False

    def test_config_required_output_dir(self):
        """Test that output_dir is required."""
        from treetune_verl.generation.config import GenerationConfig

        with pytest.raises(TypeError):
            GenerationConfig()  # type: ignore

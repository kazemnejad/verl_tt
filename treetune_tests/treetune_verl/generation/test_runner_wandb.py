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

"""Unit tests for GenerationRunner WandB upload functionality.

Tests the _upload_to_wandb() method added to GenerationRunner:
- Creates wandb.Artifact with correct type
- Adds trajectories.pkl if final_merge=True
- Adds batch files if final_merge=False
- Includes checkpoint.json
- Uses active run if available
- Creates new run if none active

TDD approach: tests written first, then implementation.

From spec:
- Artifact type: "trajectories"
- Artifact name: "trajectories-<run_name>"
- If WandB run already active: upload to current run
- If no active run: create new run using wandb_project / wandb_run_name
"""

import pickle
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from treetune_verl.generation.runner import GenerationRunner


@pytest.fixture
def runner_with_wandb_config(tmp_path):
    """Create a GenerationRunner with mocked state for WandB upload tests."""
    runner = object.__new__(GenerationRunner)
    runner.output_dir = tmp_path
    runner.completed_indices = {0, 1, 2}
    runner.saved_batches = ["batch_0000"]
    runner.total_samples = 3
    runner.config_snapshot = {"model": {"path": "/test/model"}}

    # Config with wandb settings
    runner.config = OmegaConf.create(
        {
            "generation": {
                "output_dir": str(tmp_path),
                "final_merge": True,
                "wandb_upload": True,
                "wandb_project": "test-project",
                "wandb_run_name": "test-run",
            }
        }
    )

    return runner


@pytest.fixture
def mock_wandb():
    """Create mock wandb module with all necessary components."""
    mock = MagicMock()

    # Mock Artifact class
    mock_artifact = MagicMock()
    mock.Artifact.return_value = mock_artifact

    # Mock run context
    mock_run = MagicMock()
    mock.run = None  # Default to no active run
    mock.init.return_value = mock_run

    return mock


class TestUploadCreatesArtifact:
    """Tests that _upload_to_wandb creates artifact with correct type."""

    def test_upload_creates_artifact(self, runner_with_wandb_config, mock_wandb, tmp_path):
        """Creates wandb.Artifact with type='trajectories' and correct name."""
        runner = runner_with_wandb_config

        # Create required files
        (tmp_path / "trajectories.pkl").write_bytes(pickle.dumps([]))
        (tmp_path / "checkpoint.json").write_text('{"completed_indices": [0,1,2]}')

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            runner._upload_to_wandb()

        # Verify Artifact was created with correct type and name
        mock_wandb.Artifact.assert_called_once()
        call_kwargs = mock_wandb.Artifact.call_args
        assert call_kwargs[1]["type"] == "trajectories"
        assert "trajectories-" in call_kwargs[1]["name"]


class TestUploadAddsMergedFile:
    """Tests that _upload_to_wandb adds trajectories.pkl when final_merge=True."""

    def test_upload_adds_merged_file(self, runner_with_wandb_config, mock_wandb, tmp_path):
        """Adds trajectories.pkl if final_merge=True."""
        runner = runner_with_wandb_config
        runner.config.generation.final_merge = True

        # Create merged file
        merged_path = tmp_path / "trajectories.pkl"
        merged_path.write_bytes(pickle.dumps([(0, {"data": "test"})]))
        (tmp_path / "checkpoint.json").write_text('{"completed_indices": [0]}')

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            runner._upload_to_wandb()

        # Verify add_file was called with trajectories.pkl
        artifact = mock_wandb.Artifact.return_value
        add_file_calls = [str(c) for c in artifact.add_file.call_args_list]
        assert any("trajectories.pkl" in str(c) for c in add_file_calls)


class TestUploadAddsBatchFiles:
    """Tests that _upload_to_wandb adds batch files when final_merge=False."""

    def test_upload_adds_batch_files(self, runner_with_wandb_config, mock_wandb, tmp_path):
        """Adds batch files if final_merge=False."""
        runner = runner_with_wandb_config
        runner.config.generation.final_merge = False
        runner.saved_batches = ["batch_0000", "batch_0001"]

        # Create batch files
        (tmp_path / "batch_0000.pkl").write_bytes(pickle.dumps([]))
        (tmp_path / "batch_0001.pkl").write_bytes(pickle.dumps([]))
        (tmp_path / "checkpoint.json").write_text('{"completed_indices": [0,1]}')

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            runner._upload_to_wandb()

        # Verify add_file was called for each batch file
        artifact = mock_wandb.Artifact.return_value
        add_file_calls = [str(c) for c in artifact.add_file.call_args_list]
        assert any("batch_0000.pkl" in str(c) for c in add_file_calls)
        assert any("batch_0001.pkl" in str(c) for c in add_file_calls)


class TestUploadAddsCheckpoint:
    """Tests that _upload_to_wandb always includes checkpoint.json."""

    def test_upload_adds_checkpoint(self, runner_with_wandb_config, mock_wandb, tmp_path):
        """Includes checkpoint.json in artifact."""
        runner = runner_with_wandb_config

        # Create required files
        (tmp_path / "trajectories.pkl").write_bytes(pickle.dumps([]))
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text('{"completed_indices": [0,1,2], "saved_batches": ["batch_0000"]}')

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            runner._upload_to_wandb()

        # Verify checkpoint.json was added
        artifact = mock_wandb.Artifact.return_value
        add_file_calls = [str(c) for c in artifact.add_file.call_args_list]
        assert any("checkpoint.json" in str(c) for c in add_file_calls)


class TestUploadUsesExistingRun:
    """Tests that _upload_to_wandb uses active run if available."""

    def test_upload_uses_existing_run(self, runner_with_wandb_config, mock_wandb, tmp_path):
        """Uses active run if available (does not call wandb.init)."""
        runner = runner_with_wandb_config

        # Setup active run
        mock_active_run = MagicMock()
        mock_wandb.run = mock_active_run

        # Create required files
        (tmp_path / "trajectories.pkl").write_bytes(pickle.dumps([]))
        (tmp_path / "checkpoint.json").write_text('{"completed_indices": [0]}')

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            runner._upload_to_wandb()

        # Verify init was NOT called (existing run used)
        mock_wandb.init.assert_not_called()

        # Verify artifact was logged to active run
        artifact = mock_wandb.Artifact.return_value
        mock_active_run.log_artifact.assert_called_once_with(artifact)


class TestUploadCreatesNewRun:
    """Tests that _upload_to_wandb creates run if none active."""

    def test_upload_creates_new_run(self, runner_with_wandb_config, mock_wandb, tmp_path):
        """Creates run if none active using wandb_project/wandb_run_name."""
        runner = runner_with_wandb_config

        # No active run
        mock_wandb.run = None
        mock_new_run = MagicMock()
        mock_wandb.init.return_value = mock_new_run

        # Create required files
        (tmp_path / "trajectories.pkl").write_bytes(pickle.dumps([]))
        (tmp_path / "checkpoint.json").write_text('{"completed_indices": [0]}')

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            runner._upload_to_wandb()

        # Verify init was called with correct project and run name
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["project"] == "test-project"
        assert call_kwargs["name"] == "test-run"

        # Verify artifact was logged to new run
        artifact = mock_wandb.Artifact.return_value
        mock_new_run.log_artifact.assert_called_once_with(artifact)

        # Verify run was finished
        mock_new_run.finish.assert_called_once()

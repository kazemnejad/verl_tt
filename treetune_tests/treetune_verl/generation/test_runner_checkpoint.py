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

"""Unit tests for GenerationRunner checkpoint logic.

Tests the checkpoint methods added to GenerationRunner:
- _save_checkpoint(): Write JSON to output_dir/checkpoint.json
- _load_checkpoint(): Load state if file exists
- _get_pending_indices(): Return indices not in completed_indices
- _validate_batches(): Check batch files exist and are readable

TDD approach: tests written first, then implementation.
"""

import json
import pickle

import pytest

from treetune_verl.generation.runner import GenerationRunner


@pytest.fixture
def runner_with_output_dir(tmp_path):
    """Create a GenerationRunner with mocked output_dir and state.

    We create a minimal runner instance for testing checkpoint methods.
    The runner needs:
    - output_dir: Path
    - completed_indices: set[int]
    - saved_batches: list[str]
    - total_samples: int
    - config_snapshot: dict
    """
    runner = object.__new__(GenerationRunner)
    runner.output_dir = tmp_path
    runner.completed_indices = set()
    runner.saved_batches = []
    runner.total_samples = 100
    runner.config_snapshot = {"model": {"path": "/test/model"}, "rollout": {"temperature": 0.8}}
    return runner


class TestSaveCheckpoint:
    """Tests for _save_checkpoint method."""

    def test_save_checkpoint_writes_json(self, runner_with_output_dir):
        """Writes checkpoint.json with correct structure.

        Expected JSON structure:
        {
            "completed_indices": [0, 1, 2, ...],
            "saved_batches": ["batch_0000", "batch_0001"],
            "total_samples": 10000,
            "config_snapshot": {...}
        }
        """
        runner = runner_with_output_dir
        runner.completed_indices = {0, 1, 2, 5, 10}
        runner.saved_batches = ["batch_0000", "batch_0001"]
        runner.total_samples = 1000

        # Save checkpoint
        runner._save_checkpoint()

        # Verify file exists
        checkpoint_path = runner.output_dir / "checkpoint.json"
        assert checkpoint_path.exists()

        # Verify structure
        with open(checkpoint_path) as f:
            data = json.load(f)

        assert "completed_indices" in data
        assert "saved_batches" in data
        assert "total_samples" in data
        assert "config_snapshot" in data

        # Verify values (completed_indices sorted for determinism)
        assert sorted(data["completed_indices"]) == [0, 1, 2, 5, 10]
        assert data["saved_batches"] == ["batch_0000", "batch_0001"]
        assert data["total_samples"] == 1000
        assert data["config_snapshot"] == runner.config_snapshot


class TestLoadCheckpoint:
    """Tests for _load_checkpoint method."""

    def test_load_checkpoint_restores_state(self, runner_with_output_dir):
        """Loads completed_indices, saved_batches from existing checkpoint."""
        runner = runner_with_output_dir

        # Create checkpoint file manually
        checkpoint_data = {
            "completed_indices": [0, 1, 2, 3, 4],
            "saved_batches": ["batch_0000"],
            "total_samples": 500,
            "config_snapshot": {"model": {"path": "/test/model"}},
        }
        checkpoint_path = runner.output_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        # Load checkpoint
        result = runner._load_checkpoint()

        # Verify state restored
        assert result is True
        assert runner.completed_indices == {0, 1, 2, 3, 4}
        assert runner.saved_batches == ["batch_0000"]
        # total_samples preserved from checkpoint (for validation)

    def test_load_checkpoint_missing_file(self, runner_with_output_dir):
        """Returns gracefully if no checkpoint file exists."""
        runner = runner_with_output_dir

        # Ensure no checkpoint file
        checkpoint_path = runner.output_dir / "checkpoint.json"
        assert not checkpoint_path.exists()

        # Load checkpoint - should return False (no checkpoint)
        result = runner._load_checkpoint()

        # Should return False but not raise
        assert result is False
        # State should remain at defaults
        assert runner.completed_indices == set()
        assert runner.saved_batches == []


class TestGetPendingIndices:
    """Tests for _get_pending_indices method."""

    def test_get_pending_indices_excludes_completed(self, runner_with_output_dir):
        """Filters out completed indices from total range."""
        runner = runner_with_output_dir
        runner.total_samples = 10
        runner.completed_indices = {0, 2, 4, 6, 8}

        # Get pending
        pending = runner._get_pending_indices()

        # Should exclude completed
        assert pending == [1, 3, 5, 7, 9]

    def test_get_pending_indices_empty_completed(self, runner_with_output_dir):
        """All indices pending when none completed."""
        runner = runner_with_output_dir
        runner.total_samples = 5
        runner.completed_indices = set()

        pending = runner._get_pending_indices()

        assert pending == [0, 1, 2, 3, 4]

    def test_get_pending_indices_all_completed(self, runner_with_output_dir):
        """Empty list when all completed."""
        runner = runner_with_output_dir
        runner.total_samples = 3
        runner.completed_indices = {0, 1, 2}

        pending = runner._get_pending_indices()

        assert pending == []


class TestCheckpointContainsConfigSnapshot:
    """Tests for config_snapshot in checkpoint."""

    def test_checkpoint_contains_config_snapshot(self, runner_with_output_dir):
        """Config snapshot saved for validation on resume."""
        runner = runner_with_output_dir
        runner.config_snapshot = {
            "model": {"path": "/test/model", "trust_remote_code": True},
            "rollout": {"temperature": 0.9, "top_p": 0.95},
            "generation": {"save_batch_size": 500},
        }

        runner._save_checkpoint()

        # Load and verify
        checkpoint_path = runner.output_dir / "checkpoint.json"
        with open(checkpoint_path) as f:
            data = json.load(f)

        assert data["config_snapshot"] == runner.config_snapshot


class TestValidateBatches:
    """Tests for _validate_batches method."""

    def test_validate_batches_detects_missing(self, runner_with_output_dir):
        """Identifies missing batch files."""
        runner = runner_with_output_dir
        runner.saved_batches = ["batch_0000", "batch_0001", "batch_0002"]

        # Create only batch_0000.pkl
        batch_0 = runner.output_dir / "batch_0000.pkl"
        with open(batch_0, "wb") as f:
            pickle.dump({"data": "test"}, f)

        # Validate - should return missing batch names
        missing, corrupt = runner._validate_batches()

        assert "batch_0001" in missing
        assert "batch_0002" in missing
        assert "batch_0000" not in missing
        assert corrupt == []

    def test_validate_batches_detects_corrupt(self, runner_with_output_dir):
        """Identifies corrupt pickle files (can't be loaded)."""
        runner = runner_with_output_dir
        runner.saved_batches = ["batch_0000", "batch_0001"]

        # Create valid batch_0000.pkl
        batch_0 = runner.output_dir / "batch_0000.pkl"
        with open(batch_0, "wb") as f:
            pickle.dump({"data": "test"}, f)

        # Create corrupt batch_0001.pkl (invalid pickle)
        batch_1 = runner.output_dir / "batch_0001.pkl"
        with open(batch_1, "wb") as f:
            f.write(b"not a valid pickle file!")

        # Validate
        missing, corrupt = runner._validate_batches()

        assert missing == []
        assert "batch_0001" in corrupt

    def test_validate_batches_all_valid(self, runner_with_output_dir):
        """Returns empty lists when all batches valid."""
        runner = runner_with_output_dir
        runner.saved_batches = ["batch_0000", "batch_0001"]

        # Create both valid
        for name in runner.saved_batches:
            path = runner.output_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump({"data": f"test_{name}"}, f)

        missing, corrupt = runner._validate_batches()

        assert missing == []
        assert corrupt == []

    def test_validate_batches_empty_list(self, runner_with_output_dir):
        """No validation errors when saved_batches is empty."""
        runner = runner_with_output_dir
        runner.saved_batches = []

        missing, corrupt = runner._validate_batches()

        assert missing == []
        assert corrupt == []

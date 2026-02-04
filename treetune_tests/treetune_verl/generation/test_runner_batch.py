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

"""Unit tests for GenerationRunner batch operations.

Tests the batch methods added to GenerationRunner:
- _extract_per_sample(): Extract 1D tensors from batched 2D DataProto
- _save_batch(): Save list of (index, sample_dict) to pickle
- _merge_batches(): Combine all batch files into trajectories.pkl

TDD approach: tests written first, then implementation.

Per-sample tensor structure (from spec):
{
    "input_ids": Tensor[seq_len],        # 1D, not 2D batched
    "attention_mask": Tensor[seq_len],
    "position_ids": Tensor[seq_len],
    "responses": Tensor[response_len],
    "response_mask": Tensor[response_len],
    "rollout_log_probs": Tensor[response_len],  # if present
    "prompts": Tensor[prompt_len],
}
"""

import pickle

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from treetune_verl.generation.runner import GenerationRunner
from verl.protocol import DataProto


def make_mock_dataproto(batch_size: int, seq_len: int = 128, response_len: int = 64) -> DataProto:
    """Create a mock DataProto with batched 2D tensors for testing.

    Args:
        batch_size: Number of samples in the batch
        seq_len: Sequence length for input tensors
        response_len: Length of response tensors

    Returns:
        DataProto with typical generation output fields
    """
    tensors = {
        "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.int64),
        "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        "responses": torch.randint(0, 32000, (batch_size, response_len)),
        "response_mask": torch.ones(batch_size, response_len, dtype=torch.int64),
        "rollout_log_probs": torch.randn(batch_size, response_len),
        "prompts": torch.randint(0, 32000, (batch_size, seq_len // 2)),
    }
    batch = TensorDict(source=tensors, batch_size=(batch_size,))

    # Non-tensor data (strings, metadata, etc.)
    non_tensor_batch = {
        "raw_prompt": np.array([f"prompt_{i}" for i in range(batch_size)], dtype=object),
        "data_source": np.array(["gsm8k"] * batch_size, dtype=object),
    }

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


@pytest.fixture
def runner_with_output_dir(tmp_path):
    """Create a GenerationRunner with mocked output_dir and state for batch operations."""
    runner = object.__new__(GenerationRunner)
    runner.output_dir = tmp_path
    runner.completed_indices = set()
    runner.saved_batches = []
    runner.total_samples = 100
    runner.config_snapshot = {"model": {"path": "/test/model"}}
    return runner


class TestExtractPerSampleSlicesTensors:
    """Tests for _extract_per_sample extracting 1D tensors from batched 2D."""

    def test_extract_per_sample_slices_tensors(self, runner_with_output_dir):
        """Extracts 1D tensors from batched 2D DataProto.

        Given a DataProto with shape [batch_size, seq_len], _extract_per_sample
        should return a list of (index, dict) where each dict contains 1D tensors.
        """
        runner = runner_with_output_dir
        batch_size = 4
        seq_len = 128
        dataproto = make_mock_dataproto(batch_size=batch_size, seq_len=seq_len)

        # Indices for each sample in this batch
        indices = [10, 11, 12, 13]

        # Extract per-sample data
        items = runner._extract_per_sample(dataproto, indices)

        # Should return list of (index, dict) tuples
        assert len(items) == batch_size

        # Check first item structure
        idx, sample_dict = items[0]
        assert idx == 10  # First index

        # Verify tensors are 1D (sliced from 2D batch)
        assert "input_ids" in sample_dict
        assert sample_dict["input_ids"].dim() == 1
        assert sample_dict["input_ids"].shape[0] == seq_len

        # Verify all expected tensor fields present
        assert "attention_mask" in sample_dict
        assert "position_ids" in sample_dict
        assert "responses" in sample_dict
        assert "response_mask" in sample_dict
        assert "rollout_log_probs" in sample_dict
        assert "prompts" in sample_dict


class TestExtractPerSampleHandlesAllFields:
    """Tests for _extract_per_sample handling all field types."""

    def test_extract_per_sample_handles_all_fields(self, runner_with_output_dir):
        """input_ids, responses, response_mask, etc. all extracted correctly."""
        runner = runner_with_output_dir
        batch_size = 2
        seq_len = 64
        response_len = 32
        dataproto = make_mock_dataproto(batch_size=batch_size, seq_len=seq_len, response_len=response_len)

        indices = [0, 1]
        items = runner._extract_per_sample(dataproto, indices)

        # Check second sample (index=1)
        idx, sample_dict = items[1]
        assert idx == 1

        # Verify each field's shape
        assert sample_dict["input_ids"].shape == (seq_len,)
        assert sample_dict["attention_mask"].shape == (seq_len,)
        assert sample_dict["position_ids"].shape == (seq_len,)
        assert sample_dict["responses"].shape == (response_len,)
        assert sample_dict["response_mask"].shape == (response_len,)
        assert sample_dict["rollout_log_probs"].shape == (response_len,)
        assert sample_dict["prompts"].shape == (seq_len // 2,)


class TestExtractPerSamplePreservesNumpy:
    """Tests for _extract_per_sample preserving numpy arrays in non_tensor_batch."""

    def test_extract_per_sample_preserves_numpy(self, runner_with_output_dir):
        """Handles numpy arrays in non_tensor_batch correctly."""
        runner = runner_with_output_dir
        batch_size = 3
        dataproto = make_mock_dataproto(batch_size=batch_size)

        indices = [5, 6, 7]
        items = runner._extract_per_sample(dataproto, indices)

        # Check that non-tensor data is extracted
        idx, sample_dict = items[0]
        assert idx == 5

        # Non-tensor fields should be present as individual items (not arrays)
        assert "raw_prompt" in sample_dict
        assert sample_dict["raw_prompt"] == "prompt_0"  # First sample's prompt

        # Check all samples have their correct non-tensor data
        for i, (idx, sample_dict) in enumerate(items):
            assert sample_dict["raw_prompt"] == f"prompt_{i}"
            assert sample_dict["data_source"] == "gsm8k"


class TestSaveBatchWritesPickle:
    """Tests for _save_batch writing pickle files."""

    def test_save_batch_writes_pickle(self, runner_with_output_dir):
        """Creates batch_NNNN.pkl file with correct structure."""
        runner = runner_with_output_dir

        # Create sample items (would come from _extract_per_sample)
        items = [
            (10, {"input_ids": torch.tensor([1, 2, 3]), "raw_prompt": "test1"}),
            (11, {"input_ids": torch.tensor([4, 5, 6]), "raw_prompt": "test2"}),
        ]

        # Save batch
        batch_name = runner._save_batch(items, batch_idx=0)

        # Should return batch name
        assert batch_name == "batch_0000"

        # Verify file exists
        batch_path = runner.output_dir / "batch_0000.pkl"
        assert batch_path.exists()

        # Verify contents can be loaded
        with open(batch_path, "rb") as f:
            loaded = pickle.load(f)

        # Should be list of (index, dict) tuples
        assert len(loaded) == 2
        assert loaded[0][0] == 10  # First index
        assert torch.equal(loaded[0][1]["input_ids"], torch.tensor([1, 2, 3]))
        assert loaded[1][0] == 11
        assert loaded[1][1]["raw_prompt"] == "test2"


class TestSaveBatchUpdatesCheckpoint:
    """Tests for _save_batch updating checkpoint state."""

    def test_save_batch_updates_checkpoint(self, runner_with_output_dir):
        """Adds indices and batch name to checkpoint state."""
        runner = runner_with_output_dir

        # Initial state
        assert runner.completed_indices == set()
        assert runner.saved_batches == []

        # Create and save batch
        items = [
            (10, {"input_ids": torch.tensor([1, 2, 3])}),
            (11, {"input_ids": torch.tensor([4, 5, 6])}),
            (12, {"input_ids": torch.tensor([7, 8, 9])}),
        ]
        batch_name = runner._save_batch(items, batch_idx=0)

        # Verify completed_indices updated
        assert 10 in runner.completed_indices
        assert 11 in runner.completed_indices
        assert 12 in runner.completed_indices

        # Verify saved_batches updated
        assert batch_name in runner.saved_batches
        assert runner.saved_batches == ["batch_0000"]

        # Save another batch
        items2 = [(20, {"input_ids": torch.tensor([10, 11, 12])})]
        batch_name2 = runner._save_batch(items2, batch_idx=1)

        assert batch_name2 == "batch_0001"
        assert 20 in runner.completed_indices
        assert runner.saved_batches == ["batch_0000", "batch_0001"]


class TestMergeBatchesCombinesAll:
    """Tests for _merge_batches combining all batch files."""

    def test_merge_batches_combines_all(self, runner_with_output_dir):
        """Merges batch files into trajectories.pkl."""
        runner = runner_with_output_dir

        # Create multiple batch files manually
        batch0_items = [
            (0, {"input_ids": torch.tensor([1, 2]), "prompt": "a"}),
            (1, {"input_ids": torch.tensor([3, 4]), "prompt": "b"}),
        ]
        batch1_items = [
            (2, {"input_ids": torch.tensor([5, 6]), "prompt": "c"}),
            (3, {"input_ids": torch.tensor([7, 8]), "prompt": "d"}),
        ]

        # Save batches
        with open(runner.output_dir / "batch_0000.pkl", "wb") as f:
            pickle.dump(batch0_items, f)
        with open(runner.output_dir / "batch_0001.pkl", "wb") as f:
            pickle.dump(batch1_items, f)

        runner.saved_batches = ["batch_0000", "batch_0001"]

        # Merge batches
        runner._merge_batches()

        # Verify trajectories.pkl exists
        trajectories_path = runner.output_dir / "trajectories.pkl"
        assert trajectories_path.exists()

        # Load and verify
        with open(trajectories_path, "rb") as f:
            merged = pickle.load(f)

        # Should contain all items
        assert len(merged) == 4


class TestMergeBatchesSortsByIndex:
    """Tests for _merge_batches sorting output by sample index."""

    def test_merge_batches_sorts_by_index(self, runner_with_output_dir):
        """Final output sorted by sample index."""
        runner = runner_with_output_dir

        # Create batch files with out-of-order indices
        batch0_items = [
            (5, {"input_ids": torch.tensor([1, 2]), "value": "e"}),
            (2, {"input_ids": torch.tensor([3, 4]), "value": "b"}),
        ]
        batch1_items = [
            (0, {"input_ids": torch.tensor([5, 6]), "value": "a"}),
            (3, {"input_ids": torch.tensor([7, 8]), "value": "c"}),
            (1, {"input_ids": torch.tensor([9, 10]), "value": "d"}),
        ]

        # Save batches
        with open(runner.output_dir / "batch_0000.pkl", "wb") as f:
            pickle.dump(batch0_items, f)
        with open(runner.output_dir / "batch_0001.pkl", "wb") as f:
            pickle.dump(batch1_items, f)

        runner.saved_batches = ["batch_0000", "batch_0001"]

        # Merge batches
        runner._merge_batches()

        # Load and verify sorted order
        with open(runner.output_dir / "trajectories.pkl", "rb") as f:
            merged = pickle.load(f)

        # Should be sorted by index
        indices = [item[0] for item in merged]
        assert indices == [0, 1, 2, 3, 5], f"Expected sorted indices, got {indices}"

        # Verify data matches indices
        assert merged[0][1]["value"] == "a"  # index 0
        assert merged[1][1]["value"] == "d"  # index 1
        assert merged[2][1]["value"] == "b"  # index 2
        assert merged[3][1]["value"] == "c"  # index 3
        assert merged[4][1]["value"] == "e"  # index 5

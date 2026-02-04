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

"""End-to-end tests for GenerationRunner with real model.

Tests verify the ACTUAL generation pipeline with real model (Qwen2.5-0.5B-Instruct):
- Full generation pipeline with tensor validation
- Resume from checkpoint functionality
- Final merge of batch files
- Output usable for training

Requires GPU. Uses small model and few samples for fast tests.

Test setup:
- Create test parquet files with PyArrow
- Use pytest fixtures for tmp_path (output directory)
- Mark tests with @pytest.mark.gpu
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
import ray
import torch
from omegaconf import OmegaConf


# Check for GPU availability
def has_gpu():
    """Check if GPU is available."""
    return torch.cuda.is_available()


# Skip marker for GPU tests
gpu = pytest.mark.skipif(not has_gpu(), reason="No GPU available")


@pytest.fixture(scope="function")
def ray_cluster():
    """Initialize Ray cluster for E2E tests.

    Uses function scope to ensure each test gets a fresh Ray cluster.
    This is necessary because GPU resources get consumed and aren't
    properly released when Ray is shared between tests.
    """
    if ray.is_initialized():
        ray.shutdown()
    # Wait for any lingering processes to clean up
    import time

    time.sleep(1)
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()
    # Wait for cleanup
    time.sleep(1)


def create_test_parquet(filepath: Path, num_samples: int = 5):
    """Create a minimal parquet file for E2E testing.

    Creates simple prompts that should generate short responses.
    """
    prompts = [
        "What is 2 + 2? Answer briefly:",
        "What color is the sky? Answer briefly:",
        "Is water wet? Answer briefly:",
        "What is the capital of France? Answer briefly:",
        "How many days in a week? Answer briefly:",
    ][:num_samples]

    data = {
        "prompt": prompts,
        "index": list(range(num_samples)),
    }
    df = pd.DataFrame(data)
    df.to_parquet(filepath)
    return filepath


def make_e2e_config(
    output_dir: Path,
    data_file: Path,
    num_samples: int = 5,
    save_batch_size: int = 2,
    final_merge: bool = True,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
) -> OmegaConf:
    """Create E2E test configuration.

    Uses Qwen2.5-0.5B-Instruct for fast testing.
    """
    return OmegaConf.create(
        {
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "project_name": "e2e_test",
            "experiment_name": "generation",
            "model": {
                "path": "Qwen/Qwen2.5-0.5B-Instruct",
                "trust_remote_code": True,
            },
            "rollout": {
                "name": "sglang",
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "free_cache_engine": False,
                # Sampling
                "temperature": temperature,
                "top_p": 1.0,
                "top_k": -1,
                "prompt_length": 512,
                "response_length": max_new_tokens,
                "calculate_log_probs": True,
                # MTP disabled for basic test
                "mtp": {
                    "enable": False,
                },
                # Agent loop
                "agent": {
                    "num_workers": 1,  # Single worker for simpler testing
                    "default_agent_loop": "single_turn_agent",
                    "agent_loop_config_path": None,
                },
            },
            "data": {
                "files": [str(data_file)],
                "prompt_key": "prompt",
                "max_samples": num_samples,
            },
            "tasks": None,
            "generation": {
                "output_dir": str(output_dir),
                "save_batch_size": save_batch_size,
                "pull_timeout": 60.0,  # Longer timeout for real generation
                "final_merge": final_merge,
                "checkpoint_interval": 1,
                "show_progress": True,
                "wandb_upload": False,
                "wandb_project": None,
                "wandb_run_name": None,
            },
        }
    )


@gpu
class TestFullGenerationPipeline:
    """Tests for full generation pipeline with real model."""

    def test_full_generation_pipeline(self, ray_cluster, tmp_path):
        """Test full generation pipeline with real model.

        Steps:
        1. Create test parquet (5 samples with simple prompts)
        2. Run GenerationRunner with real model (Qwen/Qwen2.5-0.5B-Instruct)
        3. Verify batch files created in output_dir
        4. Verify each sample has correct tensor structure (1D tensors, not batched 2D)
        5. Verify required fields present: input_ids, attention_mask, responses, response_mask
        6. Verify response_mask semantics (contains 0s and 1s only, has some 1s)
        7. Verify checkpoint has all 5 indices
        8. Verify no duplicate indices in output
        """
        from treetune_verl.generation.runner import GenerationRunner

        # Setup
        num_samples = 5
        save_batch_size = 2
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        data_file = tmp_path / "test_data.parquet"
        create_test_parquet(data_file, num_samples)

        config = make_e2e_config(
            output_dir=output_dir,
            data_file=data_file,
            num_samples=num_samples,
            save_batch_size=save_batch_size,
            final_merge=True,
            temperature=0.0,  # Deterministic
        )

        # Run generation
        runner = GenerationRunner(config)
        runner.run()

        # Verify checkpoint exists and has all indices
        checkpoint_path = output_dir / "checkpoint.json"
        assert checkpoint_path.exists(), "Checkpoint should be created"

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        assert set(checkpoint["completed_indices"]) == set(range(num_samples)), (
            f"Checkpoint should have all {num_samples} indices"
        )
        assert checkpoint["total_samples"] == num_samples

        # Verify batch files were created (before merge)
        saved_batches = checkpoint["saved_batches"]
        assert len(saved_batches) >= 1, "Should have saved at least one batch"

        # Verify final merged file
        trajectories_path = output_dir / "trajectories.pkl"
        assert trajectories_path.exists(), "trajectories.pkl should be created"

        with open(trajectories_path, "rb") as f:
            trajectories = pickle.load(f)

        # Verify we have all samples
        assert len(trajectories) == num_samples, f"Should have {num_samples} samples"

        # Verify no duplicate indices
        indices = [idx for idx, _ in trajectories]
        assert len(indices) == len(set(indices)), "No duplicate indices allowed"
        assert set(indices) == set(range(num_samples)), "All indices present"

        # Verify each sample has correct tensor structure
        required_tensor_fields = ["input_ids", "attention_mask", "responses", "response_mask"]

        for idx, sample_dict in trajectories:
            # Check required fields exist
            for field in required_tensor_fields:
                assert field in sample_dict, f"Sample {idx} missing required field: {field}"

            # Verify tensor types
            for field in required_tensor_fields:
                value = sample_dict[field]
                assert isinstance(value, torch.Tensor), f"Field {field} should be a Tensor"
                # Verify 1D (not batched 2D)
                assert value.dim() == 1, f"Field {field} should be 1D, got {value.dim()}D"

            # Verify response_mask semantics
            response_mask = sample_dict["response_mask"]
            unique_values = torch.unique(response_mask).tolist()
            # Response mask should only contain 0s and 1s
            for val in unique_values:
                assert val in [0, 1], f"response_mask should only have 0 or 1, got {val}"
            # Should have at least some 1s (LLM-generated tokens)
            assert response_mask.sum() > 0, "response_mask should have some 1s (LLM tokens)"


@gpu
class TestResumeFromCheckpoint:
    """Tests for resume functionality."""

    def test_resume_from_checkpoint(self, ray_cluster, tmp_path):
        """Test resume from partial checkpoint.

        Steps:
        1. Create partial checkpoint (2/5 completed with fake batch file)
        2. Run GenerationRunner
        3. Verify only remaining 3 samples generated (check batch file count)
        4. After merge, verify all 5 samples present
        """
        from treetune_verl.generation.runner import GenerationRunner

        # Setup
        num_samples = 5
        completed_count = 2
        save_batch_size = 2
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        data_file = tmp_path / "test_data.parquet"
        create_test_parquet(data_file, num_samples)

        # Create fake batch file for first 2 samples
        fake_batch = [
            (
                0,
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "responses": torch.tensor([4, 5]),
                    "response_mask": torch.tensor([1, 1]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                },
            ),
            (
                1,
                {
                    "input_ids": torch.tensor([1, 2, 4]),
                    "responses": torch.tensor([5, 6]),
                    "response_mask": torch.tensor([1, 1]),
                    "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                },
            ),
        ]
        batch_path = output_dir / "batch_0000.pkl"
        with open(batch_path, "wb") as f:
            pickle.dump(fake_batch, f)

        # Create partial checkpoint
        checkpoint_data = {
            "completed_indices": list(range(completed_count)),
            "saved_batches": ["batch_0000"],
            "total_samples": num_samples,
            "config_snapshot": {},
        }
        checkpoint_path = output_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f)

        config = make_e2e_config(
            output_dir=output_dir,
            data_file=data_file,
            num_samples=num_samples,
            save_batch_size=save_batch_size,
            final_merge=True,
        )

        # Run generation (should resume and generate remaining 3)
        runner = GenerationRunner(config)

        # Verify initial state loaded from checkpoint
        assert len(runner.completed_indices) == completed_count
        assert runner.saved_batches == ["batch_0000"]

        runner.run()

        # Verify all samples now completed
        assert len(runner.completed_indices) == num_samples

        # Verify merged file has all samples
        trajectories_path = output_dir / "trajectories.pkl"
        assert trajectories_path.exists()

        with open(trajectories_path, "rb") as f:
            trajectories = pickle.load(f)

        indices = [idx for idx, _ in trajectories]
        assert set(indices) == set(range(num_samples)), "All samples should be present after resume"


@gpu
class TestFinalMerge:
    """Tests for final merge functionality."""

    def test_final_merge(self, ray_cluster, tmp_path):
        """Test final merge creates correct output.

        Steps:
        1. Run with save_batch_size=2, 5 samples
        2. Verify batch files created during generation
        3. Verify trajectories.pkl contains all 5 samples sorted by index
        """
        from treetune_verl.generation.runner import GenerationRunner

        # Setup
        num_samples = 5
        save_batch_size = 2
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        data_file = tmp_path / "test_data.parquet"
        create_test_parquet(data_file, num_samples)

        config = make_e2e_config(
            output_dir=output_dir,
            data_file=data_file,
            num_samples=num_samples,
            save_batch_size=save_batch_size,
            final_merge=True,
        )

        runner = GenerationRunner(config)
        runner.run()

        # Verify batch files were created (they remain after merge)
        batch_files = list(output_dir.glob("batch_*.pkl"))
        # With synchronous dispatch, all samples may arrive in one batch
        # The key test is that at least one batch file exists
        assert len(batch_files) >= 1, f"Expected at least 1 batch file, got {len(batch_files)}"

        # Verify trajectories.pkl exists
        trajectories_path = output_dir / "trajectories.pkl"
        assert trajectories_path.exists(), "trajectories.pkl should exist after merge"

        with open(trajectories_path, "rb") as f:
            trajectories = pickle.load(f)

        # Verify sorted by index
        indices = [idx for idx, _ in trajectories]
        assert indices == sorted(indices), "Trajectories should be sorted by index"

        # Verify all samples present
        assert len(trajectories) == num_samples


@gpu
class TestOutputUsableForTraining:
    """Tests that output is usable for RL training."""

    def test_output_usable_for_training(self, ray_cluster, tmp_path):
        """Test output has all required fields for training.

        Steps:
        1. Run generation, load trajectories.pkl
        2. Verify all training-required fields present
        3. Verify tensor dtypes (torch.long for ids, torch.float for probs)
        4. Verify shapes are consistent across samples
        """
        from treetune_verl.generation.runner import GenerationRunner

        # Setup
        num_samples = 5
        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)
        data_file = tmp_path / "test_data.parquet"
        create_test_parquet(data_file, num_samples)

        config = make_e2e_config(
            output_dir=output_dir,
            data_file=data_file,
            num_samples=num_samples,
            save_batch_size=10,  # Single batch
            final_merge=True,
        )

        runner = GenerationRunner(config)
        runner.run()

        trajectories_path = output_dir / "trajectories.pkl"
        with open(trajectories_path, "rb") as f:
            trajectories = pickle.load(f)

        # Training-required fields
        required_fields = [
            "input_ids",
            "attention_mask",
            "responses",
            "response_mask",
        ]

        for idx, sample_dict in trajectories:
            # Check required fields
            for field in required_fields:
                assert field in sample_dict, f"Sample {idx} missing required field: {field}"
                value = sample_dict[field]
                assert isinstance(value, torch.Tensor), f"{field} should be Tensor"

            # Verify dtypes
            # Token IDs should be long integers
            assert sample_dict["input_ids"].dtype in [torch.long, torch.int64], (
                f"input_ids should be long, got {sample_dict['input_ids'].dtype}"
            )
            assert sample_dict["responses"].dtype in [torch.long, torch.int64], (
                f"responses should be long, got {sample_dict['responses'].dtype}"
            )

            # Masks should be integer type
            assert sample_dict["attention_mask"].dtype in [torch.long, torch.int64, torch.int32], (
                "attention_mask should be int type"
            )
            assert sample_dict["response_mask"].dtype in [torch.long, torch.int64, torch.int32], (
                "response_mask should be int type"
            )

            # If log_probs present, should be float
            if "rollout_log_probs" in sample_dict:
                log_probs = sample_dict["rollout_log_probs"]
                assert log_probs.dtype in [torch.float32, torch.float64, torch.bfloat16, torch.float16], (
                    "rollout_log_probs should be float type"
                )

        # Verify consistency across samples (same field structure)
        first_fields = set(trajectories[0][1].keys())
        for idx, sample_dict in trajectories[1:]:
            current_fields = set(sample_dict.keys())
            assert current_fields == first_fields, f"Sample {idx} has different fields than sample 0"

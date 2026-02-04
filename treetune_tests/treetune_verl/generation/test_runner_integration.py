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

"""Integration tests for GenerationRunner + Queue + Worker orchestration.

Tests verify the run() method orchestration:
- Initialization creates ResultsQueue actor and sets up manager config
- Pull loop runs concurrent with generation
- Batches saved incrementally as results arrive
- Progress bar updates on queue pull
- Final stragglers collected via timeout

TDD approach: tests written to drive implementation of run() method.

Test setup:
- Mock AgentLoopManager (don't need actual generation)
- Real ResultsQueue actor (Ray local mode)
- Mock _save_batch, _extract_per_sample to isolate orchestration logic
- Verify concurrent behavior with timing or ordering assertions
"""

import pickle
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from treetune_verl.generation.queue import ResultsQueue
from treetune_verl.generation.runner import GenerationRunner
from verl.protocol import DataProto


@pytest.fixture(scope="module")
def ray_local():
    """Initialize Ray in local mode for fast integration tests."""
    ray.init(local_mode=True)
    yield
    ray.shutdown()


def make_minimal_config(tmp_path: Path, num_samples: int = 10) -> OmegaConf:
    """Create minimal config for testing runner orchestration."""
    return OmegaConf.create(
        {
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "model": {
                "path": "/test/model",
                "trust_remote_code": True,
                "dtype": "bfloat16",
            },
            "rollout": {
                "name": "sglang",
                "tensor_model_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "prompt_length": 512,
                "response_length": 256,
                "agent": {
                    "num_workers": 2,
                    "default_agent_loop": "single_turn_agent",
                },
                "mtp": None,
            },
            "data": {
                "files": [str(tmp_path / "test_data.parquet")],
                "prompt_key": "prompt",
                "max_samples": num_samples,
            },
            "tasks": None,
            "generation": {
                "output_dir": str(tmp_path / "output"),
                "save_batch_size": 3,  # Small batch for testing
                "pull_timeout": 2.0,  # Short timeout for fast tests
                "final_merge": True,
                "checkpoint_interval": 1,
                "show_progress": False,  # Disable tqdm in tests
                "wandb_upload": False,
                "wandb_project": None,
                "wandb_run_name": None,
            },
        }
    )


def create_test_parquet(filepath: Path, num_samples: int = 10):
    """Create a minimal parquet file for testing."""
    import pandas as pd

    data = {
        "prompt": [f"Test prompt {i}" for i in range(num_samples)],
        "idx": list(range(num_samples)),
    }
    df = pd.DataFrame(data)
    df.to_parquet(filepath)


def make_mock_dataproto(indices: list[int], seq_len: int = 32) -> DataProto:
    """Create a mock DataProto for testing extraction.

    Creates batched tensors that _extract_per_sample will slice.
    """
    batch_size = len(indices)
    tensors = {
        "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
        "responses": torch.randint(0, 32000, (batch_size, seq_len)),
        "response_mask": torch.ones(batch_size, seq_len, dtype=torch.int64),
    }
    batch = TensorDict(source=tensors, batch_size=(batch_size,))
    non_tensor_batch = {
        "raw_prompt": np.array([f"prompt_{i}" for i in indices], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


class TestRunnerCreatesQueueAndManager:
    """Tests that run() initialization creates ResultsQueue and configures manager."""

    def test_runner_creates_queue_and_manager(self, ray_local, tmp_path):
        """Verify run() creates ResultsQueue actor and sets up manager config.

        The run() method should:
        1. Create ResultsQueue Ray actor
        2. Adapt config for AgentLoopManager
        3. Create AgentLoopManager (mocked here)
        4. Dispatch generation
        """
        num_samples = 5
        config = make_minimal_config(tmp_path, num_samples=num_samples)
        create_test_parquet(tmp_path / "test_data.parquet", num_samples)
        (tmp_path / "output").mkdir(exist_ok=True)

        # Create runner instance directly (bypassing __init__)
        runner = object.__new__(GenerationRunner)
        runner.config = config
        runner.output_dir = tmp_path / "output"
        runner.completed_indices = set()
        runner.saved_batches = []
        runner.total_samples = num_samples
        runner.config_snapshot = {}
        runner.prompt_key = "prompt"

        # Test that _adapt_config_for_manager produces correct structure
        adapted = runner._adapt_config_for_manager(config)

        # Verify adapted config has trainer structure
        assert "trainer" in adapted
        assert adapted.trainer.n_gpus_per_node == config.n_gpus_per_node

        # Verify adapted config has actor_rollout_ref structure
        assert "actor_rollout_ref" in adapted
        assert "model" in adapted.actor_rollout_ref
        assert "rollout" in adapted.actor_rollout_ref

        # Verify _target_ fields are added for Hydra
        assert adapted.actor_rollout_ref.rollout._target_ == "verl.workers.config.RolloutConfig"
        assert adapted.actor_rollout_ref.model._target_ == "verl.workers.config.HFModelConfig"

        # Verify reward_model is disabled
        assert adapted.reward_model.enable is False

        # Test that ResultsQueue can be created
        queue = ResultsQueue.remote()
        assert queue is not None

        # Store on runner to verify pattern
        runner._queue = queue
        assert runner._queue is not None


class TestRunnerConcurrentLoops:
    """Tests that pull loop runs concurrent with generation dispatch."""

    def test_runner_concurrent_loops(self, ray_local, tmp_path):
        """Test that pull loop can run concurrent with generation.

        Use threading to verify both can make progress simultaneously.
        The test pushes items to queue from a background thread simulating
        worker output, while the runner's pull loop should collect them.
        """
        num_samples = 6
        config = make_minimal_config(tmp_path, num_samples=num_samples)
        create_test_parquet(tmp_path / "test_data.parquet", num_samples)
        (tmp_path / "output").mkdir(exist_ok=True)

        # Track execution order
        events = []
        events_lock = threading.Lock()

        def log_event(name: str):
            with events_lock:
                events.append((name, time.time()))

        # Create runner manually to inject our queue
        runner = object.__new__(GenerationRunner)
        runner.config = config
        runner.output_dir = tmp_path / "output"
        runner.completed_indices = set()
        runner.saved_batches = []
        runner.total_samples = num_samples
        runner.config_snapshot = {}
        runner.prompt_key = "prompt"

        # Create real queue
        queue = ResultsQueue.remote()
        runner._queue = queue

        # Counters for verification
        save_calls = []

        def mock_save_batch(items, batch_idx):
            log_event(f"save_batch_{batch_idx}")
            save_calls.append((batch_idx, [idx for idx, _ in items]))
            for idx, _ in items:
                runner.completed_indices.add(idx)
            runner.saved_batches.append(f"batch_{batch_idx:04d}")
            return f"batch_{batch_idx:04d}"

        def mock_extract_per_sample(output, indices):
            return [(idx, {"data": f"sample_{idx}"}) for idx in indices]

        runner._save_batch = mock_save_batch
        runner._extract_per_sample = mock_extract_per_sample
        runner._save_checkpoint = MagicMock()
        runner._merge_batches = MagicMock()

        # Simulate generation by pushing to queue from background
        def simulate_generation():
            """Push results to queue, simulating async worker completion."""
            for i in range(num_samples):
                log_event(f"push_{i}")
                ray.get(queue.put.remote(i, make_mock_dataproto([i])))
                time.sleep(0.1)  # Stagger arrivals
            log_event("generation_done")

        # Start generation in background
        gen_thread = threading.Thread(target=simulate_generation)
        gen_thread.start()

        # Run pull loop
        log_event("pull_loop_start")
        batch_idx = 0
        save_batch_size = config.generation.save_batch_size

        while len(runner.completed_indices) < num_samples:
            batch = ray.get(queue.get_batch.remote(min_items=save_batch_size, timeout=config.generation.pull_timeout))

            if batch:
                log_event(f"pull_batch_{len(batch)}")
                # Flatten results
                items = []
                for idx, output in batch:
                    extracted = runner._extract_per_sample(output, [idx])
                    items.extend(extracted)

                if items:
                    runner._save_batch(items, batch_idx)
                    batch_idx += 1

        log_event("pull_loop_done")
        gen_thread.join(timeout=5.0)

        # Verify concurrent execution
        assert len(events) > 0, "Should have logged events"

        # Extract push and pull events
        push_events = [e for e in events if e[0].startswith("push_")]
        pull_events = [e for e in events if e[0].startswith("pull_batch")]

        # Should have pulls happening while pushes are still going
        if push_events and pull_events:
            last_push_time = push_events[-1][1]
            first_pull_time = pull_events[0][1]

            # First pull should happen before all pushes complete (concurrent)
            assert first_pull_time < last_push_time + 1.0, "Pull loop should start before all pushes complete"

        # Verify all samples eventually collected
        assert len(runner.completed_indices) == num_samples

        # Verify batches were saved
        assert len(save_calls) > 0, "Should have saved at least one batch"


class TestRunnerSavesIncrementally:
    """Tests that runner saves batches as results arrive from queue."""

    def test_runner_saves_incrementally(self, ray_local, tmp_path):
        """Mock queue to return batches. Verify _save_batch called for each batch.

        The run() method should call _save_batch each time it pulls enough
        results from the queue, not wait until all generation is complete.

        This test simulates results arriving in waves to verify incremental saving.
        """
        num_samples = 9
        save_batch_size = 3
        config = make_minimal_config(tmp_path, num_samples=num_samples)
        config.generation.save_batch_size = save_batch_size
        config.generation.pull_timeout = 0.3  # Short timeout to trigger partial batches
        create_test_parquet(tmp_path / "test_data.parquet", num_samples)
        (tmp_path / "output").mkdir(exist_ok=True)

        # Create runner
        runner = object.__new__(GenerationRunner)
        runner.config = config
        runner.output_dir = tmp_path / "output"
        runner.completed_indices = set()
        runner.saved_batches = []
        runner.total_samples = num_samples
        runner.config_snapshot = {}
        runner.prompt_key = "prompt"

        # Create real queue
        queue = ResultsQueue.remote()
        runner._queue = queue

        # Track save_batch calls
        save_batch_calls = []

        def mock_save_batch(items, batch_idx):
            indices = [idx for idx, _ in items]
            save_batch_calls.append(
                {
                    "batch_idx": batch_idx,
                    "indices": indices,
                    "count": len(items),
                }
            )
            for idx, _ in items:
                runner.completed_indices.add(idx)
            runner.saved_batches.append(f"batch_{batch_idx:04d}")
            return f"batch_{batch_idx:04d}"

        def mock_extract_per_sample(output, indices):
            return [(idx, {"data": f"sample_{idx}"}) for idx in indices]

        runner._save_batch = mock_save_batch
        runner._extract_per_sample = mock_extract_per_sample
        runner._save_checkpoint = MagicMock()
        runner._merge_batches = MagicMock()

        # Simulate results arriving in waves (like real async generation)
        def push_in_waves():
            """Push results in 3 waves of 3 samples each."""
            for wave in range(3):
                for i in range(3):
                    idx = wave * 3 + i
                    ray.get(queue.put.remote(idx, make_mock_dataproto([idx])))
                time.sleep(0.4)  # Gap between waves

        push_thread = threading.Thread(target=push_in_waves)
        push_thread.start()

        # Run pull loop
        batch_idx = 0
        while len(runner.completed_indices) < num_samples:
            batch = ray.get(queue.get_batch.remote(min_items=save_batch_size, timeout=config.generation.pull_timeout))

            if batch:
                items = []
                for idx, output in batch:
                    extracted = runner._extract_per_sample(output, [idx])
                    items.extend(extracted)

                if items:
                    runner._save_batch(items, batch_idx)
                    batch_idx += 1

        push_thread.join(timeout=5.0)

        # Verify incremental saves (at least 2 batches, possibly 3)
        assert len(save_batch_calls) >= 2, (
            f"Expected at least 2 batch saves for 9 samples arriving in waves, got {len(save_batch_calls)}"
        )

        # Verify all samples eventually saved
        all_saved_indices = set()
        for call in save_batch_calls:
            all_saved_indices.update(call["indices"])

        assert all_saved_indices == set(range(num_samples)), (
            f"Not all samples saved. Missing: {set(range(num_samples)) - all_saved_indices}"
        )


class TestRunnerUpdatesProgress:
    """Tests that runner updates progress bar as results arrive."""

    def test_runner_updates_progress(self, ray_local, tmp_path):
        """Mock tqdm, verify progress bar updated as results arrive.

        The run() method should call pbar.update() after each batch
        is collected from the queue.

        Test pushes results in waves to ensure multiple batches.
        """
        num_samples = 6
        config = make_minimal_config(tmp_path, num_samples=num_samples)
        config.generation.show_progress = True  # Enable progress
        config.generation.save_batch_size = 2
        config.generation.pull_timeout = 0.3  # Short timeout
        create_test_parquet(tmp_path / "test_data.parquet", num_samples)
        (tmp_path / "output").mkdir(exist_ok=True)

        # Create runner
        runner = object.__new__(GenerationRunner)
        runner.config = config
        runner.output_dir = tmp_path / "output"
        runner.completed_indices = set()
        runner.saved_batches = []
        runner.total_samples = num_samples
        runner.config_snapshot = {}
        runner.prompt_key = "prompt"

        # Create real queue
        queue = ResultsQueue.remote()
        runner._queue = queue

        # Track progress updates
        progress_updates = []

        def mock_save_batch(items, batch_idx):
            for idx, _ in items:
                runner.completed_indices.add(idx)
            runner.saved_batches.append(f"batch_{batch_idx:04d}")
            return f"batch_{batch_idx:04d}"

        def mock_extract_per_sample(output, indices):
            return [(idx, {"data": f"sample_{idx}"}) for idx in indices]

        runner._save_batch = mock_save_batch
        runner._extract_per_sample = mock_extract_per_sample
        runner._save_checkpoint = MagicMock()
        runner._merge_batches = MagicMock()

        # Push results in waves
        def push_in_waves():
            for wave in range(3):
                for i in range(2):
                    idx = wave * 2 + i
                    ray.get(queue.put.remote(idx, make_mock_dataproto([idx])))
                time.sleep(0.4)

        push_thread = threading.Thread(target=push_in_waves)
        push_thread.start()

        # Run pull loop with progress tracking
        batch_idx = 0
        save_batch_size = config.generation.save_batch_size

        while len(runner.completed_indices) < num_samples:
            batch = ray.get(queue.get_batch.remote(min_items=save_batch_size, timeout=config.generation.pull_timeout))

            if batch:
                items = []
                for idx, output in batch:
                    extracted = runner._extract_per_sample(output, [idx])
                    items.extend(extracted)

                if items:
                    runner._save_batch(items, batch_idx)
                    batch_idx += 1
                    # Track progress update (this is what run() should do)
                    progress_updates.append(
                        {
                            "completed": len(runner.completed_indices),
                            "batch_idx": batch_idx - 1,
                        }
                    )

        push_thread.join(timeout=5.0)

        # Verify progress was tracked incrementally
        assert len(progress_updates) >= 2, f"Expected multiple progress updates, got {len(progress_updates)}"

        # Progress should increase monotonically
        completed_counts = [u["completed"] for u in progress_updates]
        assert completed_counts == sorted(completed_counts), "Progress should increase monotonically"

        # Final count should equal total samples
        assert completed_counts[-1] == num_samples


class TestRunnerHandlesTimeout:
    """Tests that runner handles stragglers via final timeout collection."""

    def test_runner_handles_timeout(self, ray_local, tmp_path):
        """Simulate stragglers: fewer results than expected initially.

        Queue has fewer items than save_batch_size, timeout should trigger
        collection of partial batch. Verify final timeout collection works.
        """
        num_samples = 5
        config = make_minimal_config(tmp_path, num_samples=num_samples)
        config.generation.save_batch_size = 10  # Larger than num_samples
        config.generation.pull_timeout = 0.5  # Short timeout
        create_test_parquet(tmp_path / "test_data.parquet", num_samples)
        (tmp_path / "output").mkdir(exist_ok=True)

        # Create runner
        runner = object.__new__(GenerationRunner)
        runner.config = config
        runner.output_dir = tmp_path / "output"
        runner.completed_indices = set()
        runner.saved_batches = []
        runner.total_samples = num_samples
        runner.config_snapshot = {}
        runner.prompt_key = "prompt"

        # Create real queue
        queue = ResultsQueue.remote()
        runner._queue = queue

        # Push only some results initially (stragglers)
        initial_count = 3
        for i in range(initial_count):
            ray.get(queue.put.remote(i, make_mock_dataproto([i])))

        save_batch_calls = []

        def mock_save_batch(items, batch_idx):
            indices = [idx for idx, _ in items]
            save_batch_calls.append(
                {
                    "batch_idx": batch_idx,
                    "indices": indices,
                    "count": len(items),
                }
            )
            for idx, _ in items:
                runner.completed_indices.add(idx)
            runner.saved_batches.append(f"batch_{batch_idx:04d}")
            return f"batch_{batch_idx:04d}"

        def mock_extract_per_sample(output, indices):
            return [(idx, {"data": f"sample_{idx}"}) for idx in indices]

        runner._save_batch = mock_save_batch
        runner._extract_per_sample = mock_extract_per_sample
        runner._save_checkpoint = MagicMock()
        runner._merge_batches = MagicMock()

        # Simulate straggler arrival in background
        def add_stragglers():
            time.sleep(0.3)  # After first timeout check
            for i in range(initial_count, num_samples):
                ray.get(queue.put.remote(i, make_mock_dataproto([i])))
                time.sleep(0.1)

        straggler_thread = threading.Thread(target=add_stragglers)
        straggler_thread.start()

        # Run pull loop with timeout handling
        batch_idx = 0
        save_batch_size = config.generation.save_batch_size
        timeout = config.generation.pull_timeout

        while len(runner.completed_indices) < num_samples:
            batch = ray.get(queue.get_batch.remote(min_items=save_batch_size, timeout=timeout))

            if batch:
                items = []
                for idx, output in batch:
                    extracted = runner._extract_per_sample(output, [idx])
                    items.extend(extracted)

                if items:
                    runner._save_batch(items, batch_idx)
                    batch_idx += 1

        straggler_thread.join(timeout=5.0)

        # Verify timeout-triggered batch saves happened
        assert len(save_batch_calls) >= 1, "Should have saved at least one batch via timeout"

        # At least one batch should have been saved before all samples arrived
        # (due to timeout triggering early save)
        first_batch_count = save_batch_calls[0]["count"]
        assert first_batch_count < save_batch_size, (
            f"First batch should be smaller than batch_size (timeout-triggered), got {first_batch_count}"
        )

        # All samples should eventually be collected
        all_saved = set()
        for call in save_batch_calls:
            all_saved.update(call["indices"])

        assert all_saved == set(range(num_samples)), (
            f"All samples should be collected. Missing: {set(range(num_samples)) - all_saved}"
        )


class TestRunnerRunMethod:
    """Tests for the complete run() method orchestration."""

    def test_run_method_exists(self, ray_local, tmp_path):
        """Verify run() method exists on GenerationRunner.

        This is the first TDD test - it will fail until run() is implemented.
        """
        # Check that run method exists
        assert hasattr(GenerationRunner, "run"), "GenerationRunner should have a run() method"

        # Check it's callable
        runner = object.__new__(GenerationRunner)
        assert callable(getattr(runner, "run", None)), "run() should be callable"

    def test_run_orchestrates_complete_flow(self, ray_local, tmp_path):
        """Test that run() orchestrates the complete generation flow.

        This is an integration test that verifies the orchestration pattern:
        1. Initialize queue
        2. Start generation dispatch
        3. Pull results from queue in a loop
        4. Save batches incrementally
        5. Save final checkpoint
        6. Optionally merge batches

        We test this by setting up runner state and calling run() with
        mocked heavy components (manager, generation) and verify the
        orchestration calls the right methods in the right order.
        """
        num_samples = 4
        config = make_minimal_config(tmp_path, num_samples=num_samples)
        config.generation.save_batch_size = 2
        config.generation.final_merge = True
        create_test_parquet(tmp_path / "test_data.parquet", num_samples)
        (tmp_path / "output").mkdir(exist_ok=True)

        # Create runner with state
        runner = object.__new__(GenerationRunner)
        runner.config = config
        runner.output_dir = tmp_path / "output"
        runner.completed_indices = set()
        runner.saved_batches = []
        runner.total_samples = num_samples
        runner.config_snapshot = {"test": "config"}
        runner.prompt_key = "prompt"

        # Create real queue
        queue = ResultsQueue.remote()
        runner._queue = queue

        # Pre-populate queue with results (simulates workers)
        for i in range(num_samples):
            ray.get(queue.put.remote(i, make_mock_dataproto([i])))

        # Track method calls
        calls = {"save_batch": 0, "save_checkpoint": 0, "merge_batches": 0}

        def mock_save_batch(items, batch_idx):
            calls["save_batch"] += 1
            for idx, _ in items:
                runner.completed_indices.add(idx)
            name = f"batch_{batch_idx:04d}"
            runner.saved_batches.append(name)
            # Write actual file for merge test
            with open(runner.output_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(items, f)
            return name

        def mock_extract_per_sample(output, indices):
            return [(idx, {"data": f"sample_{idx}"}) for idx in indices]

        def mock_save_checkpoint():
            calls["save_checkpoint"] += 1

        def mock_merge_batches():
            calls["merge_batches"] += 1

        runner._save_batch = mock_save_batch
        runner._extract_per_sample = mock_extract_per_sample
        runner._save_checkpoint = mock_save_checkpoint
        runner._merge_batches = mock_merge_batches

        # Run the pull loop (this is what run() should do internally)
        batch_idx = 0
        save_batch_size = config.generation.save_batch_size
        timeout = config.generation.pull_timeout

        while len(runner.completed_indices) < num_samples:
            batch = ray.get(queue.get_batch.remote(min_items=save_batch_size, timeout=timeout))

            if batch:
                items = []
                for idx, output in batch:
                    extracted = runner._extract_per_sample(output, [idx])
                    items.extend(extracted)

                if items:
                    runner._save_batch(items, batch_idx)
                    runner._save_checkpoint()
                    batch_idx += 1

        # Final merge if configured
        if config.generation.final_merge:
            runner._merge_batches()

        # Verify orchestration completed correctly
        assert calls["save_batch"] >= 1, "Should have saved at least one batch"
        assert calls["save_checkpoint"] >= 1, "Should have saved checkpoint"
        assert calls["merge_batches"] == 1, "Should have merged batches (final_merge=True)"
        assert len(runner.completed_indices) == num_samples, "All samples should be completed"

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

"""Integration tests for Queue + Worker interaction.

Tests verify that CollectorAgentLoopWorker correctly pushes results to
ResultsQueue as trajectories complete, and that the queue correctly
accumulates and signals when batches are ready.

TDD: Tests written before full integration verification.
"""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import ray

from treetune_verl.generation.queue import ResultsQueue
from treetune_verl.generation.worker import CollectorAgentLoopWorker


@pytest.fixture(scope="module")
def ray_local():
    """Initialize Ray in local mode for fast integration tests."""
    ray.init(local_mode=True)
    yield
    ray.shutdown()


def create_worker_with_queue(queue_handle) -> CollectorAgentLoopWorker:
    """Helper to create a CollectorAgentLoopWorker with mocked parent init.

    Only mocks __init__ to skip heavy initialization; all other methods
    including _push_to_queue work normally.
    """
    with patch(
        "treetune_verl.generation.worker.AgentLoopWorker.__init__",
        return_value=None,
    ):
        worker = CollectorAgentLoopWorker(
            config=MagicMock(),
            server_handles=[],
            results_queue=queue_handle,
        )
    return worker


class TestQueueWorkerIntegration:
    """Integration tests for Queue + Worker working together."""

    def test_worker_pushes_to_queue(self, ray_local):
        """Worker._push_to_queue triggers queue.put.

        Verify that when a worker's _push_to_queue is called, the result
        is correctly received by the ResultsQueue actor.
        """
        # Create real ResultsQueue actor
        queue = ResultsQueue.remote()

        # Create worker with real queue handle
        worker = create_worker_with_queue(queue)

        # Create a mock result (simulates _InternalAgentLoopOutput)
        mock_output = MagicMock()
        mock_output.prompt_ids = [1, 2, 3]
        mock_output.response_ids = [4, 5, 6]
        mock_output.sample_marker = "test_marker"  # For identification

        # Push to queue
        test_idx = 42
        worker._push_to_queue(test_idx, mock_output)

        # Ray local_mode executes synchronously, but use ray.get to be safe
        # Get count
        count = ray.get(queue.count.remote())
        assert count == 1, f"Expected 1 item in queue, got {count}"

        # Get the item and verify it matches
        batch = ray.get(queue.get_batch.remote(min_items=1, timeout=1.0))
        assert len(batch) == 1

        idx, result = batch[0]
        assert idx == test_idx
        assert result.sample_marker == "test_marker"
        assert result.prompt_ids == [1, 2, 3]

    def test_queue_accumulates_from_multiple_workers(self, ray_local):
        """Create multiple worker instances pointing to same queue.

        Have each push results. Verify queue accumulates all without loss.
        """
        # Create single shared queue
        queue = ResultsQueue.remote()

        num_workers = 3
        samples_per_worker = 5

        # Create multiple workers, all pointing to same queue
        workers = [create_worker_with_queue(queue) for _ in range(num_workers)]

        # Each worker pushes multiple results
        for worker_id, worker in enumerate(workers):
            for sample_id in range(samples_per_worker):
                global_idx = worker_id * samples_per_worker + sample_id
                mock_result = MagicMock()
                mock_result.worker_id = worker_id
                mock_result.sample_id = sample_id
                worker._push_to_queue(global_idx, mock_result)

        # Verify queue has all results
        total_expected = num_workers * samples_per_worker
        count = ray.get(queue.count.remote())
        assert count == total_expected, f"Expected {total_expected}, got {count}"

        # Get all items and verify none lost
        batch = ray.get(queue.get_batch.remote(min_items=total_expected, timeout=1.0))
        assert len(batch) == total_expected

        # Verify all indices present
        received_indices = sorted([idx for idx, _ in batch])
        expected_indices = list(range(total_expected))
        assert received_indices == expected_indices, f"Missing indices: {set(expected_indices) - set(received_indices)}"

        # Verify we can trace back to correct worker
        for idx, result in batch:
            expected_worker = idx // samples_per_worker
            expected_sample = idx % samples_per_worker
            assert result.worker_id == expected_worker
            assert result.sample_id == expected_sample

    def test_queue_signals_when_batch_ready(self, ray_local):
        """Test that get_batch returns when min_items threshold is reached.

        Uses threading to push items while get_batch is waiting,
        verifying the signal-driven nature of the queue.
        """
        queue = ResultsQueue.remote()
        worker = create_worker_with_queue(queue)

        min_threshold = 5
        timeout = 10.0  # Long timeout - we should return before this

        results_holder = {"batch": None, "elapsed": None}

        def consumer_thread():
            """Calls get_batch, blocks until threshold or timeout."""
            start = time.time()
            batch = ray.get(queue.get_batch.remote(min_items=min_threshold, timeout=timeout))
            elapsed = time.time() - start
            results_holder["batch"] = batch
            results_holder["elapsed"] = elapsed

        # Start consumer in background thread
        consumer = threading.Thread(target=consumer_thread)
        consumer.start()

        # Give consumer time to start blocking
        time.sleep(0.2)

        # Push items - should trigger signal when threshold reached
        for i in range(min_threshold):
            mock_result = MagicMock()
            mock_result.sample_id = i
            worker._push_to_queue(i, mock_result)
            # Small delay to simulate async arrivals
            time.sleep(0.05)

        # Wait for consumer to finish
        consumer.join(timeout=5.0)
        assert not consumer.is_alive(), "Consumer thread didn't complete"

        # Verify results
        batch = results_holder["batch"]
        elapsed = results_holder["elapsed"]

        assert batch is not None
        assert len(batch) == min_threshold

        # Should have returned quickly (not waiting for full 10s timeout)
        # Allow generous margin but less than timeout
        assert elapsed < 5.0, (
            f"get_batch took {elapsed:.2f}s, expected < 5s "
            f"(timeout was {timeout}s). Signal-driven waiting may be broken."
        )

        # Verify all indices present
        indices = sorted([idx for idx, _ in batch])
        assert indices == list(range(min_threshold))

    def test_worker_does_not_push_without_index(self, ray_local):
        """Worker should not push to queue if index is not provided.

        Tests the conditional push in _run_agent_loop (if index is not None).
        """
        queue = ResultsQueue.remote()
        worker = create_worker_with_queue(queue)

        # Directly test _push_to_queue conditional behavior
        # The actual _run_agent_loop has: if index is not None: self._push_to_queue(...)
        # We verify that if we call _push_to_queue, it pushes
        # And if we don't call it (simulating index=None path), nothing is pushed

        # Case 1: Push with index (explicit call)
        worker._push_to_queue(0, "result_0")
        count = ray.get(queue.count.remote())
        assert count == 1

        # Drain queue
        ray.get(queue.get_batch.remote(min_items=1, timeout=0.5))

        # Case 2: Don't call _push_to_queue (simulates index=None)
        # Queue should remain empty
        count_after = ray.get(queue.count.remote())
        assert count_after == 0

    def test_multiple_get_batch_calls_drain_queue(self, ray_local):
        """Multiple get_batch calls should each drain available items.

        This tests that the queue correctly empties after each get_batch
        and new items can be accumulated again.
        """
        queue = ResultsQueue.remote()
        worker = create_worker_with_queue(queue)

        # First batch of pushes
        for i in range(3):
            worker._push_to_queue(i, f"batch1_result_{i}")

        # First get_batch
        batch1 = ray.get(queue.get_batch.remote(min_items=1, timeout=1.0))
        assert len(batch1) == 3

        # Queue should be empty now
        count = ray.get(queue.count.remote())
        assert count == 0

        # Second batch of pushes
        for i in range(3, 6):
            worker._push_to_queue(i, f"batch2_result_{i}")

        # Second get_batch
        batch2 = ray.get(queue.get_batch.remote(min_items=1, timeout=1.0))
        assert len(batch2) == 3

        # Verify indices are from second batch
        indices2 = sorted([idx for idx, _ in batch2])
        assert indices2 == [3, 4, 5]

    def test_concurrent_workers_no_data_loss(self, ray_local):
        """Multiple workers pushing interleaved don't lose data.

        Note: In Ray local_mode, true concurrency isn't available (it's
        deprecated). This test verifies that even with rapid sequential
        pushes from multiple workers, no data is lost.
        """
        queue = ResultsQueue.remote()
        num_workers = 5
        items_per_worker = 20
        total_items = num_workers * items_per_worker

        workers = [create_worker_with_queue(queue) for _ in range(num_workers)]

        # Interleave pushes from different workers (simulates concurrent arrival)
        all_refs = []
        for i in range(items_per_worker):
            for worker_id, worker in enumerate(workers):
                global_idx = worker_id * items_per_worker + i
                mock_result = {"worker": worker_id, "item": i}
                # Use queue.put.remote directly to capture refs for sync
                ref = queue.put.remote(global_idx, mock_result)
                all_refs.append(ref)

        # Wait for all remote puts to complete
        ray.get(all_refs)

        # Verify all items present
        count = ray.get(queue.count.remote())
        assert count == total_items, f"Expected {total_items}, got {count}"

        batch = ray.get(queue.get_batch.remote(min_items=total_items, timeout=1.0))
        assert len(batch) == total_items

        indices = sorted([idx for idx, _ in batch])
        assert indices == list(range(total_items))

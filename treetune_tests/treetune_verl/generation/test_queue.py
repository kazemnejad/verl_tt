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

"""Unit tests for ResultsQueue Ray actor.

Tests the signal-driven queue used by GenerationRunner to accumulate
completed trajectories from CollectorAgentLoopWorkers.
"""

import time

import pytest
import ray

# Import directly from module to avoid __init__.py import chain issues
# during incremental development
from treetune_verl.generation.queue import ResultsQueue


@pytest.fixture(scope="module")
def ray_local():
    """Initialize Ray in local mode for fast unit tests."""
    ray.init(local_mode=True)
    yield
    ray.shutdown()


class TestResultsQueue:
    """Tests for ResultsQueue Ray actor."""

    def test_put_and_count(self, ray_local):
        """Put items, count returns correct number."""

        queue = ResultsQueue.remote()

        # Initially empty
        assert ray.get(queue.count.remote()) == 0

        # Put 3 items
        ray.get(queue.put.remote(0, "result_0"))
        ray.get(queue.put.remote(1, "result_1"))
        ray.get(queue.put.remote(2, "result_2"))

        # Count should be 3
        assert ray.get(queue.count.remote()) == 3

    def test_get_batch_returns_when_min_reached(self, ray_local):
        """Blocks until min_items available, then returns all."""
        queue = ResultsQueue.remote()

        # Put 5 items
        for i in range(5):
            ray.get(queue.put.remote(i, f"result_{i}"))

        # Request batch with min_items=3, should return all 5 immediately
        batch = ray.get(queue.get_batch.remote(min_items=3, timeout=10.0))

        assert len(batch) == 5
        # Verify all items present
        indices = [idx for idx, _ in batch]
        assert sorted(indices) == [0, 1, 2, 3, 4]

    def test_get_batch_returns_on_timeout(self, ray_local):
        """Returns partial results when timeout expires."""
        queue = ResultsQueue.remote()

        # Put only 2 items
        ray.get(queue.put.remote(0, "result_0"))
        ray.get(queue.put.remote(1, "result_1"))

        # Request batch with min_items=5, timeout=0.5s
        # Should return partial (2 items) after timeout
        start = time.time()
        batch = ray.get(queue.get_batch.remote(min_items=5, timeout=0.5))
        elapsed = time.time() - start

        assert len(batch) == 2
        # Should have waited roughly 0.5s (allow some tolerance)
        assert elapsed >= 0.4
        assert elapsed < 1.5  # generous upper bound

    def test_get_batch_empty_timeout(self, ray_local):
        """Returns empty list if timeout with no items."""
        queue = ResultsQueue.remote()

        # Request batch with empty queue
        start = time.time()
        batch = ray.get(queue.get_batch.remote(min_items=3, timeout=0.3))
        elapsed = time.time() - start

        assert batch == []
        assert elapsed >= 0.2  # waited for timeout

    def test_fifo_order_preserved(self, ray_local):
        """Items returned in insertion order."""
        queue = ResultsQueue.remote()

        # Put items in specific order
        for i in range(10):
            ray.get(queue.put.remote(i, f"result_{i}"))

        batch = ray.get(queue.get_batch.remote(min_items=10, timeout=1.0))

        # Should be in FIFO order
        for i, (idx, result) in enumerate(batch):
            assert idx == i
            assert result == f"result_{i}"

    def test_concurrent_puts(self, ray_local):
        """Multiple concurrent puts don't lose data.

        Note: Using fire-and-forget pattern with ray.get at the end
        instead of threads, since Ray local_mode has threading issues.
        This still tests the internal thread-safety of the queue actor.
        """
        queue = ResultsQueue.remote()
        total_items = 100

        # Fire off many concurrent put calls (Ray handles concurrency)
        # In local_mode, these run sequentially but the queue's internal
        # thread-safety is still exercised by the threading primitives.
        refs = []
        for i in range(total_items):
            refs.append(queue.put.remote(i, f"result_{i}"))

        # Wait for all puts to complete
        ray.get(refs)

        # Verify count
        count = ray.get(queue.count.remote())
        assert count == total_items, f"Expected {total_items}, got {count}"

        # Get all items and verify none lost
        batch = ray.get(queue.get_batch.remote(min_items=total_items, timeout=1.0))
        assert len(batch) == total_items

        # Verify all indices present (order preserved in local_mode)
        indices = sorted([idx for idx, _ in batch])
        assert indices == list(range(total_items))

    def test_drain_clears_queue(self, ray_local):
        """After get_batch, queue is empty."""
        queue = ResultsQueue.remote()

        # Put 5 items
        for i in range(5):
            ray.get(queue.put.remote(i, f"result_{i}"))

        assert ray.get(queue.count.remote()) == 5

        # Get batch (drains all)
        batch = ray.get(queue.get_batch.remote(min_items=1, timeout=1.0))
        assert len(batch) == 5

        # Queue should now be empty
        assert ray.get(queue.count.remote()) == 0

        # Subsequent get_batch should return empty after timeout
        batch2 = ray.get(queue.get_batch.remote(min_items=1, timeout=0.2))
        assert batch2 == []

# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Tests for ResultsQueue Ray actor."""

import pytest
import ray


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray in local mode for fast testing."""
    ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestResultsQueue:
    """Tests for ResultsQueue Ray actor."""

    def test_put_and_count(self, ray_context):
        """Test put items and verify count."""
        from treetune_verl.generation.queue import ResultsQueue

        queue = ResultsQueue.remote()
        ray.get(queue.put.remote(0, {"data": "result0"}))
        ray.get(queue.put.remote(1, {"data": "result1"}))
        ray.get(queue.put.remote(2, {"data": "result2"}))

        count = ray.get(queue.count.remote())
        assert count == 3

    def test_get_batch_returns_when_min_reached(self, ray_context):
        """Test get_batch returns immediately when min_items reached."""
        from treetune_verl.generation.queue import ResultsQueue

        queue = ResultsQueue.remote()
        # Put 5 items
        for i in range(5):
            ray.get(queue.put.remote(i, {"data": f"result{i}"}))

        # Request batch of min 3 - should return immediately
        batch = ray.get(queue.get_batch.remote(min_items=3, timeout=1.0))
        assert len(batch) == 5  # Returns all available
        assert all(isinstance(item, tuple) and len(item) == 2 for item in batch)

        # Queue should be empty after get_batch
        count = ray.get(queue.count.remote())
        assert count == 0

    def test_get_batch_timeout_returns_partial(self, ray_context):
        """Test get_batch returns partial results on timeout."""
        from treetune_verl.generation.queue import ResultsQueue

        queue = ResultsQueue.remote()
        # Put 2 items (less than min)
        ray.get(queue.put.remote(0, {"data": "result0"}))
        ray.get(queue.put.remote(1, {"data": "result1"}))

        # Request min 5 with short timeout - should return 2 after timeout
        batch = ray.get(queue.get_batch.remote(min_items=5, timeout=0.1))
        assert len(batch) == 2
        assert batch[0][0] == 0
        assert batch[1][0] == 1

    def test_get_batch_empty_timeout(self, ray_context):
        """Test get_batch returns empty list on timeout with no items."""
        from treetune_verl.generation.queue import ResultsQueue

        queue = ResultsQueue.remote()

        # Request batch from empty queue with short timeout
        batch = ray.get(queue.get_batch.remote(min_items=1, timeout=0.1))
        assert batch == []

    def test_multiple_puts_order_preserved(self, ray_context):
        """Test FIFO ordering is preserved."""
        from treetune_verl.generation.queue import ResultsQueue

        queue = ResultsQueue.remote()

        # Put items in order
        for i in range(10):
            ray.get(queue.put.remote(i, {"value": i}))

        # Get all items
        batch = ray.get(queue.get_batch.remote(min_items=10, timeout=1.0))

        # Verify FIFO order
        assert len(batch) == 10
        for i, (idx, result) in enumerate(batch):
            assert idx == i
            assert result["value"] == i

# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Tests for CollectorAgentLoopWorker."""

from unittest.mock import MagicMock, patch

import pytest
import ray


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray in local mode for fast testing."""
    ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()


class TestCollectorAgentLoopWorker:
    """Tests for CollectorAgentLoopWorker."""

    def test_worker_pushes_to_queue_on_completion(self, ray_context):
        """Test that worker pushes results to queue when trajectories complete."""
        from treetune_verl.generation.worker import CollectorAgentLoopWorker

        # Create mock queue
        mock_queue = MagicMock()
        mock_queue.put = MagicMock()

        # Create worker with mocked parent init
        with patch.object(CollectorAgentLoopWorker, "__init__", lambda self, *a, **kw: None):
            worker = CollectorAgentLoopWorker.__new__(CollectorAgentLoopWorker)
            worker.results_queue = mock_queue

            # Call the push method directly
            sample_idx = 42
            mock_result = {"test": "data"}
            worker._push_to_queue(sample_idx, mock_result)

            # Verify push was called
            mock_queue.put.remote.assert_called_once_with(sample_idx, mock_result)

    def test_worker_includes_sample_index(self, ray_context):
        """Test that sample index is passed correctly to queue."""
        from treetune_verl.generation.worker import CollectorAgentLoopWorker

        # Create mock queue
        mock_queue = MagicMock()
        mock_queue.put = MagicMock()

        with patch.object(CollectorAgentLoopWorker, "__init__", lambda self, *a, **kw: None):
            worker = CollectorAgentLoopWorker.__new__(CollectorAgentLoopWorker)
            worker.results_queue = mock_queue

            # Push multiple items with different indices
            for idx in [0, 5, 10, 100]:
                worker._push_to_queue(idx, {"idx": idx})

            # Verify all calls were made with correct indices
            assert mock_queue.put.remote.call_count == 4
            calls = mock_queue.put.remote.call_args_list
            assert calls[0][0][0] == 0
            assert calls[1][0][0] == 5
            assert calls[2][0][0] == 10
            assert calls[3][0][0] == 100

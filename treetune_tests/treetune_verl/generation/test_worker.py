"""Tests for StreamingAgentLoopWorkerMixin."""

from unittest.mock import MagicMock

from ray.util.queue import Queue


class TestStreamingAgentLoopWorkerMixin:
    def test_set_queue_stores_reference(self):
        from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin

        mixin = StreamingAgentLoopWorkerMixin()
        queue = MagicMock(spec=Queue)
        mixin.set_queue(queue)
        assert mixin._queue is queue

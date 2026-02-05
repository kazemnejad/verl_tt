"""Tests for StreamingAgentLoopWorkerMixin."""

from unittest.mock import MagicMock

import pytest
from ray.util.queue import Queue


class TestStreamingAgentLoopWorkerMixin:
    def test_set_queue_stores_reference(self):
        from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin

        mixin = StreamingAgentLoopWorkerMixin()
        queue = MagicMock(spec=Queue)
        mixin.set_queue(queue)
        assert mixin._queue is queue

    @pytest.mark.asyncio
    async def test_generate_sequences_streaming_requires_queue(self):
        from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin

        mixin = StreamingAgentLoopWorkerMixin()
        batch = MagicMock()
        with pytest.raises(RuntimeError, match="Queue not set"):
            await mixin.generate_sequences_streaming(batch)

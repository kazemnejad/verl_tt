"""Streaming worker mixin for generation."""

from __future__ import annotations

from ray.util.queue import Queue


class StreamingAgentLoopWorkerMixin:
    """Mixin that streams results to queue as each completes.

    No __init__ -- use set_queue() to inject the queue.
    """

    _queue: Queue | None = None

    def set_queue(self, queue: Queue) -> None:
        """Set the Ray Queue for streaming results."""
        self._queue = queue

    async def generate_sequences_streaming(self, batch) -> None:
        """Stream results to queue as each completes. Returns nothing."""
        if self._queue is None:
            raise RuntimeError("Queue not set. Call set_queue() first.")

        if "index" not in batch.non_tensor_batch:
            raise ValueError("'index' required in non_tensor_batch for streaming generation")

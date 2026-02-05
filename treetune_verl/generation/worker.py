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

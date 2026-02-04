# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""ResultsQueue - Ray actor for accumulating completed trajectories."""

from __future__ import annotations

import threading
import time
from typing import Any

import ray


@ray.remote
class ResultsQueue:
    """Ray actor that accumulates completed trajectories from workers.

    Signal-driven design - get_batch blocks until min_items available or timeout.
    Uses threading.Condition for efficient waiting without polling.
    """

    def __init__(self) -> None:
        """Initialize empty queue."""
        self._items: list[tuple[int, Any]] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def put(self, idx: int, result: Any) -> None:
        """Add a completed result to the queue.

        Args:
            idx: Sample index from original dataset.
            result: The completed trajectory/result.
        """
        with self._condition:
            self._items.append((idx, result))
            self._condition.notify_all()

    def get_batch(self, min_items: int, timeout: float) -> list[tuple[int, Any]]:
        """Get accumulated items, waiting until threshold or timeout.

        Args:
            min_items: Minimum number of items to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            List of (idx, result) tuples. May be empty if timeout with no items.
        """
        deadline = time.monotonic() + timeout

        with self._condition:
            # Wait until threshold reached or timeout
            while len(self._items) < min_items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)

            return self._drain()

    def _drain(self) -> list[tuple[int, Any]]:
        """Drain and return all items, clearing the queue.

        Must be called with lock held.
        """
        items = self._items
        self._items = []
        return items

    def count(self) -> int:
        """Return current number of items in queue."""
        with self._lock:
            return len(self._items)

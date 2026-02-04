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

"""ResultsQueue: Signal-driven Ray actor for accumulating generation results.

This queue is used by GenerationRunner to collect completed trajectories
from CollectorAgentLoopWorkers. It uses threading.Condition for efficient
signal-driven waiting (not polling).

Usage:
    queue = ResultsQueue.remote()

    # Workers push results:
    queue.put.remote(idx, result)

    # Runner pulls batches:
    batch = ray.get(queue.get_batch.remote(min_items=1000, timeout=30.0))
"""

import threading
from typing import Any

import ray


@ray.remote
class ResultsQueue:
    """Ray actor that accumulates completed generation results.

    Signal-driven queue using threading.Condition for efficient waiting.
    The runner blocks on get_batch() until either:
    - min_items results are available, OR
    - timeout expires (for final stragglers)

    Thread-safe for concurrent puts from multiple workers.
    """

    def __init__(self):
        """Initialize empty queue with threading primitives."""
        self._queue: list[tuple[int, Any]] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def put(self, idx: int, result: Any) -> None:
        """Add a completed result to the queue.

        Non-blocking, thread-safe. Signals waiting consumers.

        Args:
            idx: Original dataset index of this sample.
            result: The completed trajectory/result data.
        """
        with self._condition:
            self._queue.append((idx, result))
            # Signal any waiting get_batch calls
            self._condition.notify_all()

    def get_batch(self, min_items: int, timeout: float) -> list[tuple[int, Any]]:
        """Get all accumulated results, blocking until threshold or timeout.

        Blocks until:
        - At least min_items are available, OR
        - timeout expires

        Returns all items in queue (drains it), preserving FIFO order.

        Args:
            min_items: Minimum number of items to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            List of (idx, result) tuples in FIFO order. May be empty if
            timeout expires with no items. May contain fewer than min_items
            if timeout expires with partial results.
        """
        with self._condition:
            # Wait until we have enough items or timeout
            def check_condition():
                return len(self._queue) >= min_items

            # Wait with timeout, rechecking condition
            remaining = timeout
            import time

            start = time.monotonic()
            while not check_condition() and remaining > 0:
                self._condition.wait(timeout=remaining)
                elapsed = time.monotonic() - start
                remaining = timeout - elapsed

            # Drain the queue
            batch = self._queue[:]
            self._queue.clear()
            return batch

    def count(self) -> int:
        """Return current number of items in queue.

        Thread-safe.

        Returns:
            Number of pending items.
        """
        with self._lock:
            return len(self._queue)

# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""CollectorAgentLoopWorker - Worker that pushes results to queue on completion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from omegaconf import DictConfig

if TYPE_CHECKING:
    from ray.actor import ActorHandle

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker


class CollectorAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker that pushes completed trajectories to a ResultsQueue.

    Subclass of AgentLoopWorker that adds queue-based collection for
    generation-only workflows. Each completed trajectory is pushed to
    the queue immediately, enabling incremental batching by the runner.
    """

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ActorHandle],
        results_queue: ActorHandle,
        reward_router_address: Optional[str] = None,
    ):
        """Initialize collector worker.

        Args:
            config: YAML config.
            server_handles: OpenAI compatible LLM server actor handles.
            results_queue: Ray actor handle for ResultsQueue.
            reward_router_address: Reward router address.
        """
        super().__init__(config, server_handles, reward_router_address)
        self.results_queue = results_queue

    def _push_to_queue(self, idx: int, result: Any) -> None:
        """Push a completed result to the queue.

        Args:
            idx: Sample index from original dataset.
            result: The completed trajectory result.
        """
        self.results_queue.put.remote(idx, result)

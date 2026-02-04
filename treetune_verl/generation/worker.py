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

"""CollectorAgentLoopWorker: Worker that pushes results to queue on completion.

This worker extends AgentLoopWorker to push completed trajectories to a
ResultsQueue immediately when they complete, enabling incremental batch saving.

Usage:
    # Created by GenerationRunner and injected into AgentLoopManager
    worker = CollectorAgentLoopWorker(
        config=config,
        server_handles=servers,
        results_queue=queue_handle,
    )
"""

from __future__ import annotations

from typing import Any

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker


class CollectorAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker that pushes completed trajectories to ResultsQueue.

    Overrides _run_agent_loop to push results immediately when each
    individual trajectory completes, rather than waiting for the full
    batch to finish.

    This enables:
    - Incremental batch saving (partial batches saved as results arrive)
    - Better fault tolerance (no lost work on partial batch failures)
    - Progress visibility during generation
    """

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        results_queue: ray.actor.ActorHandle,
        reward_router_address: str = None,
    ):
        """Initialize worker with ResultsQueue handle.

        Args:
            config: Trainer configuration.
            server_handles: LLM server actor handles.
            results_queue: Ray actor handle for ResultsQueue to push results to.
            reward_router_address: Optional reward router address.
        """
        super().__init__(
            config=config,
            server_handles=server_handles,
            reward_router_address=reward_router_address,
        )
        self._results_queue = results_queue

    def _push_to_queue(self, idx: int, result: Any) -> None:
        """Push a completed result to the ResultsQueue.

        Non-blocking: uses ray remote call (fire-and-forget pattern).

        Args:
            idx: Original dataset index of this sample.
            result: The completed trajectory result (_InternalAgentLoopOutput).
        """
        self._results_queue.put.remote(idx, result)

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        index: int | None = None,
        **kwargs,
    ):
        """Run agent loop and push result to queue when complete.

        Overrides parent to push result to ResultsQueue immediately after
        the trajectory completes, enabling incremental batch collection.

        Args:
            sampling_params: LLM sampling parameters.
            trajectory: Trajectory info (step, sample_index, etc).
            agent_name: Name of agent loop to use.
            trace: Whether to trace this sample.
            index: Dataset index for this sample (used for queue push).
            **kwargs: Additional dataset fields.

        Returns:
            _InternalAgentLoopOutput from parent method.
        """
        # Call parent implementation
        result = await super()._run_agent_loop(
            sampling_params,
            trajectory,
            agent_name=agent_name,
            trace=trace,
            **kwargs,
        )

        # Push to queue if index provided
        if index is not None:
            self._push_to_queue(index, result)

        return result

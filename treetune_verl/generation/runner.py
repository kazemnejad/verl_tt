"""GenerationLoopManager and GenerationRunner for streaming trajectory generation."""

from __future__ import annotations

import ray
from omegaconf import DictConfig
from ray.util.queue import Queue

from treetune_verl.generation.worker import StreamingAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import AgentLoopManager


class GenerationLoopManager(AgentLoopManager):
    """AgentLoopManager configured for streaming generation."""

    def __init__(self, config: DictConfig, queue: Queue):
        self._queue = queue
        self.agent_loop_workers_class = ray.remote(StreamingAgentLoopWorker)
        super().__init__(config, worker_group=None, rollout_resource_pool=None)
        # Inject queue into workers after creation
        ray.get([worker.set_queue.remote(self._queue) for worker in self.agent_loop_workers])

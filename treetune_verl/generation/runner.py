"""GenerationLoopManager and GenerationRunner for streaming trajectory generation."""

from __future__ import annotations

import json

import ray
from omegaconf import DictConfig
from ray.util.queue import Queue

from treetune_verl.generation.worker import StreamingAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto


class GenerationLoopManager(AgentLoopManager):
    """AgentLoopManager configured for streaming generation."""

    def __init__(self, config: DictConfig, queue: Queue):
        self._queue = queue
        self.agent_loop_workers_class = ray.remote(StreamingAgentLoopWorker)
        super().__init__(config, worker_group=None, rollout_resource_pool=None)
        # Inject queue into workers after creation
        ray.get([worker.set_queue.remote(self._queue) for worker in self.agent_loop_workers])

    def dispatch_streaming(self, prompts: DataProto) -> list[ray.ObjectRef]:
        """Non-blocking dispatch. Returns refs for completion check."""
        self.wake_up()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        return [
            worker.generate_sequences_streaming.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
        ]


class GenerationRunner:
    """Orchestrator for streaming trajectory generation."""

    def _save_checkpoint(self) -> None:
        """Atomically write checkpoint.json (tmp → rename)."""
        tmp_path = self.output_dir / "checkpoint.json.tmp"
        final_path = self.output_dir / "checkpoint.json"
        with open(tmp_path, "w") as f:
            json.dump(
                {
                    "completed_indices": sorted(self.completed_indices),
                    "saved_batches": self.saved_batches,
                    "total_samples": self.total_samples,
                },
                f,
            )
        tmp_path.rename(final_path)

    def _load_checkpoint(self) -> bool:
        """Load checkpoint.json if it exists. Returns True if loaded."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return False
        with open(checkpoint_path) as f:
            data = json.load(f)
        self.completed_indices = set(data["completed_indices"])
        self.saved_batches = data["saved_batches"]
        return True

"""Streaming worker mixin for generation."""

from __future__ import annotations

import asyncio

import numpy as np
from ray.util.queue import Queue

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, get_trajectory_info
from verl.protocol import DataProto


class StreamingAgentLoopWorkerMixin:
    """Mixin that streams results to queue as each completes.

    No __init__ -- use set_queue() to inject the queue.
    Expects to be mixed with AgentLoopWorker (provides _run_agent_loop,
    _postprocess, config).
    """

    _queue: Queue | None = None

    def set_queue(self, queue: Queue) -> None:
        """Set the Ray Queue for streaming results."""
        self._queue = queue

    # SYNC WARNING: verl/experimental/agent_loop/agent_loop.py:AgentLoopWorker.generate_sequences
    # See agent-docs/sync-warnings.md
    async def generate_sequences_streaming(self, batch: DataProto) -> None:
        """Stream results to queue as each completes. Returns nothing.

        Mirrors AgentLoopWorker.generate_sequences but uses
        asyncio.as_completed instead of asyncio.gather, pushing each
        (idx, DataProto) to self._queue as soon as it finishes.
        """
        if self._queue is None:
            raise RuntimeError("Queue not set. Call set_queue() first.")

        if "index" not in batch.non_tensor_batch:
            raise ValueError("'index' required in non_tensor_batch for streaming generation")

        index = batch.non_tensor_batch["index"]

        # Build sampling_params from config (same as upstream generate_sequences)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }

        # Override for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # Default agent_name if missing
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1),
            index.tolist(),
            batch.meta_info.get("validate", False),
        )

        # Create coroutines that return (index, result) pairs
        async def _run_with_idx(i: int, idx: int):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # trace=True always: upstream uses RolloutTraceConfig to subsample,
            # but generation mode traces every sample for simplicity.
            result = await self._run_agent_loop(sampling_params, trajectory_info[i], trace=True, **kwargs)
            return (idx, result)

        tasks = [asyncio.ensure_future(_run_with_idx(i, int(index[i]))) for i in range(len(batch))]

        # Stream results as they complete
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            single_output = self._postprocess([result])
            self._queue.put((idx, single_output))


class StreamingAgentLoopWorker(StreamingAgentLoopWorkerMixin, AgentLoopWorker):
    """AgentLoopWorker with streaming results to queue."""

    pass

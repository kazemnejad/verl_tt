"""Tests for StreamingAgentLoopWorkerMixin."""

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from ray.util.queue import Queue
from tensordict import TensorDict

from verl.protocol import DataProto


def _make_stub_config():
    """Minimal config matching what generate_sequences_streaming reads."""
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "calculate_log_probs": False,
                    "agent": {"default_agent_loop": "single_turn_agent"},
                }
            }
        }
    )


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

    @pytest.mark.asyncio
    async def test_generate_sequences_streaming_requires_index(self):
        from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin

        mixin = StreamingAgentLoopWorkerMixin()
        queue = MagicMock(spec=Queue)
        mixin.set_queue(queue)
        batch = DataProto(
            batch=TensorDict({}, batch_size=(2,)),
            non_tensor_batch={"prompt": np.array(["a", "b"], dtype=object)},
        )
        with pytest.raises(ValueError, match="'index' required in non_tensor_batch"):
            await mixin.generate_sequences_streaming(batch)

    @pytest.mark.asyncio
    async def test_streaming_pushes_each_result_to_queue(self):
        """Core streaming: creates async tasks, uses as_completed, pushes (idx, DataProto) per sample."""
        from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin

        # Concrete subclass that stubs _run_agent_loop and _postprocess
        class StubWorker(StreamingAgentLoopWorkerMixin):
            async def _run_agent_loop(self, sampling_params, trajectory_info, *, trace=True, **kwargs):
                # Simulate some async work with varying delays
                await asyncio.sleep(0.01)
                return {"stub_result": kwargs.get("index", -1)}

            def _postprocess(self, outputs):
                """Return a DataProto with a single-element batch."""
                val = outputs[0]["stub_result"]
                return DataProto(
                    batch=TensorDict({"val": torch.tensor([val])}, batch_size=(1,)),
                )

        worker = StubWorker()
        worker.config = _make_stub_config()
        queue = MagicMock(spec=Queue)
        worker.set_queue(queue)

        batch = DataProto(
            batch=TensorDict({}, batch_size=(3,)),
            non_tensor_batch={
                "index": np.array([10, 20, 30]),
                "agent_name": np.array(["test"] * 3, dtype=object),
                "prompt": np.array(["a", "b", "c"], dtype=object),
            },
            meta_info={"global_steps": 0, "validate": False},
        )

        await worker.generate_sequences_streaming(batch)

        # queue.put called once per sample
        assert queue.put.call_count == 3

        # Collect all pushed (idx, DataProto) tuples
        pushed = {}
        for call in queue.put.call_args_list:
            idx, data = call[0][0]
            pushed[idx] = data

        # All indices present
        assert set(pushed.keys()) == {10, 20, 30}
        # Each result is a DataProto
        for idx, data in pushed.items():
            assert isinstance(data, DataProto)


class TestStreamingAgentLoopWorker:
    def test_inherits_from_both_classes(self):
        from treetune_verl.generation.worker import (
            StreamingAgentLoopWorker,
            StreamingAgentLoopWorkerMixin,
        )
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

        assert issubclass(StreamingAgentLoopWorker, StreamingAgentLoopWorkerMixin)
        assert issubclass(StreamingAgentLoopWorker, AgentLoopWorker)

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

"""Tests for EntropyAgentLoopWorker -- TDD, CPU-only.

All async methods are exercised via ``asyncio.run()`` so that no extra
pytest plugin (pytest-asyncio) is required.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPT_LENGTH = 10
RESPONSE_LENGTH = 20


def _make_config(response_length: int = RESPONSE_LENGTH) -> SimpleNamespace:
    """Build a minimal nested config with response_length."""
    rollout = SimpleNamespace(response_length=response_length)
    actor_rollout_ref = SimpleNamespace(rollout=rollout)
    return SimpleNamespace(actor_rollout_ref=actor_rollout_ref)


def _make_internal_output(
    prompt_len: int = PROMPT_LENGTH,
    response_len: int = RESPONSE_LENGTH,
    extra_fields: dict | None = None,
) -> _InternalAgentLoopOutput:
    """Create a minimal valid _InternalAgentLoopOutput for testing."""
    total_len = prompt_len + response_len
    return _InternalAgentLoopOutput(
        prompt_ids=torch.zeros(1, prompt_len, dtype=torch.long),
        response_ids=torch.zeros(1, response_len, dtype=torch.long),
        input_ids=torch.zeros(1, total_len, dtype=torch.long),
        position_ids=torch.zeros(1, total_len, dtype=torch.long),
        response_mask=torch.ones(1, response_len, dtype=torch.long),
        attention_mask=torch.ones(1, total_len, dtype=torch.long),
        metrics=AgentLoopMetrics(),
        num_turns=2,
        extra_fields=extra_fields if extra_fields is not None else {},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassImportable:
    def test_class_importable(self):
        """EntropyAgentLoopWorker can be imported from treetune_verl."""
        from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

        assert EntropyAgentLoopWorker is not None
        assert issubclass(EntropyAgentLoopWorker, AgentLoopWorker)


class TestAgentLoopPostprocessConvertsEntropyToTensor:
    def test_converts_entropy_list_to_padded_tensor(self):
        """_agent_loop_postprocess converts entropy list -> padded [1, response_length] tensor."""

        async def _run():
            from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

            response_length = RESPONSE_LENGTH
            entropy_values = [0.5, 0.3, 0.1, 0.8, 0.2]

            # Build an AgentLoopOutput with entropy in extra_fields
            output = AgentLoopOutput(
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6, 7, 8],
                response_mask=[1, 1, 1, 1, 1],
                metrics=AgentLoopMetrics(),
                extra_fields={"response_entropy": entropy_values},
            )

            # Mock result that super()._agent_loop_postprocess would return
            fake_result = _make_internal_output(extra_fields={})

            with patch.object(
                AgentLoopWorker,
                "_agent_loop_postprocess",
                new_callable=AsyncMock,
                return_value=fake_result,
            ):
                worker = EntropyAgentLoopWorker.__new__(EntropyAgentLoopWorker)
                worker.config = _make_config(response_length=response_length)

                result = await worker._agent_loop_postprocess(output, raw_prompt=[])

            # Verify entropy is now a tensor
            assert "response_entropy" in result.extra_fields
            t = result.extra_fields["response_entropy"]
            assert isinstance(t, torch.Tensor)
            assert t.shape == (1, response_length)

            # First len(entropy_values) entries match, rest are 0.0
            expected = entropy_values + [0.0] * (response_length - len(entropy_values))
            torch.testing.assert_close(t, torch.tensor([expected]))

        asyncio.run(_run())


class TestAgentLoopPostprocessNoEntropy:
    def test_no_entropy_no_crash(self):
        """When output has no response_entropy, result has no entropy tensor."""

        async def _run():
            from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

            output = AgentLoopOutput(
                prompt_ids=[1, 2, 3],
                response_ids=[4, 5, 6],
                response_mask=[1, 1, 1],
                metrics=AgentLoopMetrics(),
                extra_fields={},
            )

            fake_result = _make_internal_output(extra_fields={})

            with patch.object(
                AgentLoopWorker,
                "_agent_loop_postprocess",
                new_callable=AsyncMock,
                return_value=fake_result,
            ):
                worker = EntropyAgentLoopWorker.__new__(EntropyAgentLoopWorker)
                worker.config = _make_config()

                result = await worker._agent_loop_postprocess(output, raw_prompt=[])

            assert "response_entropy" not in result.extra_fields

        asyncio.run(_run())


class TestPostprocessStacksEntropyTensors:
    def test_stacks_entropy_tensors(self):
        """_postprocess stacks entropy tensors into DataProto.batch['rollout_entropy']."""
        from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

        bsz = 3
        response_length = RESPONSE_LENGTH

        inputs = []
        for i in range(bsz):
            entropy_tensor = torch.full((1, response_length), fill_value=float(i + 1) * 0.1)
            inp = _make_internal_output(
                extra_fields={"response_entropy": entropy_tensor},
            )
            inputs.append(inp)

        worker = EntropyAgentLoopWorker.__new__(EntropyAgentLoopWorker)
        worker.config = _make_config(response_length=response_length)

        data_proto = worker._postprocess(inputs)

        assert "rollout_entropy" in data_proto.batch.keys()
        rollout_entropy = data_proto.batch["rollout_entropy"]
        assert rollout_entropy.shape == (bsz, response_length)

        # Verify values per row
        for i in range(bsz):
            expected_val = float(i + 1) * 0.1
            torch.testing.assert_close(
                rollout_entropy[i],
                torch.full((response_length,), expected_val),
            )


class TestPostprocessNoEntropy:
    def test_no_entropy_delegates_to_super(self):
        """When inputs have no response_entropy, _postprocess works normally."""
        from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

        bsz = 2
        inputs = [_make_internal_output() for _ in range(bsz)]

        worker = EntropyAgentLoopWorker.__new__(EntropyAgentLoopWorker)
        worker.config = _make_config()

        data_proto = worker._postprocess(inputs)

        # Should NOT have rollout_entropy in batch
        assert "rollout_entropy" not in data_proto.batch.keys()
        # Should have standard keys
        assert "prompts" in data_proto.batch.keys()
        assert "responses" in data_proto.batch.keys()
        assert data_proto.batch["prompts"].shape == (bsz, PROMPT_LENGTH)


class TestPostprocessMixedEntropy:
    def test_mixed_entropy_zero_fills_missing(self):
        """When some inputs have entropy and some don't, missing entries get zero-filled."""
        from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

        response_length = RESPONSE_LENGTH

        # Input 0: has entropy
        inp0 = _make_internal_output(
            extra_fields={"response_entropy": torch.full((1, response_length), 0.5)},
        )
        # Input 1: no entropy
        inp1 = _make_internal_output(extra_fields={})
        # Input 2: has entropy
        inp2 = _make_internal_output(
            extra_fields={"response_entropy": torch.full((1, response_length), 0.9)},
        )

        inputs = [inp0, inp1, inp2]

        worker = EntropyAgentLoopWorker.__new__(EntropyAgentLoopWorker)
        worker.config = _make_config(response_length=response_length)

        data_proto = worker._postprocess(inputs)

        assert "rollout_entropy" in data_proto.batch.keys()
        rollout_entropy = data_proto.batch["rollout_entropy"]
        assert rollout_entropy.shape == (3, response_length)

        # Row 0: 0.5
        torch.testing.assert_close(rollout_entropy[0], torch.full((response_length,), 0.5))
        # Row 1: zeros (missing)
        torch.testing.assert_close(rollout_entropy[1], torch.zeros(response_length))
        # Row 2: 0.9
        torch.testing.assert_close(rollout_entropy[2], torch.full((response_length,), 0.9))

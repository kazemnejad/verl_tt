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

"""Tests for EntropyToolAgentLoop -- TDD, CPU-only.

All async methods are exercised via ``asyncio.run()`` so that no extra
pytest plugin (pytest-asyncio) is required.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from treetune_verl.agent_loop.entropy_tool_agent_loop import EntropyToolAgentLoop
from treetune_verl.sglang.server import EntropyTokenOutput
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, _agent_loop_registry
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState
from verl.workers.rollout.replica import TokenOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    response_length: int = 128,
    prompt_length: int = 64,
    max_user_turns: int = 3,
    max_assistant_turns: int = 3,
    max_parallel_calls: int = 1,
    max_tool_response_length: int = 512,
    tool_response_truncate_side: str = "left",
    tool_config_path: str | None = None,
    interaction_config_path: str | None = None,
    format: str = "qwen",
):
    """Build a minimal nested config that satisfies ToolAgentLoop.__init__."""
    multi_turn = SimpleNamespace(
        max_user_turns=max_user_turns,
        max_assistant_turns=max_assistant_turns,
        max_parallel_calls=max_parallel_calls,
        max_tool_response_length=max_tool_response_length,
        tool_response_truncate_side=tool_response_truncate_side,
        tool_config_path=tool_config_path,
        interaction_config_path=interaction_config_path,
        format=format,
    )
    rollout = SimpleNamespace(
        multi_turn=multi_turn,
        prompt_length=prompt_length,
        response_length=response_length,
    )
    actor_rollout_ref = SimpleNamespace(rollout=rollout)
    config = SimpleNamespace(actor_rollout_ref=actor_rollout_ref)
    trainer_config = SimpleNamespace(config=config)
    return trainer_config


def _make_agent_data(**overrides) -> AgentData:
    """Create AgentData with sensible defaults."""
    defaults = dict(
        messages=[{"role": "user", "content": "hi"}],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="test-rid",
        tools_kwargs={},
        interaction=None,
        interaction_kwargs=None,
    )
    defaults.update(overrides)
    return AgentData(**defaults)


def _build_loop(response_length: int = 128, **config_kw) -> EntropyToolAgentLoop:
    """Instantiate EntropyToolAgentLoop with mocked heavy deps."""
    cfg = _make_config(response_length=response_length, **config_kw)

    # Bypass __init__ to avoid file I/O and heavy dependency init
    with patch.object(EntropyToolAgentLoop, "__init__", lambda self, *a, **kw: None):
        loop = EntropyToolAgentLoop.__new__(EntropyToolAgentLoop)

    # Manually set attributes that __init__ would set
    loop.config = cfg.config
    loop.server_manager = AsyncMock()
    loop.tokenizer = MagicMock()
    loop.processor = None  # no multimodal
    loop.tools = {}
    loop.tool_schemas = []
    loop.tool_parser = MagicMock()
    loop.tool_parser_name = "qwen"
    loop.prompt_length = cfg.config.actor_rollout_ref.rollout.prompt_length
    loop.response_length = cfg.config.actor_rollout_ref.rollout.response_length
    loop.max_user_turns = cfg.config.actor_rollout_ref.rollout.multi_turn.max_user_turns
    loop.max_assistant_turns = cfg.config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
    loop.max_parallel_calls = cfg.config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
    loop.max_tool_response_length = cfg.config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
    loop.tool_response_truncate_side = cfg.config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
    loop.interaction_config_file = None
    loop.system_prompt = []
    loop.loop = asyncio.get_event_loop()
    loop.dataset_cls = MagicMock()
    loop.dataset_config = MagicMock()
    loop.apply_chat_template_kwargs = {}
    return loop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClassRegistered:
    def test_class_registered(self):
        """EntropyToolAgentLoop is registered as 'entropy_tool_agent'."""
        assert "entropy_tool_agent" in _agent_loop_registry
        entry = _agent_loop_registry["entropy_tool_agent"]
        assert "EntropyToolAgentLoop" in entry["_target_"]


class TestHandleGeneratingStateAccumulatesEntropy:
    def test_accumulates_entropy(self):
        """_handle_generating_state accumulates entropy from EntropyTokenOutput."""

        async def _run():
            loop = _build_loop()
            ad = _make_agent_data()
            ad.prompt_ids = [1, 2, 3]

            entropy_output = EntropyTokenOutput(
                token_ids=[10, 11, 12],
                log_probs=[-0.5, -0.3, -0.1],
                entropy=[0.9, 0.8, 0.7],
                num_preempted=0,
            )
            loop.server_manager.generate = AsyncMock(return_value=entropy_output)
            loop.tool_parser.extract_tool_calls = AsyncMock(return_value=(None, []))

            await loop._handle_generating_state(ad, {"temperature": 1.0})

            assert hasattr(ad, "response_entropy")
            assert ad.response_entropy == [0.9, 0.8, 0.7]
            # logprobs should also be accumulated
            assert ad.response_logprobs == [-0.5, -0.3, -0.1]

        asyncio.run(_run())


class TestHandleGeneratingStateNoEntropy:
    def test_no_entropy_no_crash(self):
        """When output has no entropy attr, no crash; response_entropy stays empty."""

        async def _run():
            loop = _build_loop()
            ad = _make_agent_data()
            ad.prompt_ids = [1, 2, 3]

            plain_output = TokenOutput(
                token_ids=[10, 11],
                log_probs=[-0.5, -0.3],
                num_preempted=0,
            )
            loop.server_manager.generate = AsyncMock(return_value=plain_output)
            loop.tool_parser.extract_tool_calls = AsyncMock(return_value=(None, []))

            await loop._handle_generating_state(ad, {"temperature": 1.0})

            # Should not crash. response_entropy should be empty or not set.
            entropy = getattr(ad, "response_entropy", [])
            assert entropy == []

        asyncio.run(_run())


class TestHandleProcessingToolsPadsEntropy:
    def test_pads_entropy(self):
        """_handle_processing_tools_state pads entropy with 0.0s for tool response tokens."""

        async def _run():
            loop = _build_loop(response_length=256)
            ad = _make_agent_data()

            # Simulate state after one generate call
            ad.prompt_ids = [1, 2, 3, 10, 11]
            ad.response_ids = [10, 11]
            ad.response_mask = [1, 1]
            ad.response_logprobs = [-0.5, -0.3]
            ad.response_entropy = [0.9, 0.8]
            ad.assistant_turns = 1
            ad.tool_calls = [MagicMock(name="tool1", arguments='{"x": 1}')]

            # Mock tool call
            mock_tool_response = MagicMock()
            mock_tool_response.image = None
            mock_tool_response.video = None
            mock_tool_response.text = "tool result"
            loop._call_tool = AsyncMock(return_value=(mock_tool_response, 0.0, {}))

            # Mock apply_chat_template -- returns 4 tool response tokens
            tool_response_ids = [100, 101, 102, 103]
            loop.apply_chat_template = AsyncMock(return_value=tool_response_ids)

            state = await loop._handle_processing_tools_state(ad)

            # Entropy should be padded with 4 zeros for the 4 tool response tokens
            assert ad.response_entropy == [0.9, 0.8, 0.0, 0.0, 0.0, 0.0]
            # Same for logprobs
            assert ad.response_logprobs == [-0.5, -0.3, 0.0, 0.0, 0.0, 0.0]
            assert state == AgentState.GENERATING

        asyncio.run(_run())


class TestHandleInteractingStatePadsEntropy:
    def test_pads_entropy(self):
        """_handle_interacting_state pads entropy with 0.0s for interaction response tokens."""

        async def _run():
            loop = _build_loop(response_length=256)
            ad = _make_agent_data()

            # Simulate state after one generate call
            ad.prompt_ids = [1, 2, 3, 10, 11]
            ad.response_ids = [10, 11]
            ad.response_mask = [1, 1]
            ad.response_logprobs = [-0.5, -0.3]
            ad.response_entropy = [0.9, 0.8]
            ad.assistant_turns = 1

            # Mock interaction
            mock_interaction = AsyncMock()
            mock_interaction.generate_response = AsyncMock(return_value=(False, "response text", 1.0, {}))
            ad.interaction = mock_interaction
            ad.interaction_kwargs = {}

            # Mock apply_chat_template -- returns 3 interaction response tokens
            interaction_response_ids = [200, 201, 202]
            loop.apply_chat_template = AsyncMock(return_value=interaction_response_ids)

            state = await loop._handle_interacting_state(ad)

            # Entropy should be padded with 3 zeros
            assert ad.response_entropy == [0.9, 0.8, 0.0, 0.0, 0.0]
            # Same for logprobs
            assert ad.response_logprobs == [-0.5, -0.3, 0.0, 0.0, 0.0]
            assert state == AgentState.GENERATING

        asyncio.run(_run())


class TestRunOutputHasEntropyInExtraFields:
    def test_entropy_in_extra_fields(self):
        """run() includes response_entropy in extra_fields of AgentLoopOutput."""

        async def _run():
            loop = _build_loop(response_length=128, prompt_length=64)

            # Mock process_vision_info
            loop.process_vision_info = AsyncMock(return_value={})

            # Mock apply_chat_template
            prompt_ids = list(range(10))
            loop.apply_chat_template = AsyncMock(return_value=prompt_ids)

            # Mock generate to return EntropyTokenOutput then terminate
            entropy_output = EntropyTokenOutput(
                token_ids=[50, 51, 52, 53, 54],
                log_probs=[-0.1, -0.2, -0.3, -0.4, -0.5],
                entropy=[0.5, 0.6, 0.7, 0.8, 0.9],
                num_preempted=0,
            )
            loop.server_manager.generate = AsyncMock(return_value=entropy_output)

            # tool_parser returns no tool calls -> TERMINATED
            loop.tool_parser.extract_tool_calls = AsyncMock(return_value=(None, []))

            output = await loop.run(
                sampling_params={"temperature": 1.0},
                raw_prompt=[{"role": "user", "content": "hello"}],
            )

            assert isinstance(output, AgentLoopOutput)
            assert "response_entropy" in output.extra_fields
            assert output.extra_fields["response_entropy"] == [0.5, 0.6, 0.7, 0.8, 0.9]

        asyncio.run(_run())


class TestRunOutputEntropyTruncatedToResponseLength:
    def test_entropy_truncated(self):
        """Entropy in extra_fields is truncated to self.response_length."""

        async def _run():
            loop = _build_loop(response_length=3, prompt_length=64)

            loop.process_vision_info = AsyncMock(return_value={})
            prompt_ids = list(range(10))
            loop.apply_chat_template = AsyncMock(return_value=prompt_ids)

            entropy_output = EntropyTokenOutput(
                token_ids=[50, 51, 52, 53, 54],
                log_probs=[-0.1, -0.2, -0.3, -0.4, -0.5],
                entropy=[0.5, 0.6, 0.7, 0.8, 0.9],
                num_preempted=0,
            )
            loop.server_manager.generate = AsyncMock(return_value=entropy_output)
            loop.tool_parser.extract_tool_calls = AsyncMock(return_value=(None, []))

            output = await loop.run(
                sampling_params={"temperature": 1.0},
                raw_prompt=[{"role": "user", "content": "hello"}],
            )

            # response_length=3, so truncated
            assert output.extra_fields["response_entropy"] == [0.5, 0.6, 0.7]
            # response_ids also truncated
            assert output.response_ids == [50, 51, 52]

        asyncio.run(_run())

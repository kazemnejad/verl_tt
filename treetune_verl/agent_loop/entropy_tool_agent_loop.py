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

"""Entropy-aware tool agent loop.

Subclasses ToolAgentLoop to extract per-token entropy from
EntropyTokenOutput and pass it through extra_fields alongside
logprobs and tool-reward data.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("entropy_tool_agent")
class EntropyToolAgentLoop(ToolAgentLoop):
    """Tool agent loop with per-token entropy extraction.

    Overrides the state machine to accumulate per-token entropy from
    ``EntropyTokenOutput`` objects returned by an entropy-aware server.
    Entropy values are padded with ``0.0`` for non-generated tokens
    (tool responses, interaction responses) and stored in
    ``AgentLoopOutput.extra_fields["response_entropy"]``.
    """

    # ------------------------------------------------------------------
    # run() — full reimplementation to init response_entropy and emit it
    # ------------------------------------------------------------------

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        # --- Entropy addition: initialise accumulator ---
        agent_data.response_entropy = []

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})

        # --- Entropy addition: store in extra_fields ---
        if hasattr(agent_data, "response_entropy") and agent_data.response_entropy:
            output.extra_fields["response_entropy"] = agent_data.response_entropy[: self.response_length]

        return output

    # ------------------------------------------------------------------
    # _handle_generating_state — full reimplementation to access output
    # ------------------------------------------------------------------

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state with entropy accumulation."""
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # --- Entropy addition: accumulate per-token entropy ---
        entropy = getattr(output, "entropy", None)
        if entropy is not None:
            if not hasattr(agent_data, "response_entropy"):
                agent_data.response_entropy = []
            agent_data.response_entropy += list(entropy)

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    # ------------------------------------------------------------------
    # _handle_processing_tools_state — wrap super() with mask-diff trick
    # ------------------------------------------------------------------

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle processing tools state with entropy padding."""
        mask_len_before = len(agent_data.response_mask)
        state = await super()._handle_processing_tools_state(agent_data)
        mask_len_after = len(agent_data.response_mask)

        # Number of 0-mask tokens added by parent (tool response tokens)
        num_padded = mask_len_after - mask_len_before
        if hasattr(agent_data, "response_entropy") and agent_data.response_entropy:
            agent_data.response_entropy += [0.0] * num_padded

        return state

    # ------------------------------------------------------------------
    # _handle_interacting_state — wrap super() with mask-diff trick
    # ------------------------------------------------------------------

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle interacting state with entropy padding."""
        mask_len_before = len(agent_data.response_mask)
        state = await super()._handle_interacting_state(agent_data)
        mask_len_after = len(agent_data.response_mask)

        # Number of 0-mask tokens added by parent (interaction response tokens)
        num_padded = mask_len_after - mask_len_before
        if hasattr(agent_data, "response_entropy") and agent_data.response_entropy:
            agent_data.response_entropy += [0.0] * num_padded

        return state

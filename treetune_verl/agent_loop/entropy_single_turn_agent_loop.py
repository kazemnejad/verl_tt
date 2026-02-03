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

"""Entropy-aware single-turn agent loop.

Subclasses SingleTurnAgentLoop to extract per-token entropy from
EntropyTokenOutput and pass it through extra_fields.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("entropy_single_turn_agent")
class EntropySingleTurnAgentLoop(SingleTurnAgentLoop):
    """Single-turn agent loop with per-token entropy extraction."""

    # SYNC WARNING: single_turn_agent_loop.py:SingleTurnAgentLoop.run() — see agent-docs/sync-warnings.md
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            images=images,
            videos=videos,
        )

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            token_output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        response_mask = [1] * len(token_output.token_ids)

        # Extract entropy from EntropyTokenOutput (defensive getattr)
        entropy = getattr(token_output, "entropy", None)
        extra_fields = {}
        if entropy is not None:
            extra_fields["response_entropy"] = list(entropy[: self.response_length])

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=token_output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=token_output.log_probs[: self.response_length] if token_output.log_probs else None,
            routed_experts=(
                token_output.routed_experts[: len(prompt_ids) + self.response_length]
                if token_output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

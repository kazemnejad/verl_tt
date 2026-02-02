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

"""Entropy-aware AgentLoopWorker.

Subclasses ``AgentLoopWorker`` to convert per-token entropy lists
(produced by entropy-aware agent loops) into padded tensors and batch
them into ``DataProto.batch["rollout_entropy"]``.

Data flow:
    1. Agent loops (entropy_single_turn / entropy_tool) produce
       ``AgentLoopOutput`` with ``extra_fields["response_entropy"]``
       as ``list[float]``.
    2. ``_agent_loop_postprocess`` pops the list before super(),
       converts it to a padded ``[1, response_length]`` tensor,
       and stores it back in ``result.extra_fields["response_entropy"]``.
    3. ``_postprocess`` pops entropy tensors from each input's
       ``extra_fields`` (before super() would convert them to
       ``np.array``), calls super(), then stacks the tensors into
       ``DataProto.batch["rollout_entropy"]`` with shape
       ``[bsz, response_length]``.
"""

from __future__ import annotations

import logging
import os

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EntropyAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker subclass that batches per-token entropy into DataProto."""

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Convert entropy list to padded tensor after parent post-processing."""
        # Pop entropy before super() stores it in extra_fields as-is
        response_entropy = output.extra_fields.pop("response_entropy", None)

        result = await super()._agent_loop_postprocess(output, **kwargs)

        if response_entropy is not None:
            response_length = self.config.actor_rollout_ref.rollout.response_length
            pad_size = response_length - len(response_entropy)
            entropy_tensor = torch.tensor(response_entropy + [0.0] * pad_size).unsqueeze(0)
            result.extra_fields["response_entropy"] = entropy_tensor

        return result

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Pop entropy tensors, delegate to super(), then stack into batch."""
        # Pop entropy tensors BEFORE super() converts them to np.array
        has_entropy = any("response_entropy" in inp.extra_fields for inp in inputs)

        entropy_tensors: list[torch.Tensor | None] = []
        if has_entropy:
            for inp in inputs:
                entropy_tensors.append(inp.extra_fields.pop("response_entropy", None))

        data_proto = super()._postprocess(inputs)

        if has_entropy:
            response_length = self.config.actor_rollout_ref.rollout.response_length
            stacked = []
            for t in entropy_tensors:
                if t is not None:
                    stacked.append(t)
                else:
                    stacked.append(torch.zeros(1, response_length))
            data_proto.batch["rollout_entropy"] = torch.cat(stacked, dim=0)

        return data_proto

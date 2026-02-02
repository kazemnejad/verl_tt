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
import sys
import types
from unittest.mock import Mock

import torch

# SGLang's import chain (via async_sglang_server → sglang_rollout → weight_sync)
# pulls in vllm. When this module is loaded in a CPU-only Ray worker, vllm is
# not available. We install the same minimal mock as verl's _load_sglang()
# so that pickle deserialization of EntropySGLangHttpServer handles succeeds.
os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
try:
    import vllm  # noqa: F401
except ImportError:
    _mock_vllm = types.ModuleType("vllm")

    _mock_custom_ops = types.ModuleType("vllm._custom_ops")
    _mock_custom_ops.scaled_fp8_quant = Mock()
    _mock_vllm._custom_ops = _mock_custom_ops

    _mock_model_executor = types.ModuleType("vllm.model_executor")
    _mock_layers = types.ModuleType("vllm.model_executor.layers")
    _mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

    class _GeluAndMul:
        pass

    class _SiluAndMul:
        pass

    _mock_activation.GeluAndMul = _GeluAndMul
    _mock_activation.SiluAndMul = _SiluAndMul
    _mock_layers.activation = _mock_activation
    _mock_model_executor.layers = _mock_layers
    _mock_vllm.model_executor = _mock_model_executor

    sys.modules.setdefault("vllm", _mock_vllm)
    sys.modules.setdefault("vllm._custom_ops", _mock_custom_ops)
    sys.modules.setdefault("vllm.model_executor", _mock_model_executor)
    sys.modules.setdefault("vllm.model_executor.layers", _mock_layers)
    sys.modules.setdefault("vllm.model_executor.layers.activation", _mock_activation)

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class EntropyAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker subclass that batches per-token entropy into DataProto."""

    def __init__(self, *args, **kwargs):
        # Import entropy agent loops to trigger @register() in worker process.
        # Ray actors start fresh processes — without these imports, names like
        # "entropy_single_turn_agent" won't be in the _agent_loop_registry.
        import treetune_verl.agent_loop.entropy_single_turn_agent_loop  # noqa: F401
        import treetune_verl.agent_loop.entropy_tool_agent_loop  # noqa: F401

        super().__init__(*args, **kwargs)

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

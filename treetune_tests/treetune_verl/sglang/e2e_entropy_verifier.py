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

"""E2E verification subclasses for the entropy pipeline.

These classes extend the entropy manager/worker to write a JSON signal
file when ``rollout_entropy`` appears in the output DataProto, allowing
the E2E test driver to verify the full pipeline.

The signal file path is communicated via ``TREETUNE_E2E_ENTROPY_SIGNAL``.
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import Mock

# -- vllm mock ---------------------------------------------------------------
# The import chain from server.py -> async_sglang_server -> sglang_rollout
# pulls in vllm. In CPU-only Ray workers (including the TaskRunner that
# resolves the FQN), vllm is not available. Mirror verl's _load_sglang() mock.
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
# ---------------------------------------------------------------------------

import ray
from omegaconf import DictConfig

from treetune_verl.agent_loop.entropy_manager import EntropyAgentLoopManager, _load_entropy_replica
from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, _InternalAgentLoopOutput
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup


class VerifyingEntropyWorker(EntropyAgentLoopWorker):
    """EntropyAgentLoopWorker that writes a signal file for E2E verification."""

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        data_proto = super()._postprocess(inputs)
        signal_path = os.environ.get("TREETUNE_E2E_ENTROPY_SIGNAL")
        if signal_path and "rollout_entropy" in data_proto.batch.keys():
            entropy = data_proto.batch["rollout_entropy"]
            bsz = data_proto.batch["responses"].shape[0]
            resp_len = data_proto.batch["responses"].shape[1]
            signal = {
                "verified": True,
                "shape": list(entropy.shape),
                "expected_shape": [bsz, resp_len],
                "all_non_negative": bool((entropy >= -0.01).all()),
                "min_value": float(entropy.min()),
                "mean_positive": float(entropy[entropy > 0].mean()) if (entropy > 0).any() else 0.0,
                "num_positive": int((entropy > 0).sum()),
            }
            with open(signal_path, "w") as f:
                json.dump(signal, f)
            print(
                f"[E2E] rollout_entropy signal written: shape={list(entropy.shape)}, "
                f"mean_positive={signal['mean_positive']:.4f}, "
                f"num_positive={signal['num_positive']}"
            )
        return data_proto


class VerifyingEntropyManager(EntropyAgentLoopManager):
    """EntropyAgentLoopManager that uses VerifyingEntropyWorker."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        rm_resource_pool: RayResourcePool = None,
    ):
        # Preset our classes before the grandparent __init__ runs.
        self.rollout_replica_class = _load_entropy_replica()
        self.agent_loop_workers_class = ray.remote(VerifyingEntropyWorker)

        # Call AgentLoopManager.__init__ directly (skip EntropyAgentLoopManager)
        AgentLoopManager.__init__(
            self,
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            rm_resource_pool=rm_resource_pool,
        )

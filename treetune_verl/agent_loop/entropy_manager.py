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

"""Entropy-aware AgentLoopManager.

Presets ``rollout_replica_class`` and ``agent_loop_workers_class``
before ``AgentLoopManager.__init__`` runs, so the parent's
``hasattr`` guards use our entropy-aware subclasses:

- ``EntropySGLangReplica`` for the rollout server
  (launches sglang with entropy extraction patches)
- ``EntropyAgentLoopWorker`` (wrapped by ``ray.remote``)
  (registers entropy agent loops in the worker process
  and batches per-token entropy into DataProto)

Note: The SGLang import chain pulls in vllm. In the TaskRunner
process where this manager is instantiated, vllm may not be
installed. We reuse the same vllm mock pattern from
``verl.workers.rollout.replica._load_sglang``.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import Mock

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup


def _ensure_vllm_mock():
    """Install a minimal vllm mock if vllm is not importable.

    Mirrors the approach in ``verl.workers.rollout.replica._load_sglang``
    so that SGLang modules can be imported in CPU-only Ray workers.
    """
    try:
        import vllm  # noqa: F401

        return  # real vllm is available
    except ImportError:
        pass

    mock_vllm = types.ModuleType("vllm")

    mock_custom_ops = types.ModuleType("vllm._custom_ops")
    mock_custom_ops.scaled_fp8_quant = Mock()
    mock_vllm._custom_ops = mock_custom_ops

    mock_model_executor = types.ModuleType("vllm.model_executor")
    mock_layers = types.ModuleType("vllm.model_executor.layers")
    mock_activation = types.ModuleType("vllm.model_executor.layers.activation")

    class GeluAndMul:  # noqa: N801
        pass

    class SiluAndMul:  # noqa: N801
        pass

    mock_activation.GeluAndMul = GeluAndMul
    mock_activation.SiluAndMul = SiluAndMul
    mock_layers.activation = mock_activation
    mock_model_executor.layers = mock_layers
    mock_vllm.model_executor = mock_model_executor

    sys.modules["vllm"] = mock_vllm
    sys.modules["vllm._custom_ops"] = mock_custom_ops
    sys.modules["vllm.model_executor"] = mock_model_executor
    sys.modules["vllm.model_executor.layers"] = mock_layers
    sys.modules["vllm.model_executor.layers.activation"] = mock_activation


def _load_entropy_replica():
    """Lazily import EntropySGLangReplica with vllm mock and CPU engine flag."""
    old_env = os.environ.get("SGLANG_USE_CPU_ENGINE")
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    _ensure_vllm_mock()

    from treetune_verl.sglang.server import EntropySGLangReplica

    # Restore env
    if old_env is None:
        os.environ.pop("SGLANG_USE_CPU_ENGINE", None)
    else:
        os.environ["SGLANG_USE_CPU_ENGINE"] = old_env

    return EntropySGLangReplica


class EntropyAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with entropy-aware server and worker classes."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        rm_resource_pool: RayResourcePool = None,
    ):
        from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker

        # Preset before super().__init__ checks hasattr
        self.rollout_replica_class = _load_entropy_replica()
        self.agent_loop_workers_class = ray.remote(EntropyAgentLoopWorker)

        super().__init__(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=rollout_resource_pool,
            rm_resource_pool=rm_resource_pool,
        )

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

"""Tests for entropy_gen recipe: StreamingEntropyWorker + EntropyGenerationLoopManager."""

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


def _make_manager_config():
    """Minimal config for EntropyGenerationLoopManager tests."""
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"path": "dummy/model"},
                "rollout": {
                    "name": "sglang",
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "calculate_log_probs": False,
                    "free_cache_engine": False,
                    "tensor_model_parallel_size": 1,
                    "data_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "agent": {
                        "num_workers": 2,
                        "default_agent_loop": "single_turn_agent",
                    },
                },
            },
            "reward_model": {
                "enable": False,
                "use_reward_loop": False,
                "enable_resource_pool": False,
            },
            "trainer": {
                "n_gpus_per_node": 1,
                "nnodes": 1,
            },
        }
    )


def _init_workers_side_effect(self_obj):
    """Side effect that creates mock workers on the manager object."""
    self_obj.agent_loop_workers = [MagicMock() for _ in range(2)]


class TestStreamingEntropyWorkerHasMixinMethods:
    """StreamingEntropyWorker has methods from both parents."""

    def test_streaming_entropy_worker_has_mixin_methods(self):
        """Verify set_queue, generate_sequences_streaming (mixin) and _agent_loop_postprocess (entropy)."""
        from treetune_recipe.entropy_gen.agent_loop import StreamingEntropyWorker

        assert hasattr(StreamingEntropyWorker, "set_queue")
        assert hasattr(StreamingEntropyWorker, "generate_sequences_streaming")
        assert hasattr(StreamingEntropyWorker, "_agent_loop_postprocess")


class TestStreamingEntropyWorkerMRO:
    """MRO: mixin before entropy worker."""

    def test_streaming_entropy_worker_mro(self):
        """StreamingAgentLoopWorkerMixin comes before EntropyAgentLoopWorker in MRO."""
        from treetune_recipe.entropy_gen.agent_loop import StreamingEntropyWorker
        from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker
        from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin

        mro = StreamingEntropyWorker.__mro__
        mixin_idx = mro.index(StreamingAgentLoopWorkerMixin)
        entropy_idx = mro.index(EntropyAgentLoopWorker)
        assert mixin_idx < entropy_idx, f"Mixin at {mixin_idx}, EntropyWorker at {entropy_idx}"


class TestEntropyGenerationLoopManagerSetsEntropyClasses:
    """EntropyGenerationLoopManager pre-sets entropy-aware replica + worker classes."""

    @patch("treetune_recipe.entropy_gen.agent_loop._load_entropy_replica")
    @patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers")
    @patch(
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
        autospec=True,
    )
    def test_entropy_generation_loop_manager_sets_entropy_classes(
        self, mock_init_workers, mock_init_servers, mock_load_replica
    ):
        """Verify entropy worker class, replica from _load_entropy_replica, queue injected."""
        import ray

        from treetune_recipe.entropy_gen.agent_loop import EntropyGenerationLoopManager
        from treetune_verl.generation.worker import StreamingAgentLoopWorker

        mock_init_workers.side_effect = _init_workers_side_effect
        mock_load_replica.return_value = "MockEntropySGLangReplica"

        queue = MagicMock()
        config = _make_manager_config()

        with patch.object(ray, "get"):
            manager = EntropyGenerationLoopManager(config, queue)

        # Worker class is NOT the base StreamingAgentLoopWorker
        assert manager.agent_loop_workers_class is not ray.remote(StreamingAgentLoopWorker)

        # Replica class is the result of _load_entropy_replica
        assert manager.rollout_replica_class == "MockEntropySGLangReplica"

        # Queue injection happened
        for worker in manager.agent_loop_workers:
            worker.set_queue.remote.assert_called_once_with(queue)

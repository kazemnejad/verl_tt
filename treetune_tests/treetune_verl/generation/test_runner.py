"""Tests for GenerationLoopManager."""

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf


def _make_manager_config():
    """Minimal config for GenerationLoopManager tests."""
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


class TestGenerationLoopManagerInit:
    """Task 6: __init__ stores queue and sets worker class."""

    @patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers")
    @patch(
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
        autospec=True,
    )
    def test_stores_queue(self, mock_init_workers, mock_init_servers):
        """GenerationLoopManager.__init__ stores queue as self._queue."""
        import ray

        from treetune_verl.generation.runner import GenerationLoopManager

        mock_init_workers.side_effect = _init_workers_side_effect

        queue = MagicMock()
        config = _make_manager_config()

        with patch.object(ray, "get"):
            manager = GenerationLoopManager(config, queue)

        assert manager._queue is queue

    @patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers")
    @patch(
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
        autospec=True,
    )
    def test_sets_worker_class_to_streaming(self, mock_init_workers, mock_init_servers):
        """agent_loop_workers_class wraps StreamingAgentLoopWorker."""
        import ray

        from treetune_verl.generation.runner import GenerationLoopManager
        from treetune_verl.generation.worker import StreamingAgentLoopWorker

        mock_init_workers.side_effect = _init_workers_side_effect

        queue = MagicMock()
        config = _make_manager_config()

        with patch.object(ray, "remote", return_value="remote_cls") as mock_remote:
            with patch.object(ray, "get"):
                manager = GenerationLoopManager(config, queue)

        mock_remote.assert_called_once_with(StreamingAgentLoopWorker)
        assert manager.agent_loop_workers_class == "remote_cls"

    @patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers")
    @patch(
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
        autospec=True,
    )
    def test_passes_none_worker_group_and_resource_pool(self, mock_init_workers, mock_init_servers):
        """super().__init__ called with worker_group=None, rollout_resource_pool=None."""
        import ray

        from treetune_verl.generation.runner import GenerationLoopManager

        mock_init_workers.side_effect = _init_workers_side_effect

        queue = MagicMock()
        config = _make_manager_config()

        with patch.object(ray, "get"):
            manager = GenerationLoopManager(config, queue)

        # worker_group should be None (standalone mode)
        assert manager.worker_group is None


class TestGenerationLoopManagerQueueInjection:
    """Task 7: queue injection into workers after init."""

    @patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers")
    @patch(
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
        autospec=True,
    )
    def test_set_queue_called_on_each_worker(self, mock_init_workers, mock_init_servers):
        """After init, set_queue.remote(queue) called on every worker."""
        import ray

        from treetune_verl.generation.runner import GenerationLoopManager

        mock_init_workers.side_effect = _init_workers_side_effect

        queue = MagicMock()
        config = _make_manager_config()

        with patch.object(ray, "get") as mock_ray_get:
            manager = GenerationLoopManager(config, queue)

        # ray.get should have been called with a list of refs
        mock_ray_get.assert_called_once()

        # Each worker's set_queue.remote should have been called with the queue
        for worker in manager.agent_loop_workers:
            worker.set_queue.remote.assert_called_once_with(queue)

    @patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers")
    @patch(
        "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
        autospec=True,
    )
    def test_ray_get_blocks_on_set_queue_refs(self, mock_init_workers, mock_init_servers):
        """ray.get is called with the list of set_queue refs to ensure injection completes."""
        import ray

        from treetune_verl.generation.runner import GenerationLoopManager

        mock_init_workers.side_effect = _init_workers_side_effect

        queue = MagicMock()
        config = _make_manager_config()

        with patch.object(ray, "get") as mock_ray_get:
            manager = GenerationLoopManager(config, queue)

        # ray.get was called with a list of refs from set_queue.remote
        call_args = mock_ray_get.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == len(manager.agent_loop_workers)

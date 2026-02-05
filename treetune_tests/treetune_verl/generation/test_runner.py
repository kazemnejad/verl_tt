"""Tests for GenerationLoopManager and GenerationRunner."""

import json
import tempfile
from pathlib import Path
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


def _build_manager_with_mock_workers(num_workers=2):
    """Helper: create a GenerationLoopManager with mocked internals."""
    import ray

    from treetune_verl.generation.runner import GenerationLoopManager

    config = _make_manager_config()
    queue = MagicMock()

    def side_effect(self_obj):
        self_obj.agent_loop_workers = [MagicMock() for _ in range(num_workers)]

    with (
        patch("verl.experimental.agent_loop.agent_loop.AgentLoopManager._initialize_llm_servers"),
        patch(
            "verl.experimental.agent_loop.agent_loop.AgentLoopManager._init_agent_loop_workers",
            autospec=True,
            side_effect=side_effect,
        ),
        patch.object(ray, "get"),
    ):
        manager = GenerationLoopManager(config, queue)

    return manager


class TestDispatchStreaming:
    """Task 8: dispatch_streaming dispatches chunks and returns refs."""

    def test_calls_wake_up(self):
        """dispatch_streaming calls self.wake_up() first."""
        manager = _build_manager_with_mock_workers()
        prompts = MagicMock()
        prompts.chunk.return_value = [MagicMock(), MagicMock()]

        manager.wake_up = MagicMock()
        manager.dispatch_streaming(prompts)

        manager.wake_up.assert_called_once()

    def test_chunks_prompts_by_num_workers(self):
        """Prompts are chunked by the number of agent_loop_workers."""
        manager = _build_manager_with_mock_workers(num_workers=3)
        prompts = MagicMock()
        prompts.chunk.return_value = [MagicMock(), MagicMock(), MagicMock()]

        manager.wake_up = MagicMock()
        manager.dispatch_streaming(prompts)

        prompts.chunk.assert_called_once_with(3)

    def test_calls_generate_sequences_streaming_remote_per_worker(self):
        """Each worker gets generate_sequences_streaming.remote(chunk)."""
        manager = _build_manager_with_mock_workers(num_workers=2)
        chunk_a, chunk_b = MagicMock(), MagicMock()
        prompts = MagicMock()
        prompts.chunk.return_value = [chunk_a, chunk_b]

        manager.wake_up = MagicMock()
        manager.dispatch_streaming(prompts)

        w0, w1 = manager.agent_loop_workers
        w0.generate_sequences_streaming.remote.assert_called_once_with(chunk_a)
        w1.generate_sequences_streaming.remote.assert_called_once_with(chunk_b)

    def test_returns_list_of_refs(self):
        """Returns a list of ObjectRefs (one per worker)."""
        manager = _build_manager_with_mock_workers(num_workers=2)
        prompts = MagicMock()
        prompts.chunk.return_value = [MagicMock(), MagicMock()]

        # Each worker.generate_sequences_streaming.remote returns a mock ref
        ref_a = MagicMock(name="ref_a")
        ref_b = MagicMock(name="ref_b")
        manager.agent_loop_workers[0].generate_sequences_streaming.remote.return_value = ref_a
        manager.agent_loop_workers[1].generate_sequences_streaming.remote.return_value = ref_b

        manager.wake_up = MagicMock()
        result = manager.dispatch_streaming(prompts)

        assert result == [ref_a, ref_b]


# ---------------------------------------------------------------------------
# GenerationRunner helpers
# ---------------------------------------------------------------------------


def _make_runner(output_dir: Path):
    """Create a GenerationRunner without calling __init__, set attributes manually."""
    from treetune_verl.generation.runner import GenerationRunner

    runner = GenerationRunner.__new__(GenerationRunner)
    runner.output_dir = output_dir
    runner.completed_indices = set()
    runner.saved_batches = []
    runner.total_samples = 0
    return runner


# ---------------------------------------------------------------------------
# Task 9: _save_checkpoint (atomic write)
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    """Task 9: _save_checkpoint writes checkpoint.json atomically."""

    def test_writes_checkpoint_json_with_correct_content(self):
        """checkpoint.json contains completed_indices (sorted), saved_batches, total_samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.completed_indices = {3, 1, 2}
            runner.saved_batches = ["batch_0000"]
            runner.total_samples = 10

            runner._save_checkpoint()

            ckpt = json.loads((output_dir / "checkpoint.json").read_text())
            assert ckpt["completed_indices"] == [1, 2, 3]
            assert ckpt["saved_batches"] == ["batch_0000"]
            assert ckpt["total_samples"] == 10

    def test_atomic_write_no_tmp_file_left(self):
        """After _save_checkpoint, no .tmp file remains."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.completed_indices = set()
            runner.saved_batches = []
            runner.total_samples = 0

            runner._save_checkpoint()

            assert not (output_dir / "checkpoint.json.tmp").exists()
            assert (output_dir / "checkpoint.json").exists()


# ---------------------------------------------------------------------------
# Task 10: _load_checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    """Task 10: _load_checkpoint restores state or returns False."""

    def test_returns_true_and_restores_state_when_file_exists(self):
        """Loads checkpoint.json → completed_indices (set), saved_batches (list)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Write a checkpoint file
            (output_dir / "checkpoint.json").write_text(
                json.dumps(
                    {
                        "completed_indices": [5, 2, 8],
                        "saved_batches": ["batch_0000", "batch_0001"],
                        "total_samples": 20,
                    }
                )
            )

            runner = _make_runner(output_dir)
            result = runner._load_checkpoint()

            assert result is True
            assert runner.completed_indices == {2, 5, 8}
            assert runner.saved_batches == ["batch_0000", "batch_0001"]

    def test_returns_false_when_file_missing(self):
        """Returns False when checkpoint.json doesn't exist; state unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.completed_indices = set()
            runner.saved_batches = []

            result = runner._load_checkpoint()

            assert result is False
            assert runner.completed_indices == set()
            assert runner.saved_batches == []

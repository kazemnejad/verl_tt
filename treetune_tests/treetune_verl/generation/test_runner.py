"""Tests for GenerationLoopManager and GenerationRunner."""

import json
import pickle
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from verl.protocol import DataProto


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
    runner.dataset = None
    runner.collate_fn = None
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


# ---------------------------------------------------------------------------
# Task 11: _save_batch
# ---------------------------------------------------------------------------


class TestSaveBatch:
    """Task 11: _save_batch writes pickle, updates state, checkpoints."""

    def test_writes_pickle_file_with_items(self):
        """batch_NNNN.pkl contains the items list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 5

            items = [(0, "fake_data_0"), (3, "fake_data_3")]
            runner._save_batch(items, batch_idx=0)

            batch_path = output_dir / "batch_0000.pkl"
            assert batch_path.exists()
            with open(batch_path, "rb") as f:
                loaded = pickle.load(f)
            assert loaded == items

    def test_updates_completed_indices(self):
        """completed_indices gains the indices from items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 10

            runner._save_batch([(1, "d1"), (4, "d4")], batch_idx=0)

            assert runner.completed_indices == {1, 4}

    def test_appends_batch_name_to_saved_batches(self):
        """saved_batches grows with each _save_batch call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 10

            runner._save_batch([(0, "d0")], batch_idx=0)
            runner._save_batch([(1, "d1")], batch_idx=1)

            assert runner.saved_batches == ["batch_0000", "batch_0001"]

    def test_calls_save_checkpoint(self):
        """_save_batch calls _save_checkpoint after updating state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 5

            runner._save_batch([(2, "d2")], batch_idx=0)

            # checkpoint.json should exist (written by _save_checkpoint)
            assert (output_dir / "checkpoint.json").exists()
            ckpt = json.loads((output_dir / "checkpoint.json").read_text())
            assert 2 in ckpt["completed_indices"]


# ---------------------------------------------------------------------------
# Task 12: _merge_batches
# ---------------------------------------------------------------------------


def _write_batch_file(output_dir: Path, batch_name: str, items: list) -> None:
    """Helper: write a batch pickle file directly (avoids pickling MagicMock)."""
    with open(output_dir / f"{batch_name}.pkl", "wb") as f:
        pickle.dump(items, f)


class TestMergeBatches:
    """Task 12: _merge_batches reads batches, sorts, concats, saves."""

    def test_merges_batches_sorted_by_index(self):
        """Items from all batches are sorted by index before concat."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 4

            # Use simple strings as data stand-ins (picklable)
            _write_batch_file(output_dir, "batch_0000", [(3, "dp_3"), (1, "dp_1")])
            _write_batch_file(output_dir, "batch_0001", [(0, "dp_0"), (2, "dp_2")])
            runner.saved_batches = ["batch_0000", "batch_0001"]

            merged_dp = MagicMock(name="merged")
            with patch.object(DataProto, "concat", return_value=merged_dp) as mock_concat:
                result = runner._merge_batches()

            # concat called with items sorted by index: 0, 1, 2, 3
            call_args = mock_concat.call_args[0][0]
            assert call_args == ["dp_0", "dp_1", "dp_2", "dp_3"]

            # save_to_disk called
            merged_dp.save_to_disk.assert_called_once_with(output_dir / "trajectories.pkl")

            assert result is merged_dp

    def test_saves_trajectories_pkl(self):
        """Merged result is saved to trajectories.pkl via save_to_disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 1

            _write_batch_file(output_dir, "batch_0000", [(0, "dp_0")])
            runner.saved_batches = ["batch_0000"]

            merged_dp = MagicMock()
            with patch.object(DataProto, "concat", return_value=merged_dp):
                runner._merge_batches()

            merged_dp.save_to_disk.assert_called_once_with(output_dir / "trajectories.pkl")


# ---------------------------------------------------------------------------
# Task 13: _upload_artifact
# ---------------------------------------------------------------------------


class TestUploadArtifact:
    """Task 13: _upload_artifact creates zip and logs to wandb."""

    def test_creates_zip_with_existing_files(self):
        """trajectories.zip contains trajectories.pkl and checkpoint.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.tracker = MagicMock()
            runner.tracker.logger = {}  # no wandb

            # Create the files
            (output_dir / "trajectories.pkl").write_bytes(b"fake_data")
            (output_dir / "checkpoint.json").write_text('{"test": true}')

            runner._upload_artifact()

            zip_path = output_dir / "trajectories.zip"
            assert zip_path.exists()
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                assert "trajectories.pkl" in names
                assert "checkpoint.json" in names

    def test_logs_artifact_to_wandb_when_present(self):
        """When wandb in tracker.logger, creates Artifact, adds file, logs it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)

            mock_wandb = MagicMock()
            mock_artifact = MagicMock()
            mock_wandb.Artifact.return_value = mock_artifact

            runner.tracker = MagicMock()
            runner.tracker.logger = {"wandb": mock_wandb}

            # Create the files
            (output_dir / "trajectories.pkl").write_bytes(b"data")
            (output_dir / "checkpoint.json").write_text("{}")

            runner._upload_artifact()

            mock_wandb.Artifact.assert_called_once_with(name="trajectories", type="trajectories")
            mock_artifact.add_file.assert_called_once_with(
                str(output_dir / "trajectories.zip"), name="trajectories.zip"
            )
            mock_wandb.log_artifact.assert_called_once_with(mock_artifact)

    def test_skips_missing_files_in_zip(self):
        """If trajectories.pkl doesn't exist, it's not added to the zip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.tracker = MagicMock()
            runner.tracker.logger = {}

            # Only checkpoint exists
            (output_dir / "checkpoint.json").write_text("{}")

            runner._upload_artifact()

            with zipfile.ZipFile(output_dir / "trajectories.zip", "r") as zf:
                names = zf.namelist()
                assert "trajectories.pkl" not in names
                assert "checkpoint.json" in names


# ---------------------------------------------------------------------------
# Task 14: __init__
# ---------------------------------------------------------------------------


def _make_init_config(tmpdir: str):
    """Config for GenerationRunner.__init__ tests."""
    return OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": tmpdir,
                "project_name": "test_project",
                "experiment_name": "test_exp",
                "logger": ["console"],
            },
            "generation": {
                "save_batch_size": 100,
                "pull_timeout": 5.0,
                "final_merge": True,
                "show_progress": False,
                "upload_artifact": False,
            },
        }
    )


class _DummyDataset(Dataset):
    """Minimal dataset for unit tests."""

    def __init__(self, size: int = 10):
        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return {"prompt": f"prompt_{idx}"}


def _dummy_collate_fn(batch):
    return batch


class TestGenerationRunnerInit:
    """Task 14: __init__ sets state, creates output_dir, loads checkpoint."""

    @patch("treetune_verl.generation.runner.Tracking")
    def test_sets_output_dir_and_creates_it(self, mock_tracking_cls):
        """output_dir set from config and created on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = str(Path(tmpdir) / "sub" / "dir")
            config = _make_init_config(sub)

            from treetune_verl.generation.runner import GenerationRunner

            runner = GenerationRunner(config, _DummyDataset(), _dummy_collate_fn)

            assert runner.output_dir == Path(sub)
            assert runner.output_dir.exists()

    @patch("treetune_verl.generation.runner.Tracking")
    def test_initializes_empty_state(self, mock_tracking_cls):
        """completed_indices, saved_batches start empty; total_samples = len(dataset)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_init_config(tmpdir)
            dataset = _DummyDataset(size=10)

            from treetune_verl.generation.runner import GenerationRunner

            runner = GenerationRunner(config, dataset, _dummy_collate_fn)

            assert runner.completed_indices == set()
            assert runner.saved_batches == []
            assert runner.total_samples == 10
            assert runner.dataset is dataset
            assert runner.collate_fn is _dummy_collate_fn

    @patch("treetune_verl.generation.runner.Tracking")
    def test_loads_existing_checkpoint(self, mock_tracking_cls):
        """If checkpoint.json exists, state is restored from it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-write a checkpoint
            Path(tmpdir, "checkpoint.json").write_text(
                json.dumps(
                    {
                        "completed_indices": [1, 3, 5],
                        "saved_batches": ["batch_0000"],
                        "total_samples": 10,
                    }
                )
            )

            config = _make_init_config(tmpdir)

            from treetune_verl.generation.runner import GenerationRunner

            runner = GenerationRunner(config, _DummyDataset(), _dummy_collate_fn)

            assert runner.completed_indices == {1, 3, 5}
            assert runner.saved_batches == ["batch_0000"]

    @patch("treetune_verl.generation.runner.Tracking")
    def test_creates_tracking_instance(self, mock_tracking_cls):
        """Tracking is instantiated with project, experiment, logger, config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _make_init_config(tmpdir)

            from treetune_verl.generation.runner import GenerationRunner

            runner = GenerationRunner(config, _DummyDataset(), _dummy_collate_fn)

            mock_tracking_cls.assert_called_once()
            call_kwargs = mock_tracking_cls.call_args[1]
            assert call_kwargs["project_name"] == "test_project"
            assert call_kwargs["experiment_name"] == "test_exp"
            assert runner.tracker is mock_tracking_cls.return_value


# ---------------------------------------------------------------------------
# Task 15: run() orchestration
# ---------------------------------------------------------------------------


class TestRunOrchestration:
    """Task 15: run() loads data, dispatches, pulls, merges, uploads."""

    def test_all_completed_skips_to_merge_and_upload(self):
        """When all indices complete, skips dispatch; calls merge + upload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 3
            runner.completed_indices = {0, 1, 2}  # all done
            runner.config = OmegaConf.create({"generation": {"final_merge": True, "upload_artifact": True}})
            runner.tracker = MagicMock()
            runner._load_data = MagicMock()
            runner._merge_batches = MagicMock()
            runner._upload_artifact = MagicMock()

            runner.run()

            runner._load_data.assert_called_once()
            runner._merge_batches.assert_called_once()
            runner._upload_artifact.assert_called_once()

    def test_all_completed_skips_merge_when_disabled(self):
        """When final_merge=False, _merge_batches is NOT called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 2
            runner.completed_indices = {0, 1}
            runner.config = OmegaConf.create({"generation": {"final_merge": False, "upload_artifact": False}})
            runner.tracker = MagicMock()
            runner._load_data = MagicMock()
            runner._merge_batches = MagicMock()
            runner._upload_artifact = MagicMock()

            runner.run()

            runner._merge_batches.assert_not_called()
            runner._upload_artifact.assert_not_called()

    @patch("treetune_verl.generation.runner.ray")
    @patch("treetune_verl.generation.runner.GenerationLoopManager")
    @patch("treetune_verl.generation.runner.Queue")
    def test_pending_indices_dispatches_and_pulls(self, mock_queue_cls, mock_manager_cls, mock_ray):
        """With pending indices, creates manager, dispatches, runs pull loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            runner.total_samples = 3
            runner.completed_indices = {0}  # index 1, 2 pending
            runner.config = OmegaConf.create(
                {
                    "generation": {
                        "final_merge": False,
                        "upload_artifact": False,
                        "save_batch_size": 10,
                        "pull_timeout": 1.0,
                        "show_progress": False,
                    },
                }
            )
            runner.tracker = MagicMock()
            runner._load_data = MagicMock()
            runner._prepare_prompts = MagicMock(return_value=MagicMock())
            runner._save_batch = MagicMock()
            runner._merge_batches = MagicMock()
            runner._upload_artifact = MagicMock()

            # Mock queue to return items then signal completion
            mock_queue = MagicMock()
            mock_queue_cls.return_value = mock_queue

            from queue import Empty

            call_count = [0]

            def queue_get_side_effect(block=True, timeout=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    return (1, "data_1")
                elif call_count[0] == 2:
                    return (2, "data_2")
                else:
                    raise Empty()

            mock_queue.get.side_effect = queue_get_side_effect

            # Mock manager
            mock_manager = MagicMock()
            mock_manager_cls.return_value = mock_manager
            mock_manager.dispatch_streaming.return_value = [MagicMock()]

            runner.run()

            # Should have dispatched
            mock_manager.dispatch_streaming.assert_called_once()
            # Should have called ray.get on worker refs
            mock_ray.get.assert_called()
            # Manager should be slept
            mock_manager.sleep.assert_called_once()

    def test_has_load_data_stub(self):
        """_load_data exists as a method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            assert hasattr(runner, "_load_data") or callable(getattr(runner.__class__, "_load_data", None))

    def test_has_prepare_prompts_stub(self):
        """_prepare_prompts exists as a method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            runner = _make_runner(output_dir)
            assert hasattr(runner, "_prepare_prompts") or callable(getattr(runner.__class__, "_prepare_prompts", None))

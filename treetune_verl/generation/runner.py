"""GenerationLoopManager and GenerationRunner for streaming trajectory generation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.queue import Queue

from treetune_verl.generation.worker import StreamingAgentLoopWorker
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.utils.tracking import Tracking


class GenerationLoopManager(AgentLoopManager):
    """AgentLoopManager configured for streaming generation."""

    def __init__(self, config: DictConfig, queue: Queue):
        self._queue = queue
        self.agent_loop_workers_class = ray.remote(StreamingAgentLoopWorker)
        super().__init__(config, worker_group=None, rollout_resource_pool=None)
        # Inject queue into workers after creation
        ray.get([worker.set_queue.remote(self._queue) for worker in self.agent_loop_workers])

    def dispatch_streaming(self, prompts: DataProto) -> list[ray.ObjectRef]:
        """Non-blocking dispatch. Returns refs for completion check."""
        self.wake_up()
        chunks = prompts.chunk(len(self.agent_loop_workers))
        return [
            worker.generate_sequences_streaming.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
        ]

    def destroy(self) -> None:
        """Kill all Ray actors and remove placement groups to release GPU resources."""
        import logging
        import time

        from ray.util.placement_group import remove_placement_group

        logger = logging.getLogger(__name__)
        # Kill agent loop workers first (they don't hold GPUs)
        for worker in self.agent_loop_workers:
            try:
                ray.kill(worker)
            except Exception:
                pass
        for replica in self.rollout_replicas:
            # Kill all server actors in each replica
            for server in replica.servers:
                try:
                    ray.kill(server)
                except Exception:
                    pass
            # Kill all worker actors in each replica (hold GPUs via placement group)
            for worker in replica.workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
            # Remove placement group to free GPU resources
            if replica.resource_pool is not None and replica.resource_pool.pgs is not None:
                for pg in replica.resource_pool.pgs:
                    try:
                        remove_placement_group(pg)
                    except Exception:
                        pass
        # Brief wait for Ray to reclaim resources
        time.sleep(2)
        logger.info("GenerationLoopManager: destroyed all actors and freed resources")


class GenerationRunner:
    """Orchestrator for streaming trajectory generation."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.output_dir = Path(config.trainer.default_local_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.completed_indices: set[int] = set()
        self.saved_batches: list[str] = []
        self.total_samples = 0
        self._load_checkpoint()
        self.tracker = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config),
        )

    def _save_checkpoint(self) -> None:
        """Atomically write checkpoint.json (tmp → rename)."""
        tmp_path = self.output_dir / "checkpoint.json.tmp"
        final_path = self.output_dir / "checkpoint.json"
        with open(tmp_path, "w") as f:
            json.dump(
                {
                    "completed_indices": sorted(self.completed_indices),
                    "saved_batches": self.saved_batches,
                    "total_samples": self.total_samples,
                },
                f,
            )
        tmp_path.rename(final_path)

    def _load_checkpoint(self) -> bool:
        """Load checkpoint.json if it exists. Returns True if loaded."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return False
        with open(checkpoint_path) as f:
            data = json.load(f)
        self.completed_indices = set(data["completed_indices"])
        self.saved_batches = data["saved_batches"]
        return True

    def _save_batch(self, items: list[tuple[int, DataProto]], batch_idx: int) -> None:
        """Save a batch of items to a pickle file and update state."""
        batch_name = f"batch_{batch_idx:04d}"
        batch_path = self.output_dir / f"{batch_name}.pkl"
        with open(batch_path, "wb") as f:
            pickle.dump(items, f)
        for idx, _ in items:
            self.completed_indices.add(idx)
        self.saved_batches.append(batch_name)
        self._save_checkpoint()

    def _merge_batches(self) -> DataProto:
        """Read all batch files, sort by index, concat, save merged result."""
        all_items: list[tuple[int, DataProto]] = []
        for batch_name in sorted(self.saved_batches):
            with open(self.output_dir / f"{batch_name}.pkl", "rb") as f:
                all_items.extend(pickle.load(f))
        all_items.sort(key=lambda x: x[0])
        merged = DataProto.concat([item[1] for item in all_items])
        merged.save_to_disk(self.output_dir / "trajectories.pkl")
        return merged

    def _upload_artifact(self) -> None:
        """Create trajectories.zip and upload to wandb if available."""
        import zipfile

        zip_path = self.output_dir / "trajectories.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            trajectories_path = self.output_dir / "trajectories.pkl"
            if trajectories_path.exists():
                zf.write(trajectories_path, "trajectories.pkl")
            checkpoint_path = self.output_dir / "checkpoint.json"
            if checkpoint_path.exists():
                zf.write(checkpoint_path, "checkpoint.json")

        if "wandb" in self.tracker.logger:
            wandb = self.tracker.logger["wandb"]
            artifact = wandb.Artifact(name="trajectories", type="trajectories")
            artifact.add_file(str(zip_path), name="trajectories.zip")
            wandb.log_artifact(artifact)

    def _load_data(self) -> None:
        """Load parquet files, apply max_samples, set total_samples."""
        data_config = self.config.data
        data_files = data_config.files
        # Normalize to plain Python list of strings (OmegaConf ListConfig → list)
        if isinstance(data_files, str):
            data_files = [data_files]
        else:
            data_files = list(data_files)

        frames = [pd.read_parquet(str(f)) for f in data_files]
        self._df = pd.concat(frames, axis=0, ignore_index=True)

        max_samples = data_config.get("max_samples", None)
        if max_samples is not None:
            self._df = self._df.head(max_samples)

        self.total_samples = len(self._df)

    def _prepare_prompts(self, indices: list[int]) -> DataProto:
        """Build DataProto with non_tensor_batch for the given sample indices.

        The non_tensor_batch contains the fields that AgentLoopWorker expects:
        - index: sample indices (required by streaming mixin)
        - raw_prompt: chat-format prompts (required by _run_agent_loop)
        - agent_name: which agent loop to use
        - data_source: dataset source identifier (expected by reward postprocessing)
        - reward_model: reward config per sample (expected by reward postprocessing)
        """
        data_config = self.config.data
        prompt_key = data_config.get("prompt_key", "prompt")
        return_raw_chat = data_config.get("return_raw_chat", True)

        prompts_raw = self._df.iloc[indices][prompt_key].tolist()

        # Convert to chat format if needed
        if return_raw_chat:
            # Data already in chat format (list of dicts per sample)
            raw_prompts = prompts_raw
        else:
            # Plain string prompts — wrap as single-turn user message
            raw_prompts = [[{"role": "user", "content": p}] for p in prompts_raw]

        default_agent = self.config.actor_rollout_ref.rollout.agent.default_agent_loop
        n = len(indices)

        non_tensor_batch = {
            "index": np.array(indices, dtype=np.int64),
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "agent_name": np.array([default_agent] * n, dtype=object),
            "data_source": np.array(["generation"] * n, dtype=object),
            "reward_model": np.array([{}] * n, dtype=object),
        }

        return DataProto(
            non_tensor_batch=non_tensor_batch,
            meta_info={"global_steps": -1, "validate": False},
        )

    def run(self) -> None:
        """Main orchestration: load data, dispatch, pull, save, merge, upload."""
        import logging
        from queue import Empty

        logger = logging.getLogger(__name__)

        self._load_data()
        gen_config = self.config.generation

        pending = [i for i in range(self.total_samples) if i not in self.completed_indices]
        logger.info("GenerationRunner: total_samples=%d, pending=%d", self.total_samples, len(pending))

        if not pending:
            if gen_config.final_merge:
                self._merge_batches()
            if gen_config.upload_artifact:
                self._upload_artifact()
            return

        self._queue = Queue()
        manager = GenerationLoopManager(self.config, self._queue)
        prompts = self._prepare_prompts(pending)
        logger.info(
            "GenerationRunner: dispatching %d prompts to %d workers", len(prompts), len(manager.agent_loop_workers)
        )
        worker_refs = manager.dispatch_streaming(prompts)

        # Pull loop
        batch_buffer: list[tuple[int, DataProto]] = []
        batch_idx = len(self.saved_batches)
        save_batch_size = gen_config.save_batch_size
        pull_timeout = gen_config.pull_timeout
        timeout_count = 0

        while len(self.completed_indices) < self.total_samples:
            try:
                idx, result = self._queue.get(block=True, timeout=pull_timeout)
                batch_buffer.append((idx, result))
                self.completed_indices.add(idx)
                timeout_count = 0
                logger.info(
                    "GenerationRunner: received idx=%d (%d/%d)",
                    idx,
                    len(self.completed_indices),
                    self.total_samples,
                )

                if len(batch_buffer) >= save_batch_size:
                    self._save_batch(batch_buffer, batch_idx)
                    batch_buffer = []
                    batch_idx += 1
            except Empty:
                timeout_count += 1
                if batch_buffer:
                    self._save_batch(batch_buffer, batch_idx)
                    batch_buffer = []
                    batch_idx += 1
                # Check if workers have died
                done_refs, pending_refs = ray.wait(worker_refs, timeout=0)
                for ref in done_refs:
                    try:
                        ray.get(ref)
                    except Exception as e:
                        logger.error("Worker failed: %s", e)
                        raise
                if not pending_refs and len(self.completed_indices) < self.total_samples:
                    logger.error(
                        "All workers finished but only %d/%d completed",
                        len(self.completed_indices),
                        self.total_samples,
                    )
                    break

        if batch_buffer:
            self._save_batch(batch_buffer, batch_idx)

        ray.get(worker_refs)
        manager.sleep()
        manager.destroy()

        if gen_config.final_merge:
            self._merge_batches()
        if gen_config.upload_artifact:
            self._upload_artifact()

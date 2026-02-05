"""GenerationLoopManager and GenerationRunner for streaming trajectory generation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

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

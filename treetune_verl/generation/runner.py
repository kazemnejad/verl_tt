# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""GenerationRunner - Core orchestrator for generation-only workflows."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from treetune_verl.generation.checkpoint import CheckpointManager
from treetune_verl.generation.config import GenerationConfig
from treetune_verl.generation.queue import ResultsQueue

logger = logging.getLogger(__name__)


class GenerationRunner:
    """Orchestrator for generation-only workflows.

    Manages the full generation pipeline:
    - Initialize AgentLoopManager (standalone mode)
    - Create ResultsQueue for collecting trajectories
    - Dispatch prompts to workers
    - Pull results and save in batches
    - Manage checkpointing for fault tolerance
    - Final merge of batch files
    - Optional WandB upload
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize generation runner.

        Args:
            config: Hydra config with model, rollout, data, and generation settings.
        """
        self.config = config
        self.gen_config = GenerationConfig(
            output_dir=config.generation.output_dir,
            save_batch_size=config.generation.save_batch_size,
            pull_timeout=config.generation.pull_timeout,
            final_merge=config.generation.final_merge,
            checkpoint_interval=config.generation.checkpoint_interval,
            wandb_upload=config.generation.wandb_upload,
            wandb_project=config.generation.get("wandb_project"),
            wandb_run_name=config.generation.get("wandb_run_name"),
            show_progress=config.generation.show_progress,
        )
        self.output_dir = Path(self.gen_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._initialize()

    def _initialize(self) -> None:
        """Initialize components (separated for testing)."""
        import pyarrow.parquet as pq

        # Resolve tasks to data files if configured
        if self.config.get("tasks"):
            from treetune_verl.tasks import get_dataset_paths

            self.config.data.files = get_dataset_paths(self.config.tasks)

        # Load data from parquet files directly
        data_files = self.config.data.files
        if isinstance(data_files, str):
            data_files = [data_files]

        tables = [pq.read_table(f) for f in data_files]
        if len(tables) > 1:
            import pyarrow as pa

            self.dataframe = pa.concat_tables(tables).to_pandas()
        else:
            self.dataframe = tables[0].to_pandas()

        # Apply max_samples limit
        max_samples = self.config.data.get("max_samples")
        if max_samples and max_samples > 0:
            self.dataframe = self.dataframe.head(max_samples)

        self.prompt_key = self.config.data.prompt_key
        self.total_samples = len(self.dataframe)

        # Initialize checkpoint manager
        config_snapshot = OmegaConf.to_container(self.config, resolve=True)
        self.checkpoint_manager = CheckpointManager(
            output_dir=str(self.output_dir),
            total_samples=self.total_samples,
            config_snapshot=config_snapshot,
        )
        self.checkpoint_manager.load()

        # Results queue and manager will be created on run()
        self.results_queue = None
        self.manager = None

    @staticmethod
    def _adapt_config_for_manager(config: DictConfig) -> DictConfig:
        """Wrap flat generation config into structure AgentLoopManager expects.

        Args:
            config: Flat generation config.

        Returns:
            Nested config with trainer, actor_rollout_ref, reward_model structure.
        """
        # Keep as DictConfig to preserve attribute access
        return OmegaConf.create(
            {
                "trainer": OmegaConf.create(
                    {
                        "n_gpus_per_node": config.n_gpus_per_node,
                        "nnodes": config.nnodes,
                        "project_name": config.get("project_name", "generation"),
                        "experiment_name": config.get("experiment_name", "run"),
                    }
                ),
                "actor_rollout_ref": OmegaConf.create(
                    {
                        "model": config.model,
                        "rollout": config.rollout,
                    }
                ),
                "reward_model": OmegaConf.create({"enable": False}),
                "data": config.data,
            }
        )

    def _create_results_queue(self) -> ray.actor.ActorHandle:
        """Create ResultsQueue Ray actor."""
        return ResultsQueue.remote()

    def _save_batch(self, items: list[tuple[int, Any]], batch_idx: int) -> str:
        """Save a batch of results to pickle file.

        Args:
            items: List of (idx, result) tuples.
            batch_idx: Batch sequence number.

        Returns:
            Batch name (without extension).
        """
        batch_name = f"batch_{batch_idx:04d}"
        batch_path = self.output_dir / f"{batch_name}.pkl"

        with open(batch_path, "wb") as f:
            pickle.dump(items, f)

        # Update checkpoint
        indices = [idx for idx, _ in items]
        self.checkpoint_manager.add_completed(indices)
        self.checkpoint_manager.add_batch(batch_name)
        self.checkpoint_manager.save()

        logger.info(f"Saved batch {batch_name} with {len(items)} samples")
        return batch_name

    def _merge_batches(self) -> None:
        """Merge all batch files into single trajectories.pkl."""
        all_items = []

        for batch_name in self.checkpoint_manager.saved_batches:
            batch_path = self.output_dir / f"{batch_name}.pkl"
            with open(batch_path, "rb") as f:
                batch_items = pickle.load(f)
            all_items.extend(batch_items)

        # Sort by index
        all_items.sort(key=lambda x: x[0])

        merged_path = self.output_dir / "trajectories.pkl"
        with open(merged_path, "wb") as f:
            pickle.dump(all_items, f)

        logger.info(f"Merged {len(all_items)} trajectories to {merged_path}")

    def _upload_to_wandb(self) -> None:
        """Upload trajectories as WandB artifact."""
        import wandb

        # Create or use existing run
        if wandb.run is None:
            wandb.init(
                project=self.gen_config.wandb_project,
                name=self.gen_config.wandb_run_name,
            )

        # Create artifact
        artifact_name = f"trajectories-{wandb.run.name}"
        artifact = wandb.Artifact(artifact_name, type="trajectories")

        # Add files
        if self.gen_config.final_merge:
            artifact.add_file(str(self.output_dir / "trajectories.pkl"))
        else:
            for batch_name in self.checkpoint_manager.saved_batches:
                artifact.add_file(str(self.output_dir / f"{batch_name}.pkl"))

        artifact.add_file(str(self.output_dir / "checkpoint.json"))

        wandb.log_artifact(artifact)
        logger.info(f"Uploaded artifact {artifact_name} to WandB")

    def run(self) -> None:
        """Run the full generation pipeline.

        1. Create ResultsQueue
        2. Create AgentLoopManager
        3. Dispatch prompts
        4. Pull and save batches
        5. Final merge
        6. Optional WandB upload
        """
        import numpy as np

        from verl.experimental.agent_loop.agent_loop import AgentLoopManager
        from verl.protocol import DataProto

        # Get pending indices
        all_indices = list(range(self.total_samples))
        pending_indices = self.checkpoint_manager.get_pending_indices(all_indices)

        if not pending_indices:
            logger.info("All samples already completed")
            if self.gen_config.final_merge:
                self._merge_batches()
            if self.gen_config.wandb_upload:
                self._upload_to_wandb()
            return

        logger.info(f"Starting generation: {len(pending_indices)} pending / {len(all_indices)} total")

        # Create queue and manager
        self.results_queue = self._create_results_queue()
        adapted_config = self._adapt_config_for_manager(self.config)
        self.manager = AgentLoopManager(adapted_config)

        # Prepare batch from pending samples
        pending_df = self.dataframe.iloc[pending_indices]
        prompts = pending_df[self.prompt_key].tolist()

        # Create DataProto with non_tensor_batch (AgentLoopWorker expects this)
        non_tensor_batch = {
            "raw_prompt": np.array(prompts, dtype=object),
            "index": np.array(pending_indices),
        }
        # Add any other columns from dataframe
        for col in pending_df.columns:
            if col != self.prompt_key and col not in non_tensor_batch:
                non_tensor_batch[col] = np.array(pending_df[col].tolist(), dtype=object)

        batch = DataProto(non_tensor_batch=non_tensor_batch)

        # Progress bar
        pbar = None
        if self.gen_config.show_progress:
            pbar = tqdm(
                total=len(pending_indices),
                desc="Generation",
                unit="samples",
            )

        # Generate sequences (blocking call - manager handles async internally)
        output = self.manager.generate_sequences(batch)

        # Save results - output is a DataProto, save it directly
        batch_idx = len(self.checkpoint_manager.saved_batches)
        self._save_batch([(i, output) for i in pending_indices], batch_idx)

        if pbar:
            pbar.update(len(pending_indices))
            pbar.close()

        # Final merge
        if self.gen_config.final_merge:
            self._merge_batches()

        # WandB upload
        if self.gen_config.wandb_upload:
            self._upload_to_wandb()

        logger.info("Generation complete")

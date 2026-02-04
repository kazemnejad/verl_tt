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

"""GenerationRunner: Core orchestrator for generation-only workflows.

Responsibilities:
- Initialize AgentLoopManager (standalone mode)
- Create ResultsQueue for accumulating completed trajectories
- Inject collector into workers
- Dispatch entire dataset (async engine handles parallelism)
- Run pull loop concurrent with generation
- Convert results to DataProto, save in batches
- Manage checkpoint (track completed indices)
- Final merge of batch files

Does NOT handle:
- Analysis (separate scripts)
- Recipe-specific logic (injected via manager class)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import ray
from omegaconf import DictConfig, ListConfig, OmegaConf

from treetune_verl.generation.queue import ResultsQueue

if TYPE_CHECKING:
    from verl.protocol import DataProto

logger = logging.getLogger(__name__)


class GenerationRunner:
    """Core orchestrator for generation-only workflows.

    Converts flat generation config to nested structure expected by
    AgentLoopManager, manages trajectory collection via ResultsQueue,
    handles checkpointing, and saves batches incrementally.

    Usage:
        runner = GenerationRunner(config)
        runner.run()
    """

    def __init__(self, config: DictConfig):
        """Initialize GenerationRunner with configuration.

        Args:
            config: Flat generation config (from generation.yaml).
                Must contain: n_gpus_per_node, nnodes, model, rollout, data, generation
        """
        self.config = config

        # Setup output directory
        gen_config = config.generation
        self.output_dir = Path(gen_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.completed_indices: set[int] = set()
        self.saved_batches: list[str] = []
        self.total_samples: int = 0

        # Store config snapshot for checkpointing
        self.config_snapshot = OmegaConf.to_container(config, resolve=True)

        # Load data
        self._load_data()

        # Try to resume from checkpoint
        if self._load_checkpoint():
            logger.info(
                f"Resumed from checkpoint: {len(self.completed_indices)} completed, {len(self.saved_batches)} batches"
            )

        # Queue will be created in run()
        self._queue: ray.actor.ActorHandle | None = None

    def run(self) -> None:
        """Run the generation workflow.

        Orchestrates the complete generation flow:
        1. Create ResultsQueue actor
        2. Create AgentLoopManager with adapted config
        3. Dispatch generation to workers (workers push to queue)
        4. Run pull loop: collect from queue, save batches, update checkpoint
        5. Final merge and optional WandB upload

        The pull loop runs concurrent with generation dispatch, collecting
        results as they arrive rather than waiting for all to complete.
        """
        from tqdm import tqdm

        from verl.experimental.agent_loop.agent_loop import AgentLoopManager

        gen_config = self.config.generation
        save_batch_size = gen_config.save_batch_size
        pull_timeout = gen_config.pull_timeout
        show_progress = gen_config.show_progress
        checkpoint_interval = gen_config.get("checkpoint_interval", 1)

        # Get pending indices (exclude completed from previous run)
        pending_indices = self._get_pending_indices()

        if not pending_indices:
            logger.info("All samples already completed. Running final steps only.")
            # Skip to final merge/upload
            if gen_config.final_merge:
                self._merge_batches()
            if gen_config.wandb_upload:
                self._upload_to_wandb()
            return

        logger.info(f"Starting generation for {len(pending_indices)} samples")

        # Create ResultsQueue actor
        self._queue = ResultsQueue.remote()

        # Adapt config for AgentLoopManager
        adapted_config = self._adapt_config_for_manager(self.config)

        # Create AgentLoopManager (standalone mode - no worker_group)
        manager = AgentLoopManager(
            config=adapted_config,
            worker_group=None,  # Standalone mode
            rollout_resource_pool=None,  # Will create internally
        )

        # Setup progress bar
        pbar = None
        if show_progress:
            pbar = tqdm(
                total=self.total_samples,
                initial=len(self.completed_indices),
                desc="Generation",
                unit="samples",
            )

        try:
            # Dispatch generation asynchronously
            # In standalone mode, workers will push to queue as they complete
            # We inject the queue handle via the manager's worker creation
            self._dispatch_generation(manager, pending_indices)

            # Pull loop: collect results and save batches
            batch_idx = len(self.saved_batches)  # Resume from last batch
            batches_since_checkpoint = 0

            while len(self.completed_indices) < self.total_samples:
                # Block until batch ready or timeout
                batch = ray.get(
                    self._queue.get_batch.remote(
                        min_items=save_batch_size,
                        timeout=pull_timeout,
                    )
                )

                if batch:
                    # Extract per-sample data from each result
                    items: list[tuple[int, dict]] = []
                    for idx, output in batch:
                        extracted = self._extract_per_sample(output, [idx])
                        items.extend(extracted)

                    if items:
                        # Save batch
                        self._save_batch(items, batch_idx)
                        batch_idx += 1
                        batches_since_checkpoint += 1

                        # Update progress
                        if pbar:
                            pbar.n = len(self.completed_indices)
                            pbar.set_postfix(
                                batches=len(self.saved_batches),
                                pending=ray.get(self._queue.count.remote()),
                            )
                            pbar.refresh()

                        # Save checkpoint periodically
                        if batches_since_checkpoint >= checkpoint_interval:
                            self._save_checkpoint()
                            batches_since_checkpoint = 0

            # Final checkpoint
            self._save_checkpoint()

        finally:
            if pbar:
                pbar.close()

        # Final merge
        if gen_config.final_merge:
            logger.info("Merging batch files...")
            self._merge_batches()

        # WandB upload
        if gen_config.wandb_upload:
            logger.info("Uploading to WandB...")
            self._upload_to_wandb()

        logger.info(f"Generation complete: {len(self.completed_indices)} samples in {len(self.saved_batches)} batches")

    def _dispatch_generation(
        self,
        manager: Any,
        indices: list[int],
    ) -> None:
        """Dispatch generation requests to AgentLoopManager.

        This method prepares the prompts DataProto and calls the manager's
        generate_sequences method. Results are pushed to the queue for the
        pull loop to collect.

        Note: Current implementation is synchronous - calls generate_sequences()
        which waits for all samples, then pushes results to queue. A future
        implementation could use CollectorAgentLoopWorker for true incremental
        push during generation.

        Args:
            manager: AgentLoopManager instance
            indices: Dataset indices to generate for
        """
        import numpy as np
        from tensordict import TensorDict

        from verl.protocol import DataProto

        # Prepare prompts DataProto for generation
        # The agent loop expects raw_prompt in non_tensor_batch
        prompts_list = []
        for idx in indices:
            prompt = self.dataframe.iloc[idx][self.prompt_key]
            prompts_list.append(prompt)

        # Build DataProto matching what AgentLoopWorker.generate_sequences expects
        non_tensor_batch = {
            "raw_prompt": np.array(prompts_list, dtype=object),
            "index": np.array(indices, dtype=np.int64),
        }

        # Create a minimal batch TensorDict (agent loop may not need tensors initially)
        # The actual sequence generation will populate the batch tensors
        batch = TensorDict({}, batch_size=len(indices))

        prompts_proto = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={},
        )

        # Call synchronous generate_sequences
        # This blocks until all samples complete
        logger.info(f"Generating {len(indices)} samples...")
        output = manager.generate_sequences(prompts_proto)

        # Push results to queue for pull loop to collect
        # Each result is pushed individually with its index
        logger.info(f"Pushing {len(output)} results to queue...")
        for i in range(len(output)):
            idx = indices[i]
            # Create a single-sample DataProto view for this index
            single_output = output[i : i + 1]  # DataProto slice
            ray.get(self._queue.put.remote(idx, single_output))

        logger.info("All results pushed to queue")

    @staticmethod
    def _adapt_config_for_manager(config: DictConfig) -> DictConfig:
        """Wrap flat generation config into structure AgentLoopManager expects.

        AgentLoopManager expects a nested config structure with:
        - trainer.n_gpus_per_node, trainer.nnodes, trainer.project_name, etc.
        - actor_rollout_ref.model, actor_rollout_ref.rollout
        - reward_model.enable (must be False for generation-only)
        - data section

        This adapter also adds _target_ fields which are critical for Hydra's
        omega_conf_to_dataclass to work correctly when configs pass through
        Ray serialization.

        Args:
            config: Flat generation config from generation.yaml

        Returns:
            Adapted config with nested structure for AgentLoopManager
        """
        # Deep copy rollout config and add _target_
        rollout_config = OmegaConf.to_container(config.rollout, resolve=True)
        rollout_config["_target_"] = "verl.workers.config.RolloutConfig"

        # Ensure mtp config exists with _target_
        if "mtp" not in rollout_config or rollout_config["mtp"] is None:
            rollout_config["mtp"] = {
                "_target_": "verl.workers.config.MtpConfig",
                "enable": False,
            }
        else:
            # Add _target_ to existing mtp config
            rollout_config["mtp"]["_target_"] = "verl.workers.config.MtpConfig"

        # Ensure agent config has _target_
        if "agent" in rollout_config and rollout_config["agent"] is not None:
            rollout_config["agent"]["_target_"] = "verl.workers.config.AgentLoopConfig"
            # Add custom_async_server defaults if missing
            if (
                "custom_async_server" not in rollout_config["agent"]
                or rollout_config["agent"]["custom_async_server"] is None
            ):
                rollout_config["agent"]["custom_async_server"] = {
                    "_target_": "verl.workers.config.CustomAsyncServerConfig",
                    "path": None,
                    "name": None,
                }

        # Add prometheus config if missing (required by AgentLoopManager)
        if "prometheus" not in rollout_config or rollout_config["prometheus"] is None:
            rollout_config["prometheus"] = {
                "_target_": "verl.workers.config.PrometheusConfig",
                "enable": False,
                "port": 9090,
                "file": "/tmp/prometheus.yml",
                "served_model_name": None,
            }

        # Add trace config if missing
        if "trace" not in rollout_config or rollout_config["trace"] is None:
            rollout_config["trace"] = {
                "_target_": "verl.workers.config.TraceConfig",
                "backend": None,
                "token2text": False,
                "max_samples_per_step_per_worker": None,
            }

        # Add profiler config if missing
        if "profiler" not in rollout_config or rollout_config["profiler"] is None:
            rollout_config["profiler"] = {
                "_target_": "verl.utils.profiler.ProfilerConfig",
                "tool": None,
                "enable": False,
                "all_ranks": False,
                "ranks": [],
                "save_path": None,
                "tool_config": None,
            }

        # Add val_kwargs if missing
        if "val_kwargs" not in rollout_config or rollout_config["val_kwargs"] is None:
            rollout_config["val_kwargs"] = {
                "_target_": "verl.workers.config.SamplingConfig",
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 0,
                "n": 1,
                "do_sample": False,
            }

        # Add multi_turn config if missing
        if "multi_turn" not in rollout_config or rollout_config["multi_turn"] is None:
            rollout_config["multi_turn"] = {
                "_target_": "verl.workers.config.MultiTurnConfig",
                "enable": False,
                "max_assistant_turns": None,
                "tool_config_path": None,
                "max_user_turns": None,
                "max_parallel_calls": 1,
                "max_tool_response_length": 256,
                "tool_response_truncate_side": "middle",
                "interaction_config_path": None,
                "use_inference_chat_template": False,
                "tokenization_sanity_check_mode": "strict",
                "format": "hermes",
                "num_repeat_rollouts": None,
            }

        # Add other required rollout defaults
        rollout_defaults = {
            "mode": "async",
            "ignore_eos": False,
            "enforce_eager": False,
            "cudagraph_capture_sizes": None,
            "expert_parallel_size": 1,
            "max_num_batched_tokens": 8192,
            "max_model_len": None,
            "max_num_seqs": 1024,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "logprobs_mode": "processed_logprobs",
            "scheduling_policy": "fcfs",
            "load_format": "dummy",
            "log_prob_micro_batch_size": None,
            "log_prob_micro_batch_size_per_gpu": None,
            "log_prob_use_dynamic_bsz": False,
            "log_prob_max_token_len_per_gpu": 16384,
            "disable_log_stats": True,
            "do_sample": True,
            "n": 1,
            "over_sample_rate": 0,
            "multi_stage_wake_up": False,
            "engine_kwargs": {"vllm": {}, "sglang": {}, "trtllm": {}},
            "update_weights_bucket_megabytes": 512,
            "skip_rollout": False,
            "skip_dump_dir": "/tmp/rollout_dump",
            "skip_tokenizer_init": True,
            "enable_rollout_routing_replay": False,
            "quantization": None,
            "quantization_config_file": None,
            "dtype": "bfloat16",
        }
        for key, default_value in rollout_defaults.items():
            if key not in rollout_config:
                rollout_config[key] = default_value

        # Deep copy model config and add _target_
        model_config = OmegaConf.to_container(config.model, resolve=True)
        model_config["_target_"] = "verl.workers.config.HFModelConfig"

        # Deep copy data config and add required defaults
        data_config = OmegaConf.to_container(config.data, resolve=True)

        # Add data config defaults expected by SingleTurnAgentLoop
        data_defaults = {
            "tool_config_path": None,
            "return_raw_chat": True,  # Required for agent loop
        }
        for key, default_value in data_defaults.items():
            if key not in data_config:
                data_config[key] = default_value

        # Build the nested structure
        adapted = OmegaConf.create(
            {
                "trainer": {
                    "n_gpus_per_node": config.n_gpus_per_node,
                    "nnodes": config.nnodes,
                    "project_name": config.get("project_name", "generation"),
                    "experiment_name": config.get("experiment_name", "run"),
                },
                "actor_rollout_ref": {
                    "model": model_config,
                    "rollout": rollout_config,
                },
                "reward_model": {
                    "enable": False,
                    "use_reward_loop": False,
                    "enable_resource_pool": False,
                },
                "data": data_config,
            }
        )

        return adapted

    # =========================================================================
    # Checkpoint Methods
    # =========================================================================

    def _save_checkpoint(self) -> None:
        """Write checkpoint state to output_dir/checkpoint.json.

        Checkpoint contains:
        - completed_indices: List of sample indices that have been saved
        - saved_batches: List of batch file names (without .pkl extension)
        - total_samples: Total samples in dataset
        - config_snapshot: Generation config for validation on resume

        The completed_indices are sorted for deterministic output.
        """
        checkpoint_data = {
            "completed_indices": sorted(self.completed_indices),
            "saved_batches": self.saved_batches,
            "total_samples": self.total_samples,
            "config_snapshot": self.config_snapshot,
        }

        checkpoint_path = self.output_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

    def _load_checkpoint(self) -> bool:
        """Load checkpoint state if file exists.

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists.

        On successful load, restores:
        - self.completed_indices (as set)
        - self.saved_batches
        """
        checkpoint_path = self.output_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return False

        with open(checkpoint_path) as f:
            data = json.load(f)

        self.completed_indices = set(data["completed_indices"])
        self.saved_batches = data["saved_batches"]
        # Note: total_samples from checkpoint can be used for validation
        # but we don't overwrite self.total_samples as it should come from dataset

        return True

    def _get_pending_indices(self) -> list[int]:
        """Return indices not yet completed.

        Returns:
            List of indices from 0 to total_samples-1 that are not in
            completed_indices, in ascending order.
        """
        return [i for i in range(self.total_samples) if i not in self.completed_indices]

    def _validate_batches(self) -> tuple[list[str], list[str]]:
        """Validate that all saved batch files exist and are readable.

        Returns:
            Tuple of (missing_batches, corrupt_batches) where:
            - missing_batches: batch names where .pkl file doesn't exist
            - corrupt_batches: batch names where .pkl file exists but can't be loaded
        """
        missing: list[str] = []
        corrupt: list[str] = []

        for batch_name in self.saved_batches:
            batch_path = self.output_dir / f"{batch_name}.pkl"

            if not batch_path.exists():
                missing.append(batch_name)
                continue

            # Try to load to verify it's not corrupt
            try:
                with open(batch_path, "rb") as f:
                    pickle.load(f)
            except (pickle.UnpicklingError, EOFError, Exception):
                corrupt.append(batch_name)

        return missing, corrupt

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def _extract_per_sample(self, output: DataProto, indices: list[int]) -> list[tuple[int, dict]]:
        """Extract individual samples from batched DataProto.

        Converts 2D batched tensors [batch_size, seq_len] into list of
        1D tensors [seq_len] per sample, along with non-tensor data.

        Args:
            output: DataProto with batched generation output
            indices: Dataset indices corresponding to each sample in the batch

        Returns:
            List of (index, sample_dict) tuples where sample_dict contains:
            - Tensor fields as 1D tensors (sliced from batch dimension)
            - Non-tensor fields as individual items (not arrays)
        """
        items: list[tuple[int, dict]] = []
        batch_size = len(output)

        for i in range(batch_size):
            sample_dict: dict = {}

            # Extract tensor fields (1D slice from 2D batch)
            if output.batch is not None:
                for key in output.batch.keys():
                    sample_dict[key] = output.batch[key][i]

            # Extract non-tensor fields (individual items)
            for key, val in output.non_tensor_batch.items():
                sample_dict[key] = val[i]

            items.append((indices[i], sample_dict))

        return items

    def _save_batch(self, items: list[tuple[int, dict]], batch_idx: int) -> str:
        """Save batch of samples to pickle file and update state.

        Args:
            items: List of (index, sample_dict) tuples from _extract_per_sample
            batch_idx: Sequential batch index for filename

        Returns:
            Batch name (e.g., "batch_0000") without .pkl extension

        Side effects:
            - Writes batch_NNNN.pkl to output_dir
            - Adds indices to self.completed_indices
            - Appends batch name to self.saved_batches
        """
        batch_name = f"batch_{batch_idx:04d}"
        batch_path = self.output_dir / f"{batch_name}.pkl"

        # Save to pickle
        with open(batch_path, "wb") as f:
            pickle.dump(items, f)

        # Update state
        for idx, _ in items:
            self.completed_indices.add(idx)
        self.saved_batches.append(batch_name)

        return batch_name

    def _merge_batches(self) -> None:
        """Merge all saved batch files into trajectories.pkl.

        Reads all batch files, combines items, sorts by index,
        and writes to trajectories.pkl in output_dir.

        The final file contains list of (index, sample_dict) tuples
        sorted by index for consistent ordering.
        """
        all_items: list[tuple[int, dict]] = []

        # Load all batches
        for batch_name in self.saved_batches:
            batch_path = self.output_dir / f"{batch_name}.pkl"
            with open(batch_path, "rb") as f:
                batch_items = pickle.load(f)
            all_items.extend(batch_items)

        # Sort by index
        all_items.sort(key=lambda x: x[0])

        # Write merged file
        trajectories_path = self.output_dir / "trajectories.pkl"
        with open(trajectories_path, "wb") as f:
            pickle.dump(all_items, f)

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _load_data(self) -> None:
        """Load parquet files and store as DataFrame.

        Handles two modes:
        1. Direct files: Uses config.data.files paths directly
        2. Task system: If config.tasks is set, resolves via get_dataset_paths()

        Sets instance variables:
        - self.dataframe: pandas DataFrame with loaded data
        - self.prompt_key: column name for prompts (from config.data.prompt_key)
        - self.total_samples: number of samples (after max_samples limit)
        """
        # Determine file paths - task system or direct
        if self.config.tasks is not None:
            # Resolve tasks to parquet file paths
            from treetune_verl.tasks import get_dataset_paths

            task_list = list(self.config.tasks)
            files = get_dataset_paths(task_list)
        else:
            # Use files directly from config
            files = self.config.data.files
            if not isinstance(files, list | ListConfig):
                files = [files]

        # Load and concatenate all parquet files
        dataframes = []
        for filepath in files:
            df = pd.read_parquet(filepath)
            dataframes.append(df)

        self.dataframe = pd.concat(dataframes, axis=0, ignore_index=True)

        # Apply max_samples limit if set
        max_samples = self.config.data.get("max_samples", None)
        if max_samples is not None:
            self.dataframe = self.dataframe.head(max_samples)

        # Set instance variables
        self.prompt_key = self.config.data.prompt_key
        self.total_samples = len(self.dataframe)

    # =========================================================================
    # WandB Upload
    # =========================================================================

    def _upload_to_wandb(self) -> None:
        """Upload trajectory files to WandB as an artifact.

        Creates a WandB artifact containing:
        - trajectories.pkl (if final_merge=True) or batch files (if final_merge=False)
        - checkpoint.json

        Upload behavior:
        - If WandB run already active: upload to current run
        - If no active run: create new run using wandb_project / wandb_run_name

        Artifact spec:
        - Type: "trajectories"
        - Name: "trajectories-<run_name>"
        """
        import wandb

        gen_config = self.config.generation

        # Determine run name for artifact naming
        run_name = gen_config.get("wandb_run_name") or "generation"

        # Create artifact
        artifact = wandb.Artifact(
            name=f"trajectories-{run_name}",
            type="trajectories",
        )

        # Add trajectory files based on final_merge setting
        if gen_config.final_merge:
            # Add merged trajectories file
            trajectories_path = self.output_dir / "trajectories.pkl"
            if trajectories_path.exists():
                artifact.add_file(str(trajectories_path))
        else:
            # Add individual batch files
            for batch_name in self.saved_batches:
                batch_path = self.output_dir / f"{batch_name}.pkl"
                if batch_path.exists():
                    artifact.add_file(str(batch_path))

        # Always add checkpoint
        checkpoint_path = self.output_dir / "checkpoint.json"
        if checkpoint_path.exists():
            artifact.add_file(str(checkpoint_path))

        # Upload to existing or new run
        if wandb.run is not None:
            # Use existing active run
            wandb.run.log_artifact(artifact)
        else:
            # Create new run for upload
            run = wandb.init(
                project=gen_config.wandb_project,
                name=gen_config.wandb_run_name,
            )
            run.log_artifact(artifact)
            run.finish()

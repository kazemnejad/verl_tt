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
import pickle
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from verl.protocol import DataProto


class GenerationRunner:
    """Core orchestrator for generation-only workflows.

    Converts flat generation config to nested structure expected by
    AgentLoopManager, manages trajectory collection via ResultsQueue,
    handles checkpointing, and saves batches incrementally.
    """

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

        # Deep copy model config and add _target_
        model_config = OmegaConf.to_container(config.model, resolve=True)
        model_config["_target_"] = "verl.workers.config.HFModelConfig"

        # Deep copy data config
        data_config = OmegaConf.to_container(config.data, resolve=True)

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

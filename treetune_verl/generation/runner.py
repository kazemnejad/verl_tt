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

from omegaconf import DictConfig, OmegaConf


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

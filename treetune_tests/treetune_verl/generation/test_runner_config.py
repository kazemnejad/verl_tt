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

"""Unit tests for GenerationRunner config adapter.

Tests the _adapt_config_for_manager static method that converts flat
generation configs into the nested structure AgentLoopManager expects.

The adapter is critical because:
1. Our generation config is flat (easier to use)
2. AgentLoopManager expects nested structure (trainer, actor_rollout_ref, etc.)
3. _target_ fields are needed for Hydra's omega_conf_to_dataclass when
   configs pass through Ray serialization
"""

from omegaconf import DictConfig, OmegaConf

from treetune_verl.generation.runner import GenerationRunner


def make_flat_config(**overrides) -> DictConfig:
    """Create a minimal flat generation config for testing.

    Returns a config that mimics the structure from generation.yaml.
    """
    base = {
        "n_gpus_per_node": 8,
        "nnodes": 1,
        "project_name": "test_project",
        "experiment_name": "test_experiment",
        "model": {
            "path": "/path/to/model",
            "trust_remote_code": True,
        },
        "rollout": {
            "name": "sglang",
            "tensor_model_parallel_size": 2,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "prompt_length": 1024,
            "response_length": 4096,
            "calculate_log_probs": True,
            "agent": {
                "num_workers": 8,
                "default_agent_loop": "single_turn_agent",
            },
        },
        "data": {
            "files": ["/path/to/data.parquet"],
            "prompt_key": "prompt",
        },
        "generation": {
            "output_dir": "/output",
            "save_batch_size": 1000,
        },
    }
    config = OmegaConf.create(base)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))
    return config


class TestAdaptConfigForManager:
    """Tests for GenerationRunner._adapt_config_for_manager static method."""

    def test_adapt_config_adds_target_fields(self):
        """_target_ added to rollout, model, and mtp configs.

        _target_ fields are critical for Hydra's omega_conf_to_dataclass
        to work correctly when configs pass through Ray serialization.
        """
        flat_config = make_flat_config()

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        # Check model has _target_
        assert "_target_" in adapted.actor_rollout_ref.model
        assert adapted.actor_rollout_ref.model._target_ == "verl.workers.config.HFModelConfig"

        # Check rollout has _target_
        assert "_target_" in adapted.actor_rollout_ref.rollout
        assert adapted.actor_rollout_ref.rollout._target_ == "verl.workers.config.RolloutConfig"

        # Check mtp has _target_ (should be created with defaults if missing)
        assert "mtp" in adapted.actor_rollout_ref.rollout
        assert "_target_" in adapted.actor_rollout_ref.rollout.mtp
        assert adapted.actor_rollout_ref.rollout.mtp._target_ == "verl.workers.config.MtpConfig"

    def test_adapt_config_preserves_rollout_params(self):
        """Rollout params (temperature, top_p, etc.) preserved through adaptation."""
        flat_config = make_flat_config()

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        rollout = adapted.actor_rollout_ref.rollout

        # Verify sampling params preserved
        assert rollout.temperature == 0.8
        assert rollout.top_p == 0.95
        assert rollout.top_k == 50
        assert rollout.prompt_length == 1024
        assert rollout.response_length == 4096

        # Verify other rollout params
        assert rollout.name == "sglang"
        assert rollout.tensor_model_parallel_size == 2
        assert rollout.gpu_memory_utilization == 0.85
        assert rollout.calculate_log_probs is True

        # Verify agent config
        assert rollout.agent.num_workers == 8
        assert rollout.agent.default_agent_loop == "single_turn_agent"

    def test_adapt_config_creates_trainer_structure(self):
        """Creates trainer struct with n_gpus_per_node, nnodes, project/experiment names."""
        flat_config = make_flat_config()

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        # Verify trainer structure
        assert "trainer" in adapted
        assert adapted.trainer.n_gpus_per_node == 8
        assert adapted.trainer.nnodes == 1
        assert adapted.trainer.project_name == "test_project"
        assert adapted.trainer.experiment_name == "test_experiment"

    def test_adapt_config_disables_reward_model(self):
        """reward_model.enable = False for generation-only mode."""
        flat_config = make_flat_config()

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        # Verify reward_model disabled
        assert "reward_model" in adapted
        assert adapted.reward_model.enable is False
        assert adapted.reward_model.use_reward_loop is False
        assert adapted.reward_model.enable_resource_pool is False

    def test_adapt_config_handles_missing_mtp(self):
        """Creates default mtp config if missing from rollout."""
        # Config without mtp in rollout
        flat_config = make_flat_config()
        # Ensure mtp is not in the config
        assert "mtp" not in flat_config.rollout

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        # Should create default mtp config
        assert "mtp" in adapted.actor_rollout_ref.rollout
        mtp = adapted.actor_rollout_ref.rollout.mtp
        assert "_target_" in mtp
        assert mtp._target_ == "verl.workers.config.MtpConfig"
        assert mtp.enable is False

    def test_adapt_config_preserves_existing_mtp(self):
        """Preserves existing mtp config when present, adding _target_ if missing."""
        flat_config = make_flat_config(
            rollout={
                "mtp": {
                    "enable": True,
                    "enable_train": False,
                    "enable_rollout": True,
                    "num_speculative_tokens": 2,
                }
            }
        )

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        mtp = adapted.actor_rollout_ref.rollout.mtp
        assert mtp.enable is True
        assert mtp.enable_rollout is True
        assert mtp.num_speculative_tokens == 2
        # _target_ should be added
        assert mtp._target_ == "verl.workers.config.MtpConfig"

    def test_adapt_config_preserves_model_params(self):
        """Model config params preserved through adaptation."""
        flat_config = make_flat_config()

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        model = adapted.actor_rollout_ref.model
        assert model.path == "/path/to/model"
        assert model.trust_remote_code is True

    def test_adapt_config_preserves_data_section(self):
        """Data config preserved in adapted config."""
        flat_config = make_flat_config()

        adapted = GenerationRunner._adapt_config_for_manager(flat_config)

        # Data section should be preserved (needed by AgentLoopManager for dataset)
        assert "data" in adapted
        assert adapted.data.files == ["/path/to/data.parquet"]
        assert adapted.data.prompt_key == "prompt"

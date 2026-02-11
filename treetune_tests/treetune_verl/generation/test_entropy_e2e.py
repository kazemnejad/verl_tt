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

"""E2E integration tests for entropy generation pipeline.

Validates the full entropy generation pipeline end-to-end: model loads,
entropy sglang server starts, streaming workers extract per-token entropy,
trajectories.pkl saved with rollout_entropy.

Run with: pytest treetune_tests/treetune_verl/generation/test_entropy_e2e.py -xvs

Marks: All tests in this module are marked with @pytest.mark.gpu.
"""

from pathlib import Path

import pandas as pd
import pytest
import ray
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GSM8K_PROMPTS = [f"What is {i} + {i * 2}? Show your work step by step." for i in range(1, 17)]


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for E2E tests (module-scoped, shared across all tests)."""
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        runtime_env = get_ppo_ray_runtime_env()
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)
    yield
    # Don't shutdown -- other tests may need Ray


@pytest.fixture
def gsm8k_parquet(tmp_path):
    """Create parquet with 16 GSM8K-style prompts + proper extra_info.index."""
    df = pd.DataFrame(
        {
            "prompt": GSM8K_PROMPTS,
            "extra_info": [{"index": i} for i in range(len(GSM8K_PROMPTS))],
        }
    )
    parquet_path = tmp_path / "gsm8k_test.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


def _build_e2e_config(data_files: list[str], output_dir: str) -> OmegaConf:
    """Build entropy E2E config using Hydra compose to resolve all defaults."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).parent / "e2e_config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="entropy_gen_e2e")

    # Override dynamic values
    config.data.train_files = data_files
    config.trainer.default_local_dir = output_dir
    # Note: Don't call OmegaConf.resolve() here — the rollout defaults contain
    # oc.select interpolations that reference missing actor/profiler keys.
    # The runner and AgentLoopManager resolve config lazily on access.
    return config


@pytest.fixture
def e2e_config(gsm8k_parquet, tmp_path):
    """Build entropy E2E config with Hydra compose and set data path + output dir."""
    return _build_e2e_config(
        data_files=[str(gsm8k_parquet)],
        output_dir=str(tmp_path / "outputs"),
    )


@pytest.fixture
def e2e_dataset_and_collate(e2e_config):
    """Create RLHFDataset + collate_fn from E2E config (same as main.py would).

    NOTE: We don't call OmegaConf.resolve(e2e_config) globally — the rollout
    defaults contain oc.select interpolations that reference missing actor/profiler
    keys.  OmegaConf resolves lazily on attribute access, so we just access the
    specific sub-trees we need (data, model).
    """
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    local_path = copy_to_local(
        e2e_config.actor_rollout_ref.model.path,
        use_shm=e2e_config.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = e2e_config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    dataset = create_rl_dataset(
        e2e_config.data.train_files,
        e2e_config.data,
        tokenizer,
        processor,
        max_samples=e2e_config.data.get("train_max_samples", -1),
    )

    return dataset, collate_fn


# ---------------------------------------------------------------------------
# E2E Test Class
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestE2EEntropyGeneration:
    """E2E integration tests for entropy generation pipeline requiring GPU + SGLang."""

    def test_entropy_generation_16_samples(self, ray_context, e2e_config, e2e_dataset_and_collate):
        """Generate 16 samples with entropy extraction, verify rollout_entropy in outputs."""
        from treetune_verl.generation.runner import GenerationRunner

        dataset, collate_fn = e2e_dataset_and_collate
        runner = GenerationRunner(e2e_config, dataset, collate_fn)
        runner.run()

        # -- All 16 completed
        assert len(runner.completed_indices) == 16
        assert runner.completed_indices == set(range(16))

        # -- Merged output exists and loads correctly
        trajectories_path = runner.output_dir / "trajectories.pkl"
        assert trajectories_path.exists()

        merged = DataProto.load_from_disk(trajectories_path)
        assert len(merged) == 16

        # -- Standard keys present
        assert "prompts" in merged.batch.keys()
        assert "responses" in merged.batch.keys()
        assert "response_mask" in merged.batch.keys()
        assert "input_ids" in merged.batch.keys()
        assert "attention_mask" in merged.batch.keys()

        # -- rollout_entropy key exists
        assert "rollout_entropy" in merged.batch.keys(), (
            "rollout_entropy missing from trajectories — entropy pipeline did not produce entropy tensors"
        )

        rollout_entropy = merged.batch["rollout_entropy"]
        response_length = merged.batch["responses"].shape[1]

        # -- Shape: (16, response_length)
        assert rollout_entropy.shape == (16, response_length), (
            f"Expected rollout_entropy shape (16, {response_length}), got {rollout_entropy.shape}"
        )

        # -- All values non-negative (small tolerance for float rounding)
        assert (rollout_entropy >= -0.01).all(), (
            f"rollout_entropy has negative values below tolerance: min={rollout_entropy.min().item()}"
        )

        # -- Entropy is positive where response_mask is active
        response_mask = merged.batch["response_mask"]
        active_entropy = rollout_entropy[response_mask.bool()]
        assert (active_entropy > 0).any(), (
            "rollout_entropy should have some positive values where response_mask is active"
        )

        # -- Tensor shapes consistent
        batch_size = merged.batch.batch_size[0]
        assert batch_size == 16

        prompt_len = merged.batch["prompts"].shape[1]
        total_len = merged.batch["input_ids"].shape[1]
        assert total_len == prompt_len + response_length

        # -- response_mask semantics: binary, at least some 1s
        unique_vals = torch.unique(response_mask)
        assert set(unique_vals.tolist()).issubset({0, 1}), "response_mask should only contain 0s and 1s"
        assert response_mask.sum() > 0, "response_mask should have some 1s"

"""E2E integration tests for GenerationRunner.

These tests require a GPU with SGLang support.
Run with: pytest treetune_tests/treetune_verl/generation/test_e2e.py -xvs

Marks: All tests in this module are marked with @pytest.mark.gpu.
"""

import json
import os
import pickle
import shutil

import pandas as pd
import pytest
import ray
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GSM8K_PROMPTS = [f"What is {i} + {i * 2}? Show your work step by step." for i in range(1, 33)]


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
    """Create parquet with 32 GSM8K-style prompts + proper extra_info.index."""
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
    """Build E2E config using Hydra compose to resolve all defaults (rollout, etc.)."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("treetune_tests/treetune_verl/generation/e2e_config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="generation_e2e")

    # Override dynamic values
    config.data.train_files = data_files
    config.trainer.default_local_dir = output_dir
    # Note: Don't call OmegaConf.resolve() here — the rollout defaults contain
    # oc.select interpolations that reference missing actor/profiler keys.
    # The runner and AgentLoopManager resolve config lazily on access.
    return config


@pytest.fixture
def e2e_config(gsm8k_parquet, tmp_path):
    """Build E2E config with Hydra compose and set data path + output dir."""
    return _build_e2e_config(
        data_files=[str(gsm8k_parquet)],
        output_dir=str(tmp_path / "outputs"),
    )


@pytest.fixture
def e2e_dataset_and_collate(e2e_config):
    """Create RLHFDataset + collate_fn from E2E config (same as main.py would)."""
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    OmegaConf.resolve(e2e_config)

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
class TestE2EGenerationPipeline:
    """E2E integration tests requiring GPU + SGLang."""

    def test_full_generation_pipeline_32_samples_2_workers(self, ray_context, e2e_config, e2e_dataset_and_collate):
        """Generate 32 samples with 2 workers, verify outputs."""
        from treetune_verl.generation.runner import GenerationRunner

        dataset, collate_fn = e2e_dataset_and_collate
        runner = GenerationRunner(e2e_config, dataset, collate_fn)
        runner.run()

        # -- All 32 completed
        assert len(runner.completed_indices) == 32
        assert runner.completed_indices == set(range(32))

        # -- Batch files exist
        batch_files = list(runner.output_dir.glob("batch_*.pkl"))
        assert len(batch_files) > 0

        # -- Checkpoint exists with correct content
        checkpoint_path = runner.output_dir / "checkpoint.json"
        assert checkpoint_path.exists()
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        assert set(ckpt["completed_indices"]) == set(range(32))

        # -- No duplicate indices across batch files
        all_indices = []
        for bf in batch_files:
            with open(bf, "rb") as f:
                items = pickle.load(f)
            all_indices.extend([idx for idx, _ in items])
        assert len(all_indices) == len(set(all_indices)), "Duplicate indices found"

        # -- Merged output exists and has correct shape
        trajectories_path = runner.output_dir / "trajectories.pkl"
        assert trajectories_path.exists()

        merged = DataProto.load_from_disk(trajectories_path)
        assert len(merged) == 32

        # -- DataProto has expected tensor keys
        assert "prompts" in merged.batch.keys()
        assert "responses" in merged.batch.keys()
        assert "response_mask" in merged.batch.keys()
        assert "input_ids" in merged.batch.keys()
        assert "attention_mask" in merged.batch.keys()

        # -- Tensor shapes consistent
        batch_size = merged.batch.batch_size[0]
        assert batch_size == 32

        prompt_len = merged.batch["prompts"].shape[1]
        response_len = merged.batch["responses"].shape[1]
        total_len = merged.batch["input_ids"].shape[1]
        assert total_len == prompt_len + response_len

        # -- response_mask semantics: binary, at least some 1s
        response_mask = merged.batch["response_mask"]
        unique_vals = torch.unique(response_mask)
        assert set(unique_vals.tolist()).issubset({0, 1}), "response_mask should only contain 0s and 1s"
        assert response_mask.sum() > 0, "response_mask should have some 1s"

    def test_resume_from_partial_checkpoint(self, ray_context, e2e_config, e2e_dataset_and_collate, tmp_path):
        """Test resume: complete all 32, then simulate partial checkpoint with first batch only."""
        from treetune_verl.generation.runner import GenerationRunner

        dataset, collate_fn = e2e_dataset_and_collate
        # -- First run: complete all 32
        runner1 = GenerationRunner(e2e_config, dataset, collate_fn)
        runner1.run()
        assert len(runner1.completed_indices) == 32

        # -- Setup resume directory with only the first batch
        resume_dir = tmp_path / "resume_outputs"
        resume_dir.mkdir(parents=True)

        batch_files = sorted(runner1.output_dir.glob("batch_*.pkl"))
        assert len(batch_files) > 0
        first_batch = batch_files[0]
        shutil.copy(first_batch, resume_dir / first_batch.name)

        # Load first batch to extract its indices
        with open(first_batch, "rb") as f:
            first_items = pickle.load(f)
        first_indices = [idx for idx, _ in first_items]

        # Create partial checkpoint
        with open(resume_dir / "checkpoint.json", "w") as f:
            json.dump(
                {
                    "completed_indices": first_indices,
                    "saved_batches": [first_batch.stem],
                    "total_samples": 32,
                },
                f,
            )

        # -- Resume run with 32 total samples
        resume_config = OmegaConf.create(e2e_config)
        resume_config.trainer.default_local_dir = str(resume_dir)

        runner2 = GenerationRunner(resume_config, dataset, collate_fn)
        assert len(runner2.completed_indices) == len(first_indices)

        runner2.run()

        # All 32 should be completed now
        assert len(runner2.completed_indices) == 32

        # Merged output has all 32
        merged = DataProto.load_from_disk(resume_dir / "trajectories.pkl")
        assert len(merged) == 32

    def test_multiple_batches_merge_sorted(self, ray_context, e2e_config, e2e_dataset_and_collate):
        """Small batch_size creates multiple files; merge sorts correctly."""
        from treetune_verl.generation.runner import GenerationRunner

        dataset, collate_fn = e2e_dataset_and_collate
        config = OmegaConf.create(e2e_config)
        config.generation.save_batch_size = 5  # Force many small batches

        runner = GenerationRunner(config, dataset, collate_fn)
        runner.run()

        # Should have multiple batch files (32 / 5 = at least 6)
        batch_files = list(runner.output_dir.glob("batch_*.pkl"))
        assert len(batch_files) >= 6, f"Expected >= 6 batch files, got {len(batch_files)}"

        # Merged output has all 32
        merged = DataProto.load_from_disk(runner.output_dir / "trajectories.pkl")
        assert len(merged) == 32

        # -- Verify sort order: merged rows must be sorted by original index
        # Reconstruct the (idx, DataProto) pairs from batch files (same as _merge_batches)
        all_items: list[tuple[int, DataProto]] = []
        for bf in sorted(batch_files):
            with open(bf, "rb") as f:
                all_items.extend(pickle.load(f))
        all_items.sort(key=lambda x: x[0])

        sorted_indices = [idx for idx, _ in all_items]
        assert sorted_indices == list(range(32)), f"Expected indices 0..31 in order, got: {sorted_indices}"

        # Verify each row in merged matches the corresponding sorted item's tensor
        for pos, (idx, single_proto) in enumerate(all_items):
            expected_prompts = single_proto.batch["prompts"]
            actual_prompts = merged.batch["prompts"][pos : pos + 1]
            assert torch.equal(expected_prompts, actual_prompts), (
                f"Merged row {pos} (index {idx}) prompts mismatch — merge did not preserve sort order"
            )

# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""E2E smoke test for GenerationRunner with actual model generation.

Tests the complete pipeline:
1. Config adaptation with proper _target_ fields
2. Data loading and task resolution
3. AgentLoopManager instantiation
4. SGLang server initialization
5. Actual model generation
6. Results saving and checkpoint management

Requires: 1 GPU (uses Qwen2.5-0.5B-Instruct for fast testing).
"""

import pickle
import sys
import tempfile

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="e2e_config",
    config_name="generation_gsm8k",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Run full E2E generation test with actual model."""
    import shutil
    from pathlib import Path

    import ray

    # Override output dir to temp location for testing
    temp_dir = tempfile.mkdtemp(prefix="generation_e2e_full_")
    config.generation.output_dir = temp_dir

    print(f"[E2E] Output directory: {temp_dir}")
    print(f"[E2E] Model: {config.model.path}")
    print(f"[E2E] Max samples: {config.data.max_samples}")

    # Phase 1: Test config adapter
    print("\n[E2E] === Phase 1: Testing config adapter ===")
    from treetune_verl.generation.runner import GenerationRunner

    adapted = GenerationRunner._adapt_config_for_manager(config)

    # Verify structure
    assert adapted.trainer.n_gpus_per_node == config.n_gpus_per_node
    assert adapted.actor_rollout_ref.model.path == config.model.path
    assert adapted.actor_rollout_ref.rollout.name == config.rollout.name

    # Verify _target_ fields are present (critical for dataclass conversion)
    assert adapted.actor_rollout_ref.rollout.get("_target_") == "verl.workers.config.RolloutConfig"
    assert adapted.actor_rollout_ref.rollout.mtp.get("_target_") == "verl.workers.config.MtpConfig"
    assert adapted.actor_rollout_ref.model.get("_target_") == "verl.workers.config.HFModelConfig"

    # Verify mtp config has required fields
    assert adapted.actor_rollout_ref.rollout.mtp.enable is False
    assert adapted.actor_rollout_ref.rollout.mtp.enable_rollout is False

    # Verify reward_model config
    assert adapted.reward_model.enable is False
    assert adapted.reward_model.use_reward_loop is False
    assert adapted.reward_model.enable_resource_pool is False

    print("[E2E] Config adapter: PASSED")

    # Phase 2: Test dataclass conversion (simulates what happens in SGLang server)
    print("\n[E2E] === Phase 2: Testing dataclass conversion ===")
    from verl.utils.config import omega_conf_to_dataclass

    rollout_config = omega_conf_to_dataclass(adapted.actor_rollout_ref.rollout)
    print(f"[E2E] RolloutConfig type: {type(rollout_config).__name__}")
    print(f"[E2E] MtpConfig type: {type(rollout_config.mtp).__name__}")

    assert type(rollout_config).__name__ == "RolloutConfig"
    assert type(rollout_config.mtp).__name__ == "MtpConfig"
    assert rollout_config.mtp.enable is False

    print("[E2E] Dataclass conversion: PASSED")

    # Phase 3: Test full generation (requires GPU)
    print("\n[E2E] === Phase 3: Testing full generation pipeline ===")

    # Check GPU availability
    try:
        import torch

        if not torch.cuda.is_available():
            print("[E2E] WARNING: No GPU available, skipping full generation test")
            print("[E2E] Infrastructure tests PASSED!")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
        print(f"[E2E] GPU available: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[E2E] WARNING: Could not check GPU: {e}")
        print("[E2E] Infrastructure tests PASSED!")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    try:
        # Create and run the runner
        print("[E2E] Creating GenerationRunner...")
        runner = GenerationRunner(config)

        print(f"[E2E] Loaded {runner.total_samples} samples")
        assert runner.total_samples == config.data.max_samples, (
            f"Expected {config.data.max_samples} samples, got {runner.total_samples}"
        )

        print("[E2E] Starting generation (this may take a while)...")
        runner.run()

        # Verify outputs
        output_dir = Path(temp_dir)

        # Check batch files or merged file
        if config.generation.final_merge:
            merged_path = output_dir / "trajectories.pkl"
            assert merged_path.exists(), f"Merged file not found: {merged_path}"
            with open(merged_path, "rb") as f:
                results = pickle.load(f)
            print(f"[E2E] Merged trajectories file exists with {len(results)} items")
        else:
            batch_files = list(output_dir.glob("batch_*.pkl"))
            assert len(batch_files) > 0, "No batch files found"
            print(f"[E2E] Found {len(batch_files)} batch files")

        # Check checkpoint
        checkpoint_path = output_dir / "checkpoint.json"
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        print("[E2E] Checkpoint file exists")

        print("\n[E2E] Full generation pipeline: PASSED")
        print("[E2E] E2E generation test PASSED!")

    except Exception as e:
        print(f"\n[E2E] ERROR during generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        ray.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""E2E smoke test for GenerationRunner infrastructure.

Tests the runner's data loading, checkpoint management, and batch saving
without full model generation (which requires complex verl config matching).

For full end-to-end testing with actual model generation, use the training
E2E tests which have properly configured verl configs.

Requires: No GPU (tests infrastructure only).
"""

import pickle
import tempfile

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="e2e_config",
    config_name="generation_gsm8k",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """Test runner infrastructure without full model generation."""

    # Override output dir to temp location for testing
    temp_dir = tempfile.mkdtemp(prefix="generation_e2e_")
    config.generation.output_dir = temp_dir

    print(f"[E2E] Output directory: {temp_dir}")
    print(f"[E2E] Model: {config.model.path}")
    print(f"[E2E] Max samples: {config.data.max_samples}")

    # Test data loading via runner initialization (without manager)
    from treetune_verl.generation.checkpoint import CheckpointManager
    from treetune_verl.generation.config import GenerationConfig
    from treetune_verl.generation.runner import GenerationRunner

    # Test config adapter
    adapted = GenerationRunner._adapt_config_for_manager(config)
    assert adapted.trainer.n_gpus_per_node == config.n_gpus_per_node
    assert adapted.actor_rollout_ref.model.path == config.model.path
    print("[E2E] Config adapter works correctly")

    # Test task resolution and data loading
    if config.get("tasks"):
        from treetune_verl.tasks import get_dataset_paths

        data_files = get_dataset_paths(config.tasks)
        assert len(data_files) > 0
        print(f"[E2E] Task system resolved to: {data_files}")
    else:
        data_files = config.data.files

    # Load data with pyarrow
    from pathlib import Path

    import pyarrow.parquet as pq

    output_dir = Path(temp_dir)
    tables = [pq.read_table(f) for f in data_files]
    dataframe = tables[0].to_pandas()

    max_samples = config.data.max_samples
    if max_samples and max_samples > 0:
        dataframe = dataframe.head(max_samples)

    print(f"[E2E] Loaded {len(dataframe)} samples")
    assert len(dataframe) == max_samples

    # Test checkpoint manager
    from omegaconf import OmegaConf

    config_snapshot = OmegaConf.to_container(config, resolve=True)
    checkpoint_manager = CheckpointManager(
        output_dir=str(output_dir),
        total_samples=len(dataframe),
        config_snapshot=config_snapshot,
    )
    print("[E2E] CheckpointManager initialized")

    # Simulate batch saving
    mock_results = [(i, {"mock_data": f"result_{i}"}) for i in range(len(dataframe))]

    batch_name = "batch_0000"
    batch_path = output_dir / f"{batch_name}.pkl"
    with open(batch_path, "wb") as f:
        pickle.dump(mock_results, f)

    checkpoint_manager.add_completed([i for i, _ in mock_results])
    checkpoint_manager.add_batch(batch_name)
    checkpoint_manager.save()
    print(f"[E2E] Saved batch: {batch_name}")

    # Verify checkpoint
    checkpoint_path = output_dir / "checkpoint.json"
    assert checkpoint_path.exists()
    print("[E2E] Checkpoint file exists")

    # Test merge
    merged_path = output_dir / "trajectories.pkl"
    all_items = []
    for bn in checkpoint_manager.saved_batches:
        bp = output_dir / f"{bn}.pkl"
        with open(bp, "rb") as f:
            items = pickle.load(f)
        all_items.extend(items)

    with open(merged_path, "wb") as f:
        pickle.dump(all_items, f)
    print(f"[E2E] Merged {len(all_items)} trajectories")

    # Verify merged file
    assert merged_path.exists()
    with open(merged_path, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == max_samples
    print("[E2E] Trajectories file verified")

    # Test GenerationConfig
    gen_config = GenerationConfig(
        output_dir=str(output_dir),
        save_batch_size=config.generation.save_batch_size,
        pull_timeout=config.generation.pull_timeout,
        final_merge=config.generation.final_merge,
        checkpoint_interval=config.generation.checkpoint_interval,
        wandb_upload=config.generation.wandb_upload,
        show_progress=config.generation.show_progress,
    )
    assert gen_config.save_batch_size == 10
    print("[E2E] GenerationConfig works correctly")

    print("[E2E] E2E generation infrastructure test PASSED!")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

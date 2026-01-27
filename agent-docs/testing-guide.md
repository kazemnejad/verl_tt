---
summary: Testing practices, directory structure, CI workflows, utilities, and patterns.
read_when:
  - Writing or debugging tests
  - Understanding test categories
  - Setting up distributed tests
  - Running E2E tests
---

# Testing in verl — Practices, Paradigms & Utilities

## Overview

verl uses **pytest** as its testing framework. Tests are organized by the verl subnamespace they exercise, with special folders for distributed/E2E/sanity tests.
For writing tests for our modifications/implementations, prefix them with `treetune_` according to the package path. Similarly, `treetune_recipe_` for our treetune recipe tests.

| Suffix | Accelerator |
|--------|-------------|
| `*_on_cpu.py` | CPU only |
| `*.py` (no suffix) | GPU |

## Directory Structure

```
tests/
├── checkpoint_engine/      # verl/checkpoint_engine tests
├── experimental/           # verl/experimental tests (agent_loop, reward_loop, vla)
├── interactions/           # verl/interactions tests
├── models/                 # verl/models tests
├── single_controller/      # verl/single_controller tests
├── trainer/                # verl/trainer tests
│   ├── config/             # Config validation tests
│   └── ppo/                # PPO algorithm tests
├── utils/                  # verl/utils tests
│   ├── ckpt/               # Checkpoint utilities
│   ├── dataset/            # Dataset utilities
│   ├── debug/              # Debug utilities
│   ├── megatron/           # Megatron utilities
│   └── reward_score/       # Reward scoring utilities
├── workers/                # verl/workers tests
│   ├── actor/              # Actor worker tests
│   ├── config/             # Worker config tests
│   ├── critic/             # Critic worker tests
│   ├── reward_manager/     # Reward manager tests
│   └── rollout/            # Rollout worker tests
├── special_distributed/    # Multi-GPU unit tests
├── special_e2e/            # End-to-end training tests
├── special_npu/            # NPU-specific tests
├── special_sanity/         # Quick sanity checks
└── special_standalone/     # Tests requiring dedicated environments
```

## Test Categories

### 1. CPU Unit Tests (`*_on_cpu.py`)
Pure logic tests that don't need GPU.

```python
# tests/trainer/ppo/test_core_algos_on_cpu.py
def test_register_new_function():
    @register_adv_est("test_estimator")
    def test_fn():
        pass
    assert "test_estimator" in ADV_ESTIMATOR_REGISTRY
```

### 2. GPU Unit Tests
Tests requiring GPU resources.

```python
# tests/workers/actor/test_special_dp_actor.py
# Executed via: torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py
```

### 3. Distributed Tests (`special_distributed/`)
Multi-GPU unit tests. Use `torch.multiprocessing` or `torchrun`.

```python
# Pattern: spawn workers, init process group, run distributed ops
def _worker(rank, world_size, init_method, ...):
    dist.init_process_group(backend=get_nccl_backend(), ...)
    # ... test logic ...
    dist.destroy_process_group()

def test_distributed_feature(tmp_path):
    mp.spawn(_worker, args=(...), nprocs=world_size, join=True)
```

### 4. E2E Tests (`special_e2e/`)
Full training pipeline tests using shell scripts. Typically require 8+ GPUs.

```bash
# tests/special_e2e/run_grpo_lora_with_merge.sh
#!/usr/bin/env bash
set -xeuo pipefail

# 1. Train
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...

# 2. Merge
python3 -m verl.model_merger merge --backend fsdp ...

# 3. Assert
file_path="checkpoints/.../adapter_model.safetensors"
[ -f "$file_path" ] || { echo "Error: File not found!"; exit 1; }
```

### 5. Sanity Tests (`special_sanity/`)
Quick checks: imports, license headers, docstrings, etc.

```python
# tests/special_sanity/test_import.py
def test_import():
    import verl
    print(verl.__version__)
```

## Common Testing Patterns

### 1. Pytest Fixtures for Ray

```python
import pytest
import ray

@pytest.fixture
def ray_init_shutdown():
    ray.init(num_cpus=100)
    yield
    ray.shutdown()

def test_with_ray(ray_init_shutdown):
    # ray is initialized and will be cleaned up
    resource_pool = RayResourcePool([2], use_gpu=False)
    ...
```

### 2. DataProto Testing

```python
from verl import DataProto
import torch

def test_dataproto_operations():
    # Create from dict
    obs = torch.randn(100, 10)
    labels = ["a", "b", "c"] * 33 + ["d"]
    data = DataProto.from_dict(
        tensors={"obs": obs},
        non_tensors={"labels": labels},
        meta_info={"name": "test"}
    )
    
    # Test operations
    chunks = data.chunk(2)
    assert len(chunks) == 2
    
    reconstructed = DataProto.concat(chunks)
    torch.testing.assert_close(reconstructed.batch["obs"], data.batch["obs"])
```

### 3. Config Validation Testing

```python
import unittest
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig

class TestActorConfig(unittest.TestCase):
    def test_validation_exceptions(self):
        with self.assertRaises((ValueError, AssertionError)) as cm:
            ActorConfig(
                strategy="fsdp",
                loss_agg_mode="invalid-mode",
                ...
            )
        self.assertIn("Invalid loss_agg_mode", str(cm.exception))
```

### 4. Registry/Decorator Testing

```python
def test_register_new_function():
    """Test registering a new function with a string name"""
    @register_adv_est("test_estimator")
    def test_fn():
        pass
    assert "test_estimator" in ADV_ESTIMATOR_REGISTRY

def test_duplicate_registration_different_function():
    """Test that registering different functions with same name raises"""
    @register_adv_est("conflict_test")
    def fn1():
        pass
    with pytest.raises(ValueError):
        @register_adv_est("conflict_test")
        def fn2():
            pass
```

### 5. Parameterized Tests

```python
@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size, seq_len, num_groups, seed):
    torch.manual_seed(seed)
    # ... test logic ...
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
```

### 6. Version-Conditional Tests

```python
from packaging.version import parse as parse_version
import tensordict

@pytest.mark.skipif(
    parse_version(tensordict.__version__) < parse_version("0.10"),
    reason="requires at least tensordict 0.10"
)
def test_to_tensordict():
    ...
```

## Utilities

### Random Mask Creation
```python
from verl.utils.model import create_random_mask

attention_mask = create_random_mask(
    input_ids=input_ids,
    max_ratio_of_left_padding=0.1,
    max_ratio_of_valid_token=0.9,
    min_ratio_of_valid_token=0.5
)
```

### Device Utilities
```python
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device

get_torch_device().set_device(rank)
device = f"{get_device_name()}:{rank}"
```

### TensorDict Utilities
```python
from verl.utils import tensordict_utils as tu

# Contiguous for in-place modifications
data = tu.contiguous(data)
```

## Running Tests Locally

```bash
# CPU unit tests
pytest -s -x tests/*_on_cpu.py

# GPU unit tests
pytest -s -x tests/

# Specific distributed test
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    tests/workers/actor/test_special_dp_actor.py

# Specific E2E test
bash tests/special_e2e/run_grpo_lora_with_merge.sh
```

## Best Practices

1. **File naming**: Use `test_<feature>_on_cpu.py` for CPU tests; `test_<feature>.py` for GPU.
2. **Fixtures**: Use pytest fixtures for setup/teardown (Ray, distributed env, temp files).
3. **Assertions**: Prefer `torch.testing.assert_close()` for tensor comparisons with tolerances.
4. **Isolation**: Each test should be independent; clean up state in teardown.
5. **Documentation**: Add docstrings explaining what the test validates.
6. **Edge cases**: Test boundary conditions, error paths, and validation exceptions.
7. **Distributed tests**: Use `mp.spawn()` for multi-process; `torchrun` for multi-GPU scripts.
8. **E2E tests**: Use shell scripts with `set -xeuo pipefail` for fail-fast behavior.

## Adding New Tests

1. **Determine category**: CPU unit, GPU unit, distributed, E2E, or sanity.
2. **Choose location**: Match directory to verl subnamespace being tested.
3. **Name appropriately**: Add `_on_cpu` suffix for CPU-only tests.
4. **Use fixtures**: Leverage existing fixtures for Ray, distributed setup.

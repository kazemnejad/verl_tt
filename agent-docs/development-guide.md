---
summary: How to develop and implement research ideas on top of verl without breaking upstream sync.
read_when:
  - Implementing any new feature or modification
  - Creating a new recipe or experiment
  - Adding shared utilities to treetune_verl
  - Deciding where to put new code
  - Understanding the extension strategy (subclass, config injection, monkey-patch)
---

# Development Guide

Guide for developing research ideas on top of verl. **Follow strictly.**

## Golden Rule: Never Touch Upstream

Modifications to the upstream verl codebase are **forbidden** unless the implementation plan explicitly says otherwise. This preserves sync-ability with the upstream repo.

**Upstream (read-only) directories:**

| Directory    | Contents                        |
| ------------ | ------------------------------- |
| `verl/`      | Core framework                  |
| `recipe/`    | Upstream recipes                |
| `tests/`     | Upstream tests                  |
| `docs/`      | Official documentation          |
| `examples/`  | Example launch scripts          |
| `scripts/`   | Upstream utility scripts        |

## Parallel Directory Layout

All custom work lives in `treetune_*` mirrors:

| Upstream     | Custom mirror          | Purpose                                               |
| ------------ | ---------------------- | ----------------------------------------------------- |
| `verl/`      | `treetune_verl/`       | Shared custom implementations (workers, algos, utils) |
| `recipe/`    | `treetune_recipe/`     | Self-contained experiments / algorithms                |
| `tests/`     | `treetune_tests/`      | Tests for all treetune code                           |
| `scripts/`   | `treetune_scripts/`    | One-off / general-purpose scripts                     |

## Extension Strategy (Priority Order)

When you need to change verl behavior, try these approaches **in order**:

### 1. Config-based injection (best)
verl's Hydra config system and registries allow plugging in custom classes without touching any source. Check if the component you need to modify supports a configurable `_target_`, a registry lookup (e.g. `adv_estimator`, `reward_manager`), or a class reference passed as a parameter.

**Example** — PRIME registers a custom reward manager via config:
```yaml
# recipe/prime/config/prime_trainer.yaml
reward_model:
  reward_manager: prime          # looked up in verl's reward manager registry
algorithm:
  adv_estimator: rloo            # looked up in ADV_ESTIMATOR_REGISTRY
```

### 2. Subclassing (preferred)
Import the upstream class and override only the methods you need.

**Example** — PRIME subclasses the trainer:
```python
# recipe/prime/prime_ray_trainer.py
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

class RayPRIMETrainer(RayPPOTrainer):
    def _create_dataloader(self, ...):
        # custom data filtering logic
        ...

    def _save_checkpoint(self):
        # custom checkpoint logic
        ...
```

### 3. Monkey-patching (last resort)
When subclassing cannot reach the code path, patch the module/class/function at runtime. Contain patches in a single, clearly-named module (e.g. `patches.py`) so they're easy to find and remove later.

**Example** — applying model-level optimizations at runtime:
```python
from verl.models.transformers.monkey_patch import apply_monkey_patch
apply_monkey_patch(model=reward_module, ulysses_sp_size=sp_size, ...)
```

### What NOT to do
- **Do not copy-paste** entire files from `verl/` into `treetune_*/`. This creates drift and makes upstream merges painful.
- **Do not modify upstream files** for convenience. If you think an upstream change is warranted, propose it separately.

## `treetune_verl/` — Shared Library

Reusable implementations shared across recipes: custom workers, algorithm variants, utility functions, data processing, etc.

- Mirror `verl/`'s subdirectory structure where applicable (e.g. `treetune_verl/workers/`, `treetune_verl/trainer/`).
- Keep `__init__.py` files in place so the package is importable.
- Import from `verl` for base classes; import from `treetune_verl` for shared custom code.

## `treetune_recipe/` — Experiments & Algorithms

Each recipe is a **self-contained unit** for a specific algorithm, baseline, or data pipeline. A recipe may contain:

```
treetune_recipe/<recipe_name>/
├── config/
│   └── <recipe_name>_trainer.yaml   # Inherits ppo_trainer.yaml via Hydra defaults
├── <recipe_name>_ray_trainer.py     # Custom trainer (subclasses RayPPOTrainer)
├── <recipe_name>_fsdp_workers.py    # Custom workers (if needed)
├── <recipe_name>_core_algos.py      # Custom algorithm logic (if needed)
├── main_<recipe_name>.py            # Hydra entrypoint
└── run_<recipe_name>_*.sh           # Launch scripts
```

### Config pattern
Inherit the base config and override only what you need:
```yaml
# treetune_recipe/my_algo/config/my_algo_trainer.yaml
hydra:
  searchpath:
    - file://verl/trainer/config        # so Hydra finds ppo_trainer.yaml

defaults:
  - ppo_trainer                          # inherit everything
  - _self_                               # then apply overrides

algorithm:
  adv_estimator: my_custom_estimator
```

### Entrypoint pattern
```python
# treetune_recipe/my_algo/main_my_algo.py
import hydra
from .my_algo_ray_trainer import RayMyAlgoTrainer

@hydra.main(config_path="config", config_name="my_algo_trainer", version_base=None)
def main(config):
    ...

if __name__ == "__main__":
    main()
```

### Trainer subclass pattern
```python
# treetune_recipe/my_algo/my_algo_ray_trainer.py
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

class RayMyAlgoTrainer(RayPPOTrainer):
    # override only what differs
    ...
```

Recipes can import from both `verl` (upstream) and `treetune_verl` (shared custom code).

## `treetune_tests/` — Custom Tests

Follows `agent-docs/testing-guide.md` and `tests/README.md` strictly.

- Mirror directory structure: `treetune_tests/treetune_verl/...`, `treetune_tests/treetune_recipe/...`.
- Naming: `test_<feature>.py` (GPU), `test_<feature>_on_cpu.py` (CPU-only).
- Use pytest fixtures for Ray init/shutdown, distributed setup.
- Use `torch.testing.assert_close()` for tensor comparisons.
- E2E tests: shell scripts with `set -xeuo pipefail`.
- **PYTHONPATH in scripts**: `python3 <script>` sets `sys.path[0]` to the script's directory, not `cwd`. E2E and standalone scripts must export the project root so `treetune_verl` / `treetune_recipe` are importable:
  ```bash
  export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
  ```

## `treetune_scripts/` — Utility Scripts

Single-purpose, standalone scripts not tied to a specific recipe or test. Examples: data preprocessing, result analysis, one-off evaluations. One script per file.


## Recommended Checklist: Adding a New Recipe

1. Create `treetune_recipe/<name>/` with the directory structure above.
2. Write a config YAML that inherits core configs and overrides only what's needed.
3. Start implementing the functionality. override minimal methods.
4. Create `main_<name>.py` entrypoint with `@hydra.main`.
5. Add a launch script (`run_<name>_*.sh`).
6. Add tests under
7. Ensure `__init__.py` files exist for all packages.

## Recommended Checklist: Adding Shared Code to `treetune_verl`

1. Place the module under the appropriate subdirectory mirroring `verl/`.
2. Subclass or wrap the upstream class; do not copy it.
3. Add tests under `treetune_tests/`.
4. Import from `treetune_verl` in recipes that need it.

## Code Linting and Formatting

Pre-commit hooks enforce style. Usage:
```bash
pre-commit run                # staged changes only
pre-commit run --all-files    # entire repo
# single hook:
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

Run before committing.

## License Headers

Every `.py` file must contain a recognized license header — enforced by the `check-license` pre-commit hook (`tests/special_sanity/check_license.py`). Commits will fail without one.

For our (treetune) code, use:
```python
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
```

Other recognized headers (upstream): Bytedance, PRIME, SGLang, ModelBest, Amazon, Facebook, Meituan, Huawei. See `check_license.py` for the full list.

## Implementation Approach

1. Study the docs, code, spec, implementation plan.
2. Implement failing unit tests.
3. Implement the functionality/logic/code/configs/etc.
4. Run the tests and keep developing/debugging until you ensure they pass.
5. Create a debug script to run the recipe/experiment/etc with small data/batch size/etc (end to end testing).
6. If all tests pass, commit the changes. Refactor if needed while ensuring tests pass.

## Cross-References

- **Architecture & abstractions**: `agent-docs/verl-framework-guide.md`
- **Training loop lifecycle**: `agent-docs/ppo-trainer-lifecycle.md`
- **Worker internals**: `agent-docs/actor-rollout-ref-worker.md`
- **Config system**: `agent-docs/config-system.md`
- **Running code**: `agent-docs/running-code.md`
- **Testing**: `agent-docs/testing-guide.md`

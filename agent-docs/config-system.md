---
summary: How YAML configs work—inheritance, overrides, validation via Hydra and dataclasses.
read_when:
  - Creating or modifying configs
  - Debugging config issues
  - Understanding Hydra defaults and inheritance
  - Adding new config fields
---

# verl's Config System

verl uses Hydra for hierarchical config composition and Python dataclasses for validation.

## Config Structure

```
verl/trainer/config/
├── ppo_trainer.yaml          # Main trainer config (entry point)
├── actor/
│   ├── actor.yaml            # Base actor config
│   ├── dp_actor.yaml         # FSDP actor (inherits actor.yaml)
│   └── megatron_actor.yaml   # Megatron actor (inherits actor.yaml)
├── rollout/
├── ref/
├── critic/
├── optim/
├── engine/
└── ...
```

## Inheritance via Hydra `defaults`

Configs inherit from other configs using Hydra's `defaults` list. Order matters: later entries override earlier ones.

**Example:** `ppo_trainer.yaml` (entry point)
```yaml
defaults:
  - actor@actor_rollout_ref.actor: dp_actor    # Load dp_actor.yaml into actor_rollout_ref.actor
  - rollout@actor_rollout_ref.rollout: rollout
  - _self_                                      # Apply this file's fields last
```

**Example:** `actor/dp_actor.yaml` (component config)
```yaml
defaults:
  - ../optim@optim: fsdp        # Load ../optim/fsdp.yaml into optim
  - ../engine@fsdp_config: fsdp # Load ../engine/fsdp.yaml into fsdp_config
  - actor                       # Inherit base actor.yaml
  - _self_

_target_: verl.workers.config.FSDPActorConfig
strategy: fsdp
grad_clip: 1.0
```

**Syntax:** `<folder>@<target_path>: <file>` loads `<folder>/<file>.yaml` into `<target_path>` in the config tree.

## User Overrides

User configs (e.g., in `examples/`) inherit from trainer configs and override specific fields:

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config   # Where to find base configs

defaults:
  - ppo_trainer                    # Inherit full ppo_trainer.yaml
  - _self_

# Override specific fields
data:
  max_prompt_length: 1024
actor_rollout_ref:
  rollout:
    name: sglang
```

## Validation via Dataclasses

The `_target_` field links YAML configs to Python dataclasses for type checking and validation.

**Example:** `verl/trainer/config/config.py`
```python
@dataclass
class RewardManagerConfig(BaseConfig):
    source: str = "register"
    name: str = "naive"
    module: Optional[ModuleConfig] = None

    def __post_init__(self):
        # Validation logic
        if self.source == "register":
            assert self.name in REWARD_MANAGER_REGISTRY
```

`BaseConfig` (`verl/base_config.py`) provides:
- Dict-like access (`config["field"]` and `config.field`)
- Immutability by default (fields frozen after init)
- `_mutable_fields` whitelist for runtime-modifiable fields

## Variable Interpolation

Hydra supports cross-references using `${}`:

```yaml
actor:
  use_remove_padding: ${oc.select:actor_rollout_ref.model.use_remove_padding,false}
  #                    ^^^^^^^^^ fallback to false if path doesn't exist
```

## Task Definitions — Composable List Items via Interpolation

Hydra lists replace on override — you can't merge individual items from separate files. The `task_defs` pattern solves this by defining tasks as named dicts in a single file, then composing them into lists via `${}` interpolation.

### Setup

Task definitions live in `treetune_verl/tasks/config/task_definitions.yaml`. Each task is a named entry:

```yaml
# task_definitions.yaml
_math_like_cls: &math_like_cls        # YAML anchor for DRY
  path: treetune_verl/tasks/math_like_task.py
  name: MathLikeTask

aime_2025:
  custom_cls: *math_like_cls
  loading_params:
    args: [MathArena/aime_2025]
    kwargs: { split: train }
  problem_key: problem
  answer_key: answer
  data_source: aime_2025

deepscaler:
  custom_cls: *math_like_cls
  loading_params: ...
  data_source: deepscaler
```

### Usage in recipe configs

Add the searchpath and load via `@task_defs` package directive:

```yaml
hydra:
  searchpath:
    - file://treetune_verl/tasks/config    # find task_definitions.yaml

defaults:
  - task_definitions@task_defs             # load into config.task_defs
  - _self_

train_tasks:
  - ${task_defs.aime_2025}                 # compose by reference
  - ${task_defs.deepscaler}
```

### CLI overrides

```bash
# Override a task property (propagates through interpolation)
python main.py task_defs.aime_2025.problem_key=question

# Recompose the task list
python main.py 'train_tasks=[${task_defs.aime_2025},${task_defs.konkur}]'
```

### Adding new task definitions

Add a new named entry to `task_definitions.yaml`. It becomes immediately available as `${task_defs.<name>}` in any recipe that loads the file. No other files need to change.

## Complete Config Reference

For the complete config reference (all fields: data, actor_rollout_ref, critic, reward_model, algorithm, trainer), see `docs/examples/config.rst`.
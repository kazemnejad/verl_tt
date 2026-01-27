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

## Complete Config Reference

For the complete config reference (all fields: data, actor_rollout_ref, critic, reward_model, algorithm, trainer), see `docs/examples/config.rst`.
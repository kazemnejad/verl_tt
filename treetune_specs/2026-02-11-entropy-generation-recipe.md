# Entropy Generation Recipe

## Goal

Generate trajectories with per-token entropy from a reasoning model. Enables analysis of token entropy patterns in CoT reasoning (reproducing Section 3 of "Beyond the 80/20 Rule").

## Design

Compose existing components in a recipe — zero new abstractions. The entropy extraction pipeline (`treetune_verl/sglang/`, `treetune_verl/agent_loop/entropy_*`) and the generation runner (`treetune_verl/generation/`) already exist independently. This recipe wires them together.

## Recipe Structure

```
treetune_recipe/entropy_gen/
├── __init__.py
├── main.py              # Hydra entrypoint → run_generation(config)
├── agent_loop.py        # StreamingEntropyWorker + EntropyGenerationLoopManager
└── config/
    └── entropy_gen.yaml # Inherits generation + task_definitions, entropy overrides
```

### `agent_loop.py`

Two classes, both thin composition:

```python
from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin
from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker
from treetune_verl.agent_loop.entropy_manager import _load_entropy_replica
from treetune_verl.generation.runner import GenerationLoopManager
import ray
from ray.util.queue import Queue


class StreamingEntropyWorker(StreamingAgentLoopWorkerMixin, EntropyAgentLoopWorker):
    """Streaming generation with per-token entropy extraction."""
    pass


class EntropyGenerationLoopManager(GenerationLoopManager):
    """GenerationLoopManager wired with entropy-aware replica + worker."""

    def __init__(self, config, queue: Queue):
        self.rollout_replica_class = _load_entropy_replica()
        self.agent_loop_workers_class = ray.remote(StreamingEntropyWorker)
        super().__init__(config, queue)
```

`StreamingEntropyWorker` inherits:
- From `StreamingAgentLoopWorkerMixin`: `generate_sequences_streaming()`, `set_queue()`
- From `EntropyAgentLoopWorker`: `_agent_loop_postprocess()` (entropy → padded tensor), `_postprocess()` (entropy → `DataProto["rollout_entropy"]`)

`EntropyGenerationLoopManager` sets entropy-aware replica and worker classes before calling `super().__init__()`. Inherits `dispatch_streaming()` and `destroy()` from `GenerationLoopManager`.

**Requires**: `GenerationLoopManager.__init__` must guard its default assignments with `hasattr` checks so subclasses can pre-set these attributes (see Changes to Existing Code below).

### `main.py`

Standard entrypoint — identical pattern to `math_gen/main.py`. No extra args; everything config-driven.

```python
import hydra
from verl.utils.device import auto_set_device
from treetune_verl.generation.main import run_generation


@hydra.main(config_path="config", config_name="entropy_gen", version_base=None)
def main(config):
    auto_set_device(config)
    run_generation(config)


if __name__ == "__main__":
    main()
```

### `config/entropy_gen.yaml`

```yaml
hydra:
  searchpath:
    - file://treetune_verl/generation/config
    - file://treetune_verl/tasks/config
    - file://verl/trainer/config

defaults:
  - generation
  - task_definitions@task_defs
  - _self_

trainer:
  project_name: entropy_gen
  experiment_name: ${now:%Y%m%d_%H%M%S}

actor_rollout_ref:
  model:
    path: ???
  rollout:
    entropy_top_k: 0  # 0 = full-vocab entropy
    agent:
      default_agent_loop: entropy_single_turn_agent

generation:
  custom_manager_cls: treetune_recipe.entropy_gen.agent_loop.EntropyGenerationLoopManager

train_tasks:
  - ${task_defs.hmmt_feb_2025}
```

## Changes to Existing Code

### 1. `GenerationLoopManager.__init__` — guard defaults for subclassing

In `treetune_verl/generation/runner.py`, guard the default worker class assignment so subclasses can pre-set it:

```python
# Before:
self.agent_loop_workers_class = ray.remote(StreamingAgentLoopWorker)

# After:
if not hasattr(self, 'agent_loop_workers_class'):
    self.agent_loop_workers_class = ray.remote(StreamingAgentLoopWorker)
```

This allows subclasses (like `EntropyGenerationLoopManager`) to set `agent_loop_workers_class` before calling `super().__init__()` without it being overwritten.

### 2. `GenerationRunner.run()` — custom manager class support

In `treetune_verl/generation/runner.py`, where `GenerationLoopManager` is instantiated, add dynamic class loading:

```python
from verl.utils.import_utils import load_class_from_fqn

# In run():
cls_fqn = self.config.generation.get("custom_manager_cls", None)
if cls_fqn:
    manager_cls = load_class_from_fqn(cls_fqn, description="GenerationLoopManager")
else:
    manager_cls = GenerationLoopManager

manager = manager_cls(self.config, self._queue)
```

### 3. `generation.yaml` — add default for new field

```yaml
generation:
  custom_manager_cls: null   # FQN of custom GenerationLoopManager subclass
  # ... existing fields unchanged
```

### 4. `task_definitions.yaml` — add HMMT Feb 2025

```yaml
hmmt_feb_2025:
  custom_cls: *math_like_cls
  loading_params:
    args: [MathArena/hmmt_feb_2025]
    kwargs: { split: train }
  problem_key: problem
  answer_key: answer
  prompt_template: *math_prompt
  data_source: hmmt_feb_2025
```

30 problems from Harvard-MIT Mathematics Tournament February 2025. Same schema as AIME tasks (`problem`/`answer` fields). Source: [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025).

## Output

`trajectories.pkl` contains a `DataProto` with:

```
batch:
  prompts:          [N, prompt_length]
  responses:        [N, response_length]
  response_mask:    [N, response_length]
  rollout_log_probs:[N, response_length]
  rollout_entropy:  [N, response_length]   ← per-token entropy, padded with 0.0
  input_ids:        [N, total_length]
  attention_mask:   [N, total_length]
  position_ids:     [N, total_length]

non_tensor_batch:
  index, raw_prompt, data_source, reward_model, ...
```

`rollout_entropy` is masked by `response_mask` (0.0 beyond actual sequence length).

## Data Flow

```
entropy_gen.yaml
  → custom_manager_cls = ...EntropyGenerationLoopManager
  → rollout.entropy_top_k = 0
  → rollout.agent.default_agent_loop = entropy_single_turn_agent

GenerationRunner.run()
  → load_class_from_fqn(custom_manager_cls)
  → EntropyGenerationLoopManager(config, queue)
    → EntropySGLangReplica (entropy-aware sglang server)
    → StreamingEntropyWorker (streaming + entropy postprocessing)

StreamingEntropyWorker.generate_sequences_streaming()
  → EntropySingleTurnAgentLoop.run()
    → EntropySGLangHttpServer.generate() → EntropyTokenOutput.entropy
    → extra_fields["response_entropy"] = [float, ...]
  → EntropyAgentLoopWorker._agent_loop_postprocess()
    → padded tensor [1, response_length]
  → EntropyAgentLoopWorker._postprocess()
    → DataProto["rollout_entropy"]
  → queue.put((idx, DataProto))

GenerationRunner pull loop
  → save batches → merge → trajectories.pkl
```

## CLI

```bash
# Run entropy generation on HMMT Feb 2025
python -m treetune_recipe.entropy_gen.main \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  trainer.n_gpus_per_node=1

# Override dataset
python -m treetune_recipe.entropy_gen.main \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  train_tasks='[${task_defs.aime_2025}]'

# Override entropy mode
python -m treetune_recipe.entropy_gen.main \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.rollout.entropy_top_k=50
```

## Implementation Order

1. Add `hasattr` guard in `GenerationLoopManager.__init__` for subclass-friendly defaults
2. Add `custom_manager_cls: null` default to `generation.yaml`
3. Add `load_class_from_fqn` dispatch in `GenerationRunner.run()`
4. Add `hmmt_feb_2025` to `task_definitions.yaml`
5. Create `treetune_recipe/entropy_gen/` (agent_loop.py, main.py, config, __init__.py)

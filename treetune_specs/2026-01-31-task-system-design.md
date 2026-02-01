# Task System Design

## Problem

verl requires preprocessed `.parquet` files. Every dataset needs a standalone script that downloads from HuggingFace, transforms rows, saves to parquet — then you point verl at the files. Two disconnected steps, extra scripts to maintain, friction when switching datasets.

## Solution

A **Task** class that loads HuggingFace datasets, constructs prompts from config, caches the result as parquet, and feeds cached paths into verl's existing pipeline. Resolved in the entrypoint — trainer is unaware.

## Flow

```
YAML config (train_tasks / val_tasks)
    → Task.get_parquet_path()
        → cache hit?  → return path
        → cache miss? → load_dataset() → build prompts → save parquet → return path
    → patched into config.data.train_files / config.data.val_files
        → standard verl RLHFDataset → Trainer
```

## Directory layout

```
treetune_verl/tasks/
├── __init__.py       # public API
├── task.py           # Task base class
└── cache.py          # cache key computation, source file hashing
```

Task subclasses live in recipes or alongside their use site:
```
treetune_recipe/math/
├── math_task.py      # MathTask(Task) with answer parsing
├── config/
│   └── math_grpo.yaml
└── main.py
```

## Task base class

`Task` is initialized with a config dict and an optional `cache_dir`. Cache dir resolution priority: constructor arg → `TREETUNE_TASK_CACHE_DIR` env var → `~/.cache/treetune/tasks/`.

### Key methods

- **`build_dataset()`**: Calls `load_dataset(*loading_params.args, **loading_params.kwargs)`, then applies `_make_map_fn()` over every row. Returns an HF `Dataset`. Subclasses override this or `_make_map_fn()` for custom preprocessing.

- **`get_parquet_path()`**: Computes a cache key from the class source file hash + config hash. On cache hit, returns the path. On cache miss, calls `build_dataset()`, saves to parquet, returns the path.

- **`_make_map_fn()`**: Returns a per-row transform function that reads config and produces the output row. The base implementation handles prompt construction (see below). Each output row contains at minimum `data_source`, `prompt` (chat messages), and `extra_info` (with `index` + any `extra_fields` from the original dataset).

## Config structure

### YAML shape

```yaml
train_tasks:
  - loading_params:
      args: ["openai/gsm8k"]
      kwargs:
        name: main
        split: train
    prompt_format: template
    prompt_template: "{question}"
    system_prompt: "You are a math tutor. Solve step by step."
    data_source: "gsm8k"
    extra_fields: ["answer"]

val_tasks:
  - loading_params:
      args: ["openai/gsm8k"]
      kwargs:
        name: main
        split: test
    prompt_format: template
    prompt_template: "{question}"
    data_source: "gsm8k"
    extra_fields: ["answer"]
```

### Config keys

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `loading_params.args` | yes | — | Positional args for `load_dataset` (at minimum the dataset path) |
| `loading_params.kwargs` | no | `{}` | Keyword args for `load_dataset` (`name`, `split`, `revision`, `data_dir`, `data_files`, etc.) |
| `prompt_format` | no | `"template"` | `"template"` or `"chat_messages"` |
| `prompt_template` | no | `"{}"` | Python format string referencing dataset columns directly |
| `system_prompt` | no | `null` | System message prepended to prompt |
| `chat_messages_field` | no | `"messages"` | Column name holding message list (only used in `chat_messages` mode) |
| `data_source` | no | `"unknown"` | Identifier for reward function dispatch |
| `extra_fields` | no | `[]` | Dataset columns to carry into `extra_info` |
| `custom_cls.path` | no | `null` | Path to Python file containing a Task subclass |
| `custom_cls.name` | no | `"Task"` | Class name to load from that file |

### `loading_params`

Passed directly to `datasets.load_dataset(*args, **kwargs)`. Full HF flexibility — supports `name` (subset), `split`, `revision`, `data_dir`, `data_files`.

### Prompt construction

Two modes controlled by `prompt_format`:

**`template`** (default): Constructs chat messages from scratch.
- `prompt_template` is a Python format string referencing dataset columns directly: `{question}`, `{subject}`, etc.
- Result wrapped as `[{"role": "user", "content": formatted_text}]`.
- `system_prompt` (optional) prepended as `{"role": "system", ...}`.

**`chat_messages`**: Dataset already has a column with chat message lists.
- `chat_messages_field` specifies which column (default: `"messages"`).
- Passed through as-is.
- `system_prompt` prepended only if first message isn't already a system message.

### Multi-task

Multiple tasks in one list. Each produces a separate parquet. All paths fed as a list to `config.data.train_files`:

```yaml
train_tasks:
  - loading_params: { args: ["openai/gsm8k"], kwargs: { name: main, split: train } }
    prompt_template: "{question}"
    data_source: "gsm8k"

  - loading_params: { args: ["hendrycks/competition_math"], kwargs: { split: train } }
    prompt_template: "{problem}"
    system_prompt: "Solve step by step."
    data_source: "math"
```

### Chat messages example

```yaml
train_tasks:
  - loading_params: { args: ["my-org/my-chat-dataset"], kwargs: { split: train } }
    prompt_format: chat_messages
    chat_messages_field: "conversations"
    data_source: "my_chat"
```

## Subclassing

For tasks needing custom preprocessing (answer parsing, filtering, custom output columns), create a Task subclass. Subclasses own their config shape and processing logic. They typically override `_make_map_fn()` to add task-specific columns (e.g., `reward_model` with parsed ground truth) on top of the base prompt construction, or override `build_dataset()` entirely for full control.

Subclasses are specified in config via verl's `custom_cls` pattern. `custom_cls.name` defaults to `"Task"` when omitted:

```yaml
train_tasks:
  - custom_cls:
      path: "treetune_recipe/math/math_task.py"
      name: MathTask
    loading_params:
      args: ["openai/gsm8k"]
      kwargs: { name: main, split: train }
    prompt_template: "{question}"
    answer_key: "answer"
    data_source: "gsm8k"
```

## Caching

### Cache key

Two components combined: `{impl_hash}_{config_hash}`.

- **`impl_hash`**: SHA256 of the Task class source file bytes. Any change to the `.py` file invalidates the cache.
- **`config_hash`**: SHA256 of the frozen config dict (serialized via `pickle.dumps` after `OmegaConf.to_container(resolve=True)`).

### Cache location

Priority:
1. `cache_dir` constructor argument (if passed)
2. `TREETUNE_TASK_CACHE_DIR` environment variable
3. Default: `~/.cache/treetune/tasks/`

### Invalidation triggers

- Task class source file changes → impl_hash changes
- Any config value changes → config_hash changes
- Manual deletion of cached `.parquet` files

## Integration

Three utility functions, from low-level to high-level:

- **`get_dataset_paths(task_configs, cache_dir=None)`**: Iterates task configs, resolves each to a Task class (via `custom_cls` or falling back to base `Task`), instantiates it, calls `get_parquet_path()`. Returns list of parquet paths.

- **`resolve_tasks_into_config(config)`**: If `config.train_tasks` exists, calls `get_dataset_paths` and patches the result into `config.data.train_files`. Same for `config.val_tasks` → `config.data.val_files`. Returns the modified config. Only touches these two fields — nothing else.

- **`run_with_tasks(config)`**: Calls `resolve_tasks_into_config`, then delegates to verl's `run_ppo(config)`.

### Recipe entrypoint usage

High-level (most recipes):
```python
from treetune_verl.tasks import run_with_tasks

@hydra.main(config_path="config", config_name="my_recipe", version_base=None)
def main(config):
    run_with_tasks(config)
```

Lower-level (custom setup between resolution and training):
```python
from treetune_verl.tasks import resolve_tasks_into_config
from verl.trainer.main_ppo import run_ppo

@hydra.main(...)
def main(config):
    resolve_tasks_into_config(config)
    # ... custom setup ...
    run_ppo(config)
```

## What this does NOT touch

- `verl/` upstream — zero modifications
- `RLHFDataset` — consumed as-is via parquet paths
- `RayPPOTrainer` — unaware of tasks
- Reward functions — task subclasses embed reward-related columns in parquet; dispatch unchanged

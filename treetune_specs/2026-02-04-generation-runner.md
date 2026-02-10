# Generation Runner

## Overview

A reusable generation-only infrastructure for collecting trajectories from LLMs without training.

**Goals:**
- Generate trajectories using async agent loop architecture
- Stream results to queue as each sample completes
- Save results incrementally with checkpointing (fault tolerance)
- Storage-friendly batching (few large files, not many small ones)
- Reusable backbone for future generation-only recipes

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           GenerationRunner                                │
│  (runner.py)                                                              │
│  - Orchestrates generation                                                │
│  - Manages checkpointing                                                  │
│  - Pulls from queue, saves batches                                       │
│  - Merges into final DataProto                                           │
│  - Uploads trajectories.zip artifact                                     │
└──────────────────────────────────────────────────────────────────────────┘
         │                                           ▲
         │ dispatch_streaming()                      │ queue.get(timeout=...)
         ▼                                           │
┌─────────────────────────┐               ┌─────────────────────────────────┐
│  GenerationLoopManager  │               │     ray.util.queue.Queue        │
│  (runner.py)            │               │                                 │
│                         │               │  - Built-in Ray queue           │
│  - Standalone mode      │               │  - Blocking get with timeout    │
│  - Injects queue via    │               │  - Workers put, runner gets     │
│    set_queue()          │               └─────────────────────────────────┘
└─────────────────────────┘                              ▲
         │                                               │
         │ dispatches chunks to                          │ queue.put((idx, DataProto))
         ▼                                               │
┌─────────────────────────┐                              │
│  StreamingAgentLoop-    │──────────────────────────────┘
│  Workers                │
│  (worker.py)            │
│                         │
│  - generate_sequences_  │
│    streaming()          │
│  - asyncio.as_completed │
│  - _postprocess([item]) │
│  - Push each result     │
│    immediately          │
└─────────────────────────┘
         │
         │ generate requests
         ▼
┌─────────────────────────┐
│   SGLang Replicas       │
│   (standalone mode)     │
└─────────────────────────┘
```

## Key Design: Streaming via `asyncio.as_completed`

The worker overrides generation to stream results instead of batching them.

Upstream `AgentLoopWorker.generate_sequences` uses `asyncio.gather()` which waits for all tasks before returning. The mixin provides `generate_sequences_streaming()` which uses `asyncio.as_completed()` to push each result to a `ray.util.queue.Queue` as soon as it completes.

`_postprocess()` takes `list[_InternalAgentLoopOutput]` — calling it with a single-item list works because each trajectory is independent with no inter-sample dependencies.

`index` in `non_tensor_batch` is required (already used for tracing upstream). The mixin fails fast if missing.

## Components

### 1. Results Queue

Uses Ray's built-in `ray.util.queue.Queue` directly.

```python
from ray.util.queue import Queue

# In runner
self._queue = Queue()

# In worker (push)
self._queue.put((idx, single_output))

# In runner (pull)
try:
    idx, result = self._queue.get(block=True, timeout=pull_timeout)
except Empty:
    pass  # timeout, continue loop
```

Workers push `(idx, DataProto)` tuples. Runner pulls one at a time, buffers, and saves in batches.

### 2. StreamingAgentLoopWorkerMixin

Mixin that adds `generate_sequences_streaming()` to any `AgentLoopWorker`.

```
Location: treetune_verl/generation/worker.py

- No __init__ — uses set_queue() setter
- generate_sequences_streaming(): uses asyncio.as_completed(), pushes each result to queue
- Requires 'index' in non_tensor_batch (fails if missing)
```

```python
from ray.util.queue import Queue

class StreamingAgentLoopWorkerMixin:
    """Mixin that streams results to queue as each completes.

    No __init__ — use set_queue() to inject the queue.
    """

    _queue: Queue | None = None

    def set_queue(self, queue: Queue) -> None:
        """Set the Ray Queue for streaming results."""
        self._queue = queue

    async def generate_sequences_streaming(self, batch: DataProto) -> None:
        """Streams results to queue as each completes. Returns nothing."""
        if self._queue is None:
            raise RuntimeError("Queue not set. Call set_queue() first.")

        if "index" not in batch.non_tensor_batch:
            raise ValueError("'index' required in non_tensor_batch for streaming generation")

        index = batch.non_tensor_batch["index"]

        # ... setup sampling_params, traced_indices same as parent ...

        async def run_with_idx(i: int, idx: int, **kwargs):
            result = await self._run_agent_loop(sampling_params, trajectory_info[i], **kwargs)
            return (idx, result)

        tasks = [
            run_with_idx(i, index[i], **{k: v[i] for k, v in batch.non_tensor_batch.items()})
            for i in range(len(batch))
        ]

        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            single_output = self._postprocess([result])
            self._queue.put((idx, single_output))


# Concrete class for Ray (worker.py)
class StreamingAgentLoopWorker(StreamingAgentLoopWorkerMixin, AgentLoopWorker):
    pass
```

### 3. GenerationLoopManager

Lightweight `AgentLoopManager` subclass for standalone streaming generation.

```
Location: treetune_verl/generation/runner.py

- Sets agent_loop_workers_class to StreamingAgentLoopWorker
- Injects queue into workers via set_queue() after creation
- dispatch_streaming() returns refs without blocking
```

```python
from ray.util.queue import Queue

class GenerationLoopManager(AgentLoopManager):
    """AgentLoopManager configured for streaming generation."""

    def __init__(self, config: DictConfig, queue: Queue):
        self._queue = queue
        self.agent_loop_workers_class = ray.remote(StreamingAgentLoopWorker)

        super().__init__(config, worker_group=None, rollout_resource_pool=None)

        # Inject queue into workers after creation
        ray.get([
            worker.set_queue.remote(self._queue)
            for worker in self.agent_loop_workers
        ])

    def dispatch_streaming(self, prompts: DataProto) -> list[ray.ObjectRef]:
        """Non-blocking dispatch. Returns refs for completion check."""
        self.wake_up()

        chunks = prompts.chunk(len(self.agent_loop_workers))
        return [
            worker.generate_sequences_streaming.remote(chunk)
            for worker, chunk in zip(self.agent_loop_workers, chunks)
        ]
```

### 4. GenerationRunner

Orchestrator. Creates queue and manager, dispatches, pulls, saves, merges.

```
Location: treetune_verl/generation/runner.py
```

```python
from queue import Empty
from ray.util.queue import Queue

class GenerationRunner:
    def __init__(self, config: DictConfig, dataset: Dataset, collate_fn):
        self.config = config
        self.output_dir = Path(config.trainer.default_local_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.total_samples = len(dataset)
        ...

    def _prepare_prompts(self, indices: list[int]) -> DataProto:
        """Load and prepare prompts for given indices via DataLoader."""
        subset = Subset(self.dataset, indices)
        dataloader = DataLoader(subset, batch_size=len(subset), collate_fn=self.collate_fn)
        batch_dict = next(iter(dataloader))
        return DataProto.from_single_dict(batch_dict)

    def run(self) -> None:
        self._queue = Queue()
        manager = GenerationLoopManager(self.config, self._queue)

        prompts = self._prepare_prompts(pending_indices)
        worker_refs = manager.dispatch_streaming(prompts)

        # Pull loop with optional tqdm progress bar (controlled by generation.show_progress)
        batch_buffer = []
        pbar = tqdm(total=self.total_samples, initial=len(self.completed_indices)) \
            if gen_config.show_progress else None
        try:
            while len(self.completed_indices) < self.total_samples:
                try:
                    idx, result = self._queue.get(block=True, timeout=pull_timeout)
                    batch_buffer.append((idx, result))
                    if pbar:
                        pbar.update(1)

                    if len(batch_buffer) >= save_batch_size:
                        self._save_batch(batch_buffer, batch_idx)
                        batch_buffer = []
                        batch_idx += 1
                except Empty:
                    if batch_buffer:
                        self._save_batch(batch_buffer, batch_idx)
                        batch_buffer = []
                        batch_idx += 1
        finally:
            if pbar:
                pbar.close()

        if batch_buffer:
            self._save_batch(batch_buffer, batch_idx)

        ray.get(worker_refs)
        manager.sleep()

        if gen_config.final_merge:
            self._merge_batches()
        if gen_config.upload_artifact:
            self._upload_artifact()
```

## Config Structure

Trainer-like nested structure. Matches what `AgentLoopManager` expects directly.

### Base Config: `treetune_verl/generation/config/generation.yaml`

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - model@actor_rollout_ref.model: hf_model
  - rollout@actor_rollout_ref.rollout: rollout
  - data@data: legacy_data
  - _self_

# === Trainer (hardware + output + logging) ===
trainer:
  n_gpus_per_node: 8
  nnodes: 1
  project_name: generation
  experiment_name: ${now:%Y%m%d_%H%M%S}
  default_local_dir: outputs/${trainer.project_name}/${trainer.experiment_name}
  logger: ["console", "wandb"]

# === Model (inherits from hf_model defaults) ===
actor_rollout_ref:
  model:
    path: ???  # Only override what differs from hf_model defaults
  rollout:
    name: sglang
    tensor_model_parallel_size: 1
    data_parallel_size: 1
    pipeline_model_parallel_size: 1
    gpu_memory_utilization: 0.85
    temperature: 1.0
    top_p: 1.0
    top_k: -1
    prompt_length: 1024
    response_length: 4096
    agent:
      num_workers: 8
      default_agent_loop: single_turn_agent

# === Reward Model (disabled) ===
reward_model:
  enable: false
  use_reward_loop: false
  enable_resource_pool: false

# === Data ===
# Inherits from legacy_data defaults (same as PPO pipeline).
# Uses RLHFDataset via create_rl_dataset() for consistency with training.
# Key fields: train_files (parquet paths), prompt_key, max_prompt_length,
#   return_raw_chat, filter_overlong_prompts, tool_config_path, custom_cls, etc.
# See verl/trainer/config/data/legacy_data.yaml for all available options.
data:
  train_files: ???
  max_prompt_length: 1024
  train_max_samples: -1  # -1 = use full dataset

# === Task System (optional) ===
tasks: null

# === Generation Runner ===
generation:
  save_batch_size: 1000
  pull_timeout: 30.0
  final_merge: true
  show_progress: true
  upload_artifact: true

# === Ray (optional) ===
ray_kwargs:
  ray_init: {}  # Merged with defaults from get_ppo_ray_runtime_env()
  timeline_json_file: null  # Optional Ray timeline trace for profiling
```

## Data Flow

```
== run_generation() in main.py (setup, same as main_ppo.py) ==
1. Init Ray, resolve config
2. Resolve tasks: train_tasks → config.data.train_files (if configured)
3. Create tokenizer + processor from model path
4. Create dataset via RLHFDataset (create_rl_dataset)
   - Handles: parquet loading, chat template, tool schemas, multimodal, prompt filtering
5. Pass dataset + collate_fn to GenerationRunner(config, dataset, collate_fn)

== GenerationRunner.run() (orchestration) ==
6. Load checkpoint if exists, compute pending indices
7. Create ray.util.queue.Queue
8. Create GenerationLoopManager (injects queue into workers)
9. dispatch_streaming(prompts) → returns worker refs (non-blocking)

10. Concurrent execution:

    [Workers]
    For each sample in chunk:
      - Run agent loop
      - _postprocess([result]) → queue.put((idx, DataProto))

    [Runner pull loop]
    While completed < total:
      - idx, result = queue.get(timeout=...)
      - Buffer, save batch when full, checkpoint

11. ray.get(worker_refs) — ensure all done
12. manager.sleep() + destroy()
13. Merge batches → trajectories.pkl (DataProto)
14. Upload trajectories.zip artifact
```

## Storage Layout

Uses `trainer.default_local_dir`:

```
${trainer.default_local_dir}/
├── checkpoint.json         # Progress checkpoint
├── batch_0000.pkl          # list[tuple[int, DataProto]]
├── batch_0001.pkl
├── ...
├── trajectories.pkl        # Final merged DataProto (if final_merge)
└── trajectories.zip        # Upload artifact (if upload_artifact)
```

## Batch Saving

Each batch is a pickle file containing `list[tuple[int, DataProto]]`:

```python
def _save_batch(self, items: list[tuple[int, DataProto]], batch_idx: int) -> None:
    batch_name = f"batch_{batch_idx:04d}"
    batch_path = self.output_dir / f"{batch_name}.pkl"

    with open(batch_path, "wb") as f:
        pickle.dump(items, f)

    for idx, _ in items:
        self.completed_indices.add(idx)
    self.saved_batches.append(batch_name)
    self._save_checkpoint()
```

## Merging → Final DataProto

```python
def _merge_batches(self) -> DataProto:
    all_items: list[tuple[int, DataProto]] = []
    for batch_name in sorted(self.saved_batches):
        with open(self.output_dir / f"{batch_name}.pkl", "rb") as f:
            all_items.extend(pickle.load(f))

    all_items.sort(key=lambda x: x[0])
    merged = DataProto.concat([item[1] for item in all_items])

    merged.save_to_disk(self.output_dir / "trajectories.pkl")

    return merged
```

## Final Artifact

`trajectories.pkl` contains a single `DataProto`:

```python
DataProto(
    batch=TensorDict({
        "prompts": Tensor[N, prompt_length],
        "responses": Tensor[N, response_length],
        "response_mask": Tensor[N, response_length],
        "input_ids": Tensor[N, total_length],
        "attention_mask": Tensor[N, total_length],
        "position_ids": Tensor[N, total_length],
        "rollout_log_probs": Tensor[N, response_length],  # if calculated
    }),
    non_tensor_batch={
        "__num_turns__": ndarray[N],
        "raw_prompt": ndarray[N, object],
        ...
    },
    meta_info={...}
)
```

## Artifact Upload

Uses verl's `Tracking` class (`trainer.logger` config) for logging. Uploads a zip artifact when `generation.upload_artifact=true` and wandb is in the logger.

```python
from verl.utils.tracking import Tracking

class GenerationRunner:
    def __init__(self, config):
        ...
        self.tracker = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config),
        )

    def _upload_artifact(self) -> None:
        import zipfile

        zip_path = self.output_dir / "trajectories.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            trajectories_path = self.output_dir / "trajectories.pkl"
            if trajectories_path.exists():
                zf.write(trajectories_path, "trajectories.pkl")
            checkpoint_path = self.output_dir / "checkpoint.json"
            if checkpoint_path.exists():
                zf.write(checkpoint_path, "checkpoint.json")

        if "wandb" in self.tracker.logger:
            wandb = self.tracker.logger["wandb"]
            artifact = wandb.Artifact(name="trajectories", type="trajectories")
            artifact.add_file(str(zip_path), name="trajectories.zip")
            wandb.log_artifact(artifact)
```

**Zip contents:**
```
trajectories.zip
├── trajectories.pkl    # DataProto with all N samples
└── checkpoint.json
```

## Checkpointing & Resumption

### Checkpoint Structure

```python
{
    "completed_indices": [0, 1, 2, ...],
    "saved_batches": ["batch_0000", ...],
    "total_samples": 10000,
}
```

### Saving

After each batch save. Atomic write (tmp → rename):

```python
def _save_checkpoint(self) -> None:
    tmp_path = self.output_dir / "checkpoint.json.tmp"
    final_path = self.output_dir / "checkpoint.json"

    with open(tmp_path, "w") as f:
        json.dump({
            "completed_indices": sorted(self.completed_indices),
            "saved_batches": self.saved_batches,
            "total_samples": self.total_samples,
        }, f)

    tmp_path.rename(final_path)
```

### Resume

```
1. Runner.__init__():
   - Load checkpoint.json if exists
   - Set self.completed_indices, self.saved_batches

2. Runner.run():
   - pending = [i for i in range(total) if i not in completed_indices]
   - If no pending: skip to final merge
   - Else: dispatch only pending indices
   - Batch numbering resumes from len(saved_batches)
```

```python
def _load_checkpoint(self) -> bool:
    checkpoint_path = self.output_dir / "checkpoint.json"
    if not checkpoint_path.exists():
        return False

    with open(checkpoint_path) as f:
        data = json.load(f)

    self.completed_indices = set(data["completed_indices"])
    self.saved_batches = data["saved_batches"]
    return True
```

### CLI

```bash
# Run generation
python -m treetune_verl.generation.main \
  actor_rollout_ref.model.path=/models/llama-8b \
  data.train_files=[/data/prompts.parquet]

# Resume (same command — auto-detects checkpoint in default_local_dir)
python -m treetune_verl.generation.main \
  actor_rollout_ref.model.path=/models/llama-8b \
  data.train_files=[/data/prompts.parquet]

# Force fresh start
rm ${default_local_dir}/checkpoint.json
```

## Entrypoint

**Location**: `treetune_verl/generation/main.py`

**Responsibilities (mirrors `main_ppo.py` pattern):**
- Hydra decorator: `@hydra.main(config_path="config", config_name="generation")`
- Auto-detect device (NPU/CUDA) via `verl.utils.device.auto_set_device()`
- Initialize Ray with runtime_env (SGLang env vars from `verl.trainer.constants_ppo.get_ppo_ray_runtime_env()`)
- Merge `config.ray_kwargs.ray_init` if present
- Print hostname, PID, and resolved config for debugging
- Resolve config via `OmegaConf.resolve()`
- **Resolve tasks:** `resolve_tasks_into_config(config)` — patches `train_tasks` → `config.data.train_files` (if configured)
- **Load data (same as main_ppo.py):**
  - Instantiate tokenizer via `hf_tokenizer(local_path, ...)`
  - Instantiate processor via `hf_processor(local_path, ...)`
  - Create dataset via `create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, ...)`
- Pass `dataset` + `collate_fn` to `GenerationRunner(config, dataset, collate_fn)` and call `run()`
- Optional: Save Ray timeline trace if `config.ray_kwargs.timeline_json_file` set
- Reference: `verl/trainer/main_ppo.py`

```python
def run_generation(config) -> None:
    """Initialize Ray, load data, and run generation.

    Data loading follows the same pattern as main_ppo.py:
    tokenizer/processor/dataset created here, passed to runner.
    """
    # Ray init ...

    # Task system: resolve train_tasks → config.data.train_files (if configured)
    # Same pattern as run_with_tasks() wrapping run_ppo().
    # See treetune_specs/2026-01-31-task-system-design.md
    from treetune_verl.tasks import resolve_tasks_into_config
    resolve_tasks_into_config(config)

    # Data loading (same as main_ppo.py lines 310-345)
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_dataset

    local_path = copy_to_local(config.actor_rollout_ref.model.path, ...)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
    processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False))

    dataset = create_rl_dataset(
        config.data.train_files, config.data,
        tokenizer, processor,
        max_samples=config.data.get("train_max_samples", -1),
    )

    # Pass dataset + collate_fn to runner (same as RayPPOTrainer pattern)
    runner = GenerationRunner(config, dataset, collate_fn)
    runner.run()
```

**Pattern:** Runs on driver (no TaskRunner needed). Orchestration is lightweight (pull loop + checkpointing); heavy work is in AgentLoopWorker actors.

**Separate `run_generation(config)` function:** Allows programmatic use without Hydra.

## Directory Layout

```
treetune_verl/generation/
├── __init__.py
├── main.py                # Hydra entrypoint
├── runner.py              # GenerationRunner, GenerationLoopManager
├── worker.py              # StreamingAgentLoopWorkerMixin, StreamingAgentLoopWorker
└── config/
    └── generation.yaml
```

## Implementation Order

1. **Phase 1: Core**
   - [ ] `worker.py` — StreamingAgentLoopWorkerMixin, StreamingAgentLoopWorker
   - [ ] `runner.py` — GenerationLoopManager, GenerationRunner (data via RLHFDataset)
   - [ ] `config/generation.yaml` (with legacy_data + hf_model defaults)
   - [ ] `main.py` — Hydra entrypoint with Ray init and config resolution

2. **Phase 2: Testing**
   - [ ] Unit tests for streaming worker
   - [ ] Integration test (small model, few samples)
   - [ ] Checkpoint/resume validation

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Worker crash during generation | Checkpointing; lose at most in-flight samples |
| Queue memory overflow | Save batches frequently; queue only holds unsaved |
| Missing index in batch | Fail fast with clear error message |
| SGLang server OOM | Configure gpu_memory_utilization appropriately |

## References

- Development guide: `agent-docs/development-guide.md`
- Agent loop docs: `agent-docs/agent-loop.md`
- Task system: `treetune_specs/2026-01-31-task-system-design.md`

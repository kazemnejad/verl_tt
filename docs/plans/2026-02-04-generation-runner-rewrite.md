# GenerationRunner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-ng:executing-plans to implement this plan task-by-task.

**Goal:** Implement GenerationRunner per spec - queue-based incremental collection with concurrent pull loop, fault-tolerant checkpointing, and proper per-sample extraction.

**Architecture:**
- `ResultsQueue` (Ray actor): Signal-driven accumulation of completed trajectories
- `CollectorAgentLoopWorker`: Subclass that pushes to queue on trajectory completion
- `GenerationRunner`: Orchestrator with concurrent generation + pull loops, inline checkpoint logic
- Config via Hydra YAML with all spec options (final_merge, wandb_upload, etc.)

**Tech Stack:** Ray, asyncio, PyArrow, Hydra, pickle, pytest

---

## Pre-Implementation: Clean Slate

**Step 0: Delete existing broken implementation**

Delete these files:
- `treetune_verl/generation/runner.py`
- `treetune_verl/generation/queue.py`
- `treetune_verl/generation/worker.py`
- `treetune_verl/generation/checkpoint.py`
- `treetune_verl/generation/config.py`
- `treetune_tests/treetune_verl/generation/*` (all test files, keep `__init__.py`)

Keep:
- `treetune_verl/generation/__init__.py` (will update exports)
- `treetune_verl/generation/config/generation.yaml` (will update)

---

## Task 1: ResultsQueue Unit Tests + Implementation

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_queue.py`
- Impl: `treetune_verl/generation/queue.py`

**What to test (write tests FIRST):**

1. `test_put_and_count` - Put items, count returns correct number
2. `test_get_batch_returns_when_min_reached` - Blocks until min_items available, then returns all
3. `test_get_batch_returns_on_timeout` - Returns partial results when timeout expires
4. `test_get_batch_empty_timeout` - Returns empty list if timeout with no items
5. `test_fifo_order_preserved` - Items returned in insertion order
6. `test_concurrent_puts` - Multiple concurrent puts don't lose data (use threading)
7. `test_drain_clears_queue` - After get_batch, queue is empty

**Test setup:** Use `ray.init(local_mode=True)` fixture for fast unit tests.

**Implementation requirements (from spec):**
- Ray actor (`@ray.remote`)
- `put(idx: int, result: Any) -> None` - Non-blocking, thread-safe
- `get_batch(min_items: int, timeout: float) -> list[tuple[int, Any]]` - Blocks until threshold OR timeout
- `count() -> int` - Current queue size
- Use `threading.Condition` for signal-driven waiting (not polling)

**Commit after:** Tests pass for ResultsQueue

---

## Task 2: CollectorAgentLoopWorker Unit Tests + Implementation

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_worker.py`
- Impl: `treetune_verl/generation/worker.py`

**What to test (write tests FIRST):**

1. `test_worker_inherits_from_agent_loop_worker` - Verify inheritance
2. `test_worker_stores_queue_handle` - Constructor stores results_queue
3. `test_push_to_queue_calls_remote_put` - `_push_to_queue(idx, result)` calls `queue.put.remote(idx, result)`
4. `test_worker_calls_push_on_trajectory_completion` - Override point is hooked correctly

**Test setup:** Mock `AgentLoopWorker` parent class, mock Ray actor handle.

**Implementation requirements (from spec):**
- Subclass `verl.experimental.agent_loop.agent_loop.AgentLoopWorker`
- Constructor takes `results_queue: ActorHandle` in addition to parent args
- `_push_to_queue(idx: int, result: Any)` method that calls `results_queue.put.remote()`
- Override the trajectory completion hook to call `_push_to_queue`

**Key challenge:** Find the right override point in `AgentLoopWorker` where individual trajectories complete. Study `verl/experimental/agent_loop/agent_loop.py` to identify the hook.

**Commit after:** Tests pass for CollectorAgentLoopWorker

---

## Task 3: GenerationRunner Unit Tests (Config Adapter)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_runner_config.py`
- Impl: `treetune_verl/generation/runner.py` (partial - config adapter only)

**What to test (write tests FIRST):**

1. `test_adapt_config_adds_target_fields` - `_target_` added to rollout, model, mtp configs
2. `test_adapt_config_preserves_rollout_params` - temperature, top_p, etc. preserved
3. `test_adapt_config_creates_trainer_structure` - trainer.n_gpus_per_node, trainer.nnodes present
4. `test_adapt_config_disables_reward_model` - reward_model.enable = False
5. `test_adapt_config_handles_missing_mtp` - Creates default mtp config if missing

**Implementation requirements:**
- `_adapt_config_for_manager(config: DictConfig) -> DictConfig` static method
- Converts flat generation config to nested structure AgentLoopManager expects
- Adds `_target_` fields for Hydra instantiation (critical for Ray serialization)

**Commit after:** Config adapter tests pass

---

## Task 4: GenerationRunner Unit Tests (Checkpoint Logic)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_runner_checkpoint.py`
- Impl: `treetune_verl/generation/runner.py` (partial - checkpoint methods)

**What to test (write tests FIRST):**

1. `test_save_checkpoint_writes_json` - Writes checkpoint.json with correct structure
2. `test_load_checkpoint_restores_state` - Loads completed_indices, saved_batches
3. `test_load_checkpoint_missing_file` - Returns gracefully if no checkpoint
4. `test_get_pending_indices_excludes_completed` - Filters out completed indices
5. `test_checkpoint_contains_config_snapshot` - Config snapshot saved for validation
6. `test_validate_batches_detects_missing` - Identifies missing batch files
7. `test_validate_batches_detects_corrupt` - Identifies corrupt pickle files

**Implementation requirements (inline in runner, no separate class):**
- `_save_checkpoint()` - Write JSON to `output_dir/checkpoint.json`
- `_load_checkpoint()` - Load state if file exists
- `_get_pending_indices()` - Return indices not in completed_indices
- `_validate_batches()` - Check batch files exist and are readable

**Checkpoint JSON structure (from spec):**
```
{
  "completed_indices": [0, 1, 2, ...],
  "saved_batches": ["batch_0000", "batch_0001"],
  "total_samples": 10000,
  "config_snapshot": {...}
}
```

**Commit after:** Checkpoint tests pass

---

## Task 5: GenerationRunner Unit Tests (Batch Operations)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_runner_batch.py`
- Impl: `treetune_verl/generation/runner.py` (partial - batch methods)

**What to test (write tests FIRST):**

1. `test_extract_per_sample_slices_tensors` - Extracts 1D tensors from batched 2D
2. `test_extract_per_sample_handles_all_fields` - input_ids, responses, response_mask, etc.
3. `test_extract_per_sample_preserves_numpy` - Handles numpy arrays in non_tensor_batch
4. `test_save_batch_writes_pickle` - Creates batch_NNNN.pkl file
5. `test_save_batch_updates_checkpoint` - Adds indices and batch name to checkpoint
6. `test_merge_batches_combines_all` - Merges batch files into trajectories.pkl
7. `test_merge_batches_sorts_by_index` - Final output sorted by sample index

**Implementation requirements:**
- `_extract_per_sample(output: DataProto, indices: list[int]) -> list[tuple[int, dict]]`
- `_save_batch(items: list[tuple[int, dict]], batch_idx: int) -> str`
- `_merge_batches()` - Combine all batch files into trajectories.pkl

**Per-sample tensor structure (from spec):**
```
{
  "input_ids": Tensor[seq_len],        # 1D, not 2D batched
  "attention_mask": Tensor[seq_len],
  "position_ids": Tensor[seq_len],
  "responses": Tensor[response_len],
  "response_mask": Tensor[response_len],
  "rollout_log_probs": Tensor[response_len],  # if calculate_log_probs=True
  "prompts": Tensor[prompt_len],
}
```

**Commit after:** Batch operation tests pass

---

## Task 6: GenerationRunner Unit Tests (Data Loading)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_runner_data.py`
- Impl: `treetune_verl/generation/runner.py` (partial - data loading)

**What to test (write tests FIRST):**

1. `test_load_single_parquet` - Loads single parquet file
2. `test_load_multiple_parquet` - Concatenates multiple files
3. `test_load_applies_max_samples` - Limits to max_samples if set
4. `test_load_extracts_prompt_key` - Uses configured prompt_key column
5. `test_task_system_resolves_to_files` - If tasks configured, resolves to data.files

**Test setup:** Create minimal parquet files with PyArrow in tmp_path fixture.

**Implementation requirements:**
- `_load_data()` - Load parquet, apply max_samples, store dataframe
- Task system integration: if `config.tasks` set, call `get_dataset_paths()`

**Commit after:** Data loading tests pass

---

## Task 7: GenerationRunner Unit Tests (WandB Upload)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_runner_wandb.py`
- Impl: `treetune_verl/generation/runner.py` (partial - wandb methods)

**What to test (write tests FIRST):**

1. `test_upload_creates_artifact` - Creates wandb.Artifact with correct type
2. `test_upload_adds_merged_file` - Adds trajectories.pkl if final_merge=True
3. `test_upload_adds_batch_files` - Adds batch files if final_merge=False
4. `test_upload_adds_checkpoint` - Includes checkpoint.json
5. `test_upload_uses_existing_run` - Uses active run if available
6. `test_upload_creates_new_run` - Creates run if none active

**Test setup:** Mock wandb module.

**Implementation requirements:**
- `_upload_to_wandb()` - Create and upload artifact per spec

**Commit after:** WandB tests pass

---

## Task 8: Integration Test (Queue + Worker)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_queue_worker_integration.py`

**What to test:**

1. `test_worker_pushes_to_queue` - Worker completion triggers queue.put
2. `test_queue_accumulates_from_multiple_workers` - Multiple workers, single queue
3. `test_queue_signals_when_batch_ready` - get_batch returns when threshold reached

**Test setup:**
- Ray local mode
- Mock LLM server (don't need actual generation)
- Create minimal worker instances with queue handle

**Commit after:** Integration tests pass

---

## Task 9: Integration Test (Runner + Queue + Worker)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_runner_integration.py`

**What to test:**

1. `test_runner_creates_queue_and_manager` - Initialization creates components
2. `test_runner_concurrent_loops` - Pull loop runs concurrent with generation
3. `test_runner_saves_incrementally` - Batches saved as results arrive
4. `test_runner_updates_progress` - Progress bar updates on queue pull
5. `test_runner_handles_timeout` - Final stragglers collected via timeout

**Test setup:**
- Mock AgentLoopManager to simulate async generation
- Real ResultsQueue actor
- Verify concurrent behavior with timing assertions

**Commit after:** Runner integration tests pass

---

## Task 10: E2E Test (Full Pipeline)

**Files:**
- Test: `treetune_tests/treetune_verl/generation/test_e2e.py`
- Config: `treetune_tests/treetune_verl/generation/e2e_config/generation_test.yaml`

**What to test (requires GPU):**

1. `test_full_generation_pipeline`:
   - Create test parquet (5 samples)
   - Run GenerationRunner with real model (Qwen2.5-0.5B-Instruct)
   - Verify batch files created
   - Verify each sample has correct tensor structure (1D, correct fields)
   - Verify response_mask semantics (0s and 1s only, some 1s present)
   - Verify checkpoint has all indices
   - Verify no duplicate indices

2. `test_resume_from_checkpoint`:
   - Create partial checkpoint (2/5 completed)
   - Run GenerationRunner
   - Verify only remaining 3 samples generated
   - Verify final output has all 5 samples

3. `test_final_merge`:
   - Run with save_batch_size=2, 5 samples
   - Verify 3 batch files created
   - Verify trajectories.pkl contains all 5 samples sorted

4. `test_output_usable_for_training`:
   - Verify all training-required fields present
   - Verify tensor dtypes are correct
   - Verify shapes are consistent

**Test config requirements:**
- Small model: Qwen/Qwen2.5-0.5B-Instruct
- Few samples: max_samples=5
- Small batches: save_batch_size=2
- Deterministic: temperature=0.0

**Commit after:** E2E tests pass

---

## Task 11: Hydra Config + Exports

**Files:**
- Update: `treetune_verl/generation/config/generation.yaml`
- Update: `treetune_verl/generation/__init__.py`

**Config requirements (from spec):**
- All generation options: output_dir, save_batch_size, pull_timeout, final_merge, checkpoint_interval, show_progress
- WandB options: wandb_upload, wandb_project, wandb_run_name
- Import rollout from verl via Hydra defaults
- Task system support: tasks field

**Exports:**
- `GenerationRunner`
- `ResultsQueue`
- `CollectorAgentLoopWorker`

**Commit after:** Config complete, imports work

---

## Task 12: Final Validation

**Steps:**

1. Run full test suite:
   ```
   pytest treetune_tests/treetune_verl/generation/ -v
   ```

2. Run pre-commit:
   ```
   pre-commit run --all-files
   ```

3. Verify no regressions in other tests

4. Final commit with all changes

---

## Spec Compliance Checklist

| Spec Requirement | Task | Status |
|-----------------|------|--------|
| ResultsQueue Ray actor | Task 1 | |
| Signal-driven get_batch (not polling) | Task 1 | |
| CollectorAgentLoopWorker pushes on completion | Task 2 | |
| Config adapter with _target_ fields | Task 3 | |
| Checkpoint save/load (inline, no separate class) | Task 4 | |
| Per-sample tensor extraction (1D, not batched) | Task 5 | |
| Batch file saving | Task 5 | |
| Final merge option | Task 5 | |
| Data loading from parquet | Task 6 | |
| Task system integration | Task 6 | |
| WandB artifact upload | Task 7 | |
| Concurrent generation + pull loops | Task 9 | |
| Progress bar with queue stats | Task 9 | |
| Resume from checkpoint | Task 10 | |
| All config options in YAML | Task 11 | |

---

## Key Implementation Notes

**Finding the worker override point:**
Study `AgentLoopManager._run_agent_loop()` and `AgentLoopWorker` to find where individual trajectory completion happens. The spec says "override trajectory completion to push to ResultsQueue". Look for where `AgentLoopOutput` is created/returned.

**Concurrent loops pattern:**
Use `asyncio.gather()` or `asyncio.create_task()` to run generation dispatch and pull loop concurrently. The pull loop should:
1. Call `queue.get_batch(min=save_batch_size, timeout=pull_timeout)`
2. Extract per-sample tensors
3. Save batch
4. Update checkpoint
5. Update progress bar
6. Repeat until all samples completed

**Progress bar format (from spec):**
```
Generation: 45%|████████████████                    | 4523/10000 [12:34<15:21, 5.94 samples/s]
Saved: 4 batches | Pending: 523 | In-flight: 477
```

**DataProto output structure:**
The batched output from `AgentLoopManager.generate_sequences()` has shape `[batch_size, seq_len]`. Must slice `tensor[i]` to get per-sample 1D tensor.

# ActorRolloutRefWorker - Design & Implementation

## Overview

ActorRolloutRefWorker is a worker (SPMD block) that fuses the Actor (training engine), Rollout (inference engine), and Reference policy (frozen model for forward pass) into a single hybrid worker. A single instance holds and manages all three engines.

This fusion suits on-policy training where only one role is active at a time: Rollout for trajectory generation, Actor for policy gradient updates, or Reference for log-prob computation. Sharing the same GPU across all three roles enables efficient resource usage and zero-copy weight transfer between training and inference engines.

## Implementation Details and Class Hierarchy
There are few implemntations of ActorRolloutRefWorker in verl:

**The legacy implementation:**
- `verl/workers/fsdp_workers.py`: This implements the Actor role using FSDP as training engine and reference policy, support both SGLang or vLLM as for rollout.
- `verl/workers/megatron_workers.py`: This implements the Actor role using Megatron-LM as training engine and reference policy, support both SGLang or vLLM as for rollout.

These implementations are well-tested and used in many projects, but now they're deprecated. We try to avoid them as much as possible and but it's okay to use them if needed.

**The new implementation:**
- `verl/workers/engine_workers.py`: In this implementation, the details of the training engine are abstracted away into an Engine interface (`verl/workers/engine/base.py`), where both FSDP and Megatron implements this interface. 
As a result, the ActorRolloutRefWorker only interacts with the Engine interface and doesn't need to know the concrete implementation of the training engine. The reference also uses the same Engine interface (without optimizer), and the same rollouts are supported.

This is our default choice.

### ActorRolloutRefWorker Interface Summary

**Location:** `verl/workers/engine_workers.py:358`

#### Primary Instance Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `actor` | `TrainingWorker \| None` | Actor (training) worker instance |
| `ref` | `TrainingWorker \| None` | Reference policy worker instance |
| `rollout` | `BaseRollout \| None` | Rollout (inference) engine instance |

#### TrainingWorker

`TrainingWorker` (`engine_workers.py:52`) wraps a `BaseEngine` to provide train/infer APIs. It abstracts away FSDP vs Megatron differences—both implement `BaseEngine` (`verl/workers/engine/base.py`). The `actor` and `ref` attributes above are both `TrainingWorker` instances; `ref` simply runs inference-only (no optimizer updates).

#### Public Methods (Ray-registered)

**Initialization & Control:**
- `init_model()` — Initializes actor, ref, and/or rollout engines based on `role`
- `set_loss_fn(loss_fn)` — Sets loss function on actor
- `to(device, model=True, optimizer=True, grad=True)` — Manual load/offload control

**Compute:**
- `compute_log_prob(data: TensorDict) -> TensorDict` — Compute log probs using actor model
- `compute_ref_log_prob(data: TensorDict) -> TensorDict` — Compute log probs using ref model
- `update_actor(data: TensorDict) -> TensorDict` — PPO training update on actor

**Checkpoint:**
- `save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)`
- `load_checkpoint(local_path, hdfs_path, del_local_after_load)`

**Context Switching (async):**
- `sleep()` — Switch from rollout mode to trainer mode; offloads rollout, restores trainer RNG
- `wake_up()` — Switch from trainer to rollout mode; syncs weights to rollout engine, offloads trainer model

**Rollout/Generation (async):**
- `generate(prompt_ids, sampling_params, request_id, image_data)` — SGLang generation
- `chat_completion(json_request)` — SGLang chat completion
- `get_zeromq_address()` — vLLM ZeroMQ address getter


## How training backends works?
- FSDP: implements data parallelism across the entire workers group and shard the model and optimizer states across the workers group. The sharding is standard torch FSDP. Also, using ulysses, sequence parallelism can be enabled to support long sequences.
- Megatron: It is a full-featured distributed training with DP, TP, EP, CP, and PP. It doesn't support all models, but it's high performance and used for large models.

## Rollout and Inference Engine

> Note: verl has many rollout abstractions; most are deprecated. This section covers the current design using SGLang (default). vLLM follows similar logic.

A **rollout** in verl universe is usually a wrapper that manages the inference engine lifecycle and provides a unified interface.

**Architecture:**
- an inference engine runs on the Ray actors, spanning multiple GPUs/nodes; Supports TP, EP, and PP for large models across GPUs (single actor for small models, many for large).
- Usually multiple replicas of the same engine to increase trajectory generation throughput
- Engines expose server APIs: accept token IDs, return response IDs
- Continuous batching + efficient KV cache management => high performance
- Inference Engines are async & handle many concurrent requests

**Workflow** (see `agent-docs/agent-loop.md`):
1. Agent loop tokenizes prompts and dispatches to inference servers (with load balancing)
2. Servers generate responses asynchronously
3. Agent loop parses tool calls from responses and triggers next turn
4. All generations run concurrently for high throughput

There are three modes of rollout: (`verl/workers/rollout/replica.py:53`)

- **HYBRID** (default): same process & GPU, zero-copy weight sync; On-policy, memory-efficient.
- **COLOCATED**: same placement group, separate processes but share the same GPU; Usually no weight sync; Typically used for reward models or LLM-as-judge.
- **STANDALONE**: separate GPU pools; use NCCL to sync weights; Typically used for off-policy training, or async reward computation.

**Class Organization:**

- **ActorRolloutRefWorker.rollout** → `ServerAdapter` (`verl/workers/rollout/sglang_rollout/sglang_rollout.py:126`)
  - Mostly a placeholder to reserve GPU in Ray scheduling
  - In HYBRID mode: initiates weight sync to inference server

- **RolloutReplica** (`verl/workers/rollout/replica.py`) — single inference server instance (single or multi-node). Unit that AgentLoop talks to.
  - AgentLoop creates a pool of these (`verl/workers/rollout/sglang/async_sglang_server.py:SGLangReplica`)
  - Each replica holds multiple `SGLangHttpServer` instances (`verl/workers/rollout/sglang_rollout/async_sglang_server.py`)
  - For large models across nodes: one `SGLangHttpServer` per node, all grouped into one replica

- **ServerAdapter** — holds pointers to `SGLangHttpServer` instances; calls server APIs for weight sync 

Diagram:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            AgentLoop                                    │
│                     (trajectory generation)                             │
└────────────────────────────┬────────────────────────────────────────────┘
                             │ creates pool of
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    RolloutReplica (pool)                                │
│               verl/workers/rollout/replica.py                           │
├─────────────────────┬─────────────────────┬─────────────────────────────┤
│     Replica 0       │     Replica 1       │        Replica N            │
│  ┌───────────────┐  │  ┌───────────────┐  │     ┌───────────────┐       │
│  │SGLangHttpSrv 0│  │  │SGLangHttpSrv 0│  │     │SGLangHttpSrv 0│       │
│  │SGLangHttpSrv 1│  │  │SGLangHttpSrv 1│  │ ... │SGLangHttpSrv 1│       │
│  │  (per node)   │  │  │  (per node)   │  │     │  (per node)   │       │
│  └───────────────┘  │  └───────────────┘  │     └───────────────┘       │
└─────────────────────┴─────────────────────┴─────────────────────────────┘
                             ▲
                             │ holds pointers (for weight sync)
┌────────────────────────────┴────────────────────────────────────────────┐
│                         ServerAdapter                                   │
│            verl/workers/rollout/sglang_rollout/sglang_rollout.py        │
│        (placeholder for Ray GPU scheduling; HYBRID: weight sync)        │
└─────────────────────────────────────────────────────────────────────────┘
                             ▲
                             │ .rollout attribute
                             │ 
┌─────────────────────────────────────────────────────────────────────────┐
│                     ActorRolloutRefWorker                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Weight Sync between Actor (training engine) and Rollout (inference engine)

In HYBRID mode, actor and rollout share the same GPU. Context switching via `sleep()` and `wake_up()` manages which engine occupies GPU memory. At every RL step, the trainer call the wake_up() to switch to rollout mode. once the generation is done, it calls the sleep() to switch to trainer mode.

### `sleep()` — Rollout → Trainer Mode

Prepares GPU for training:
1. releases rollout memory (KV cache + weights)
2. Clears GPU cache
3. brings parameters from CPU to GPU
4. Swaps RNG state from generation → training (reproducibility)

### `wake_up()` — Trainer → Rollout Mode

Syncs updated weights to inference engine:
1. Extracts weights from training engine (loads parameters to the GPU)
2. Resumes rollout weight buffers (if released)
3. Copies new weights to inference engine (weight sync)
4. Offloads training engine to CPU (optional)
5. Resumes KV cache
6. Swaps RNG state from training → generation
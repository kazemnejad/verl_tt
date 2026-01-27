---
summary: Big picture of how verl works—architecture, core abstractions, how components communicate, supported backends. The most important doc. Read first.
read_when:
  - Starting any verl-related work
  - Understanding overall architecture
  - Learning how components communicate
  - Checking supported backends
---

# verl Framework - Architecture & Implementation Guide

## 1. Overview

**verl** is an open-source RL training library for LLMs. It implements the HybridFlow programming model that combines the single-controller and multi-controller paradigms. 

The Key Insight:
```
Hybrid-Controller = Single-Controller + N × Multi-Controller
```

RL training has two levels of structure:
1. **High-level dataflow**: The sequence of operations (generate → compute rewards → compute values → update policy) - naturally expressed with **single-controller**
2. **Low-level computation**: Each operation (generation, training) is a distributed workload across many GPUs - naturally expressed with **multi-controller (SPMD)**

verl uses a **single-controller at the trainer level** that orchestrates **multiple multi-controller worker groups** via RPC calls.

### Driver-Worker Architecture

```
┌─────────────────┐          ┌─────────────────────┐          ┌─────────────────────────┐
│  Driver Process │          │     WorkerGroup     │          │     Resource Pool       │
└─────────────────┘          └─────────────────────┘          └─────────────────────────┘

    Call API ──────────►  Actor   ┌─────────┬─────────┐      ┌─────────┬─────────┐
    ◄────── Receive Future        │ Worker 0│ Worker 1│ ───► │  GPU 0  │  GPU 1  │  Resource Pool 0
                                  └─────────┴─────────┘      └─────────┴─────────┘
                                                                      ▲
    Call API ──────────►  Rollout ┌─────────┬─────────┐               │
    ◄────── Receive Future        │ Worker 0│ Worker 1│ ──────────────┘
                                  └─────────┴─────────┘

    Call API ──────────►  Critic  ┌─────────┬─────────┐      ┌─────────┬─────────┐
    ◄────── Receive Future        │ Worker 0│ Worker 1│ ───► │  GPU 2  │  GPU 3  │  Resource Pool 1
                                  └─────────┴─────────┘      └─────────┴─────────┘

    Call API ──────────►  Reference ┌─────────┬─────────┐    ┌─────────┬─────────┐
    ◄────── Receive Future Policy   │ Worker 0│ Worker 1│ ─► │  GPU 4  │  GPU 5  │  Resource Pool 2
                                    └─────────┴─────────┘    └─────────┴─────────┘
                                                                      ▲
    Call API ──────────►  Reward  ┌─────────┬─────────┐               │
    ◄────── Receive Future Model  │ Worker 0│ Worker 1│ ──────────────┘
                                  └─────────┴─────────┘
```

**Key Components:**
| Component | Description |
|-----------|-------------|
| **Driver Process (usually the trainer)** | Single-controller that coordinates all operations via Call API / Receive Future pattern |
| **WorkerGroup** | Contains multiple workers (Worker 0, Worker 1, ...) for each role |
| **Roles** | Actor, Critic, Rollout, Reference Policy, Reward Model |
| **Resource Pool** | GPU resources that can be shared across worker groups |

**Resource Pool Mapping (Colocation):**
- **Resource Pool 0** (GPU 0, GPU 1): Shared by Actor and Rollout
- **Resource Pool 1** (GPU 2, GPU 3): Used by Critic
- **Resource Pool 2** (GPU 4, GPU 5): Shared by Reference Policy and Reward Model

## 2. Building blocks

verl uses Ray for distributed training. The core abstractions (all in `verl/single_controller/ray/base.py`):

| Abstraction | Purpose |
|-------------|---------|
| `RayResourcePool` | Manages GPU/node allocation |
| `RayWorkerGroup` | Wraps a group of Ray actors (aka workers) distributed across GPUs and nodes and runs the same program (aka Worker class) on all of them|
| `Worker` (`verl/single_controller/base/worker.py`) | Base class for implementing SPMD programs/blocks |

**Key concepts:**
- **Workers Group = SPMD block**: Same program runs on multiple process spread across GPUs/nodes. Each SPDM program is a class that inherits from `Worker`.
- **Roles**: Each SPMD block implements a role—Actor (policy training), Rollout (generation), Critic (value estimation), etc.
- **Driver**: Single-threaded process (usuallythe trainer) that orchestrates the training loop by calling into worker groups. This is where custom algorithm logic lives.

**For implementation work:** Existing roles (Actor, Rollout, Critic) are feature-rich and reusable. New algorithms typically modify the driver. Some may require hooks/modifications to existing workers or new worker types.


---

## RL Training Loop outline

Here's an outline of an on-policy actor-critic training loop implemented in verl (verl/trainer/ppo/ray_trainer.py) in the driver process:

```python
for prompts in dataloader:
    # Stage 1: Generation (Rollout); sampling from the policy
    batch = rollout_wg.generate_sequences(prompts)

    # Stage 2: Experience Preparation
    batch = reward_wg.compute_reward(batch)
    batch = reference_wg.compute_log_prob(batch)
    batch = critic_wg.compute_values(batch)
    batch = compute_advantage(batch, "gae")

    # Stage 3: Training (Parameter Updates)
    critic_wg.update_critic(batch)
    actor_wg.update_actor(batch)
```

1. The rollout worker group generates response sequences given prompts. This uses the generation backend (vLLM or SGLang) for efficient inference
2. The reward worker group scores the responses
3. The reference worker group computes log probabilities (for KL penalty)
4. The critic worker group computes value estimates
5. The advantage computation (e.g., GAE) combines these into training signals
6. The critic worker group updates the critic network
7. The actor worker group updates the actor network

Note that this is just rough outline. Some of these steps can be overlapped. Also, the Actor, Rollout, and Reference worker groups are merged into a single worker group in the default implementation (only for on-policy training) as they share the exact same physical resources (i.e. GPU).

## Common Abstractions, Roles, and Worker Groups

verl has four main types of workers that form distributed worker groups:

**ActorRolloutRefWorker:** Handles generation, policy updates, and reference computation
- `generate_sequences(prompts)` - produces response sequences via rollout engine
- `compute_log_prob(batch)` - computes log π(a|s) for current policy
- `update_actor(batch)` - performs policy gradient update
- `compute_ref_log_prob(batch)` - computes log π_ref(a|s) for KL penalty (if reference not separate)
Note that in verl terminology, the Actor role handles the training of the policy (using performant training backend like FSDP2) and the Rollout role handles the generation of trajectories from the policy (using efficient inference engine like SGLang) and these two roles sync their weights on-the-fly.

**CriticWorker:** Handles value function
- `compute_values(batch)` - computes V(s) for advantage estimation
- `update_critic(batch)` - performs value function update

**ReferenceWorker:** Handles reference policy (frozen) - optional, can be colocated with actor
- `compute_ref_log_prob(batch)` - computes log π_ref(a|s) for KL penalty

**RewardModelWorker:** Handles reward model computation (optional)
- `compute_rm_score(batch)` - computes reward scores using neural reward model

**RewardManagers:** They abstract whether you're using a reward model or a rule-based reward function.

**RLDataset:** Handles the loading and preprocessing of the dataset.

### How HybridFlow Programs are written in Practice

When you write trainer code like:
```python
batch = rollout_wg.generate_sequences(prompts)
batch = critic_wg.compute_values(batch)
actor_wg.update_actor(batch)
```

Each of these calls (`generate_sequences`, `compute_values`, `update_actor`) is actually an **RPC call** to a distributed worker group. The worker group internally uses SPMD parallelism (data parallel, tensor parallel, pipeline parallel) to execute the computation efficiently.

The `@register` decorator on worker methods exposes them as RPC endpoints and handles the distributed data transfer:

```python
class CriticWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, batch: DataProto):
        values = self.critic.forward(batch)
        batch.update(values=values)
        return batch

class ActorRolloutRefWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, batch: DataProto):
        metrics = self.actor.update_policy(batch)
        return metrics
```

The `dispatch_mode` tells verl how to shard and distribute the `DataProto` batch across the worker group's parallel dimensions. For example:
- `ONE_TO_ALL`: Broadcast same data to all workers
- `DP_COMPUTE_PROTO`: Shard data along the data-parallel dimension

## DataProto

`DataProto` is the universal data container that flows between all components (defined in `verl/protocol.py`, 1,253 lines). It's built on top of TensorDict for tensors and numpy arrays for non-tensors, and handles:
- Serialization and padding for distributed transfer
- Concatenation and splitting for batching
- Field accumulation through the pipeline

Key operations:
```python
# Creation
batch = DataProto.from_single_dict(batch_dict)

# Adding fields (mutable)
batch.update(values=values)
batch.update(rewards=rewards, log_probs=log_probs)

# Merging batches
batch = new_batch.union(gen_batch_output)  # merge fields from two batches
batch = DataProto.concat([batch1, batch2])  # concatenate along batch dim

# Indexing/slicing
subset = batch[indices]
subset = batch[:100]

# Splitting for mini-batches
mini_batches = batch.split(mini_batch_size)
```
Key patterns:
- `DataProto.from_single_dict()` to create batch from dataloader
- `.union()` to merge generation outputs into batch
- `DataProto.concat()` to accumulate across iterations
- Slicing with `batch[indices]` or `batch[:n]`

A typical DataProto accumulates these fields through the RL training pipeline:
- After dataloader: `prompt`, `data_source`, `reward_model` (ground truth), `extra_info`
- After generation: adds `responses`, `input_ids`, `attention_mask`, `position_ids`, `rollout_log_probs`
- After reference: adds `ref_log_prob`
- After reward: adds `rewards`, `rm_scores`
- After critic: adds `values`
- After advantage: adds `advantages`, `returns`
- After old_log_prob recomputation: adds `old_log_probs`, `entropys`

## Parallelism and Backends

### Training Backends

**FSDP / FSDP2:** PyTorch's Fully Sharded Data Parallel
- Simpler to set up, works well with most models
- ZeRO-3 style sharding (FULL_SHARD) or ZeRO-2 (SHARD_GRAD_OP)
- Supports device mesh for hybrid data/tensor parallelism
- Good for models up to ~70B parameters
- FSDP2 is our default training backend.

**Megatron:** NVIDIA's framework for large model training
- Better for very large models (100B+)
- Native tensor parallelism and pipeline parallelism
- Supports virtual pipeline stages for better load balancing
- More complex configuration but better scaling

Note that these are just the backends. We rarely interact with these directly. The algorithm code interact with the WorkerGroups that wrap these backends. Though for some custom implementations, you might need to interact with these backends directly.

### Generation Backends

**vLLM:**
- Mature and widely used

**SGLang:**
- Alternative inference engine with better performance characteristics
- This is our default inference engine.

### Parallelism Strategies

verl supports combining multiple parallelism dimensions:
- **Data Parallelism (DP):** Replicate model, shard data across replicas (available in FSDP & Megatron)
- **Tensor Parallelism (TP):** Shard model layers across GPUs (intra-layer) (only available in Megatron)
- **Pipeline Parallelism (PP):** Shard model stages across GPUs (inter-layer) (only available in Megatron)
- **Context/Sequence Parallelism (SP):** Shard along sequence length (Ulysses) (only available in FSDP & Megatron)
- **Expert Parallelism (EP):** For Mixture of Experts models (only available in Megatron)

### Default Available Hybrid Engine Optimizations

**Offloading & Reloading:**
- Move parameters to CPU when not actively used
- Enables training larger models than fit in GPU memory
- Automatically managed between generation and training phases
- Configured via `is_offload_param` and `is_offload_optimizer`

**Resharding:**
- Switch parallelism strategy between generation and training phases
- Generation might use different TP/PP than training
- `ShardingManager` handles the parameter conversion
- Enables optimal resource utilization for each phase

**Weight Synchronization:**
- After actor training, new weights are sent to rollout engine
- Supports LoRA for efficient weight updates
- FP8 quantization for memory efficiency during inference

**Sequence Packing (Remove Padding):**

Removes padding tokens and packs multiple sequences into single rows:
1. Concatenate sequences without padding
2. Adjust attention masks to prevent cross-contamination
3. Adjust position IDs for each sequence

This is enabled by default in the default implementation. You can disable it by setting `use_remove_padding=False` in the config.

**DP Load Balancing:**

Data parallelism requires synchronization between ranks. If ranks have unequal workloads, fast ranks wait for slow ones.

This is enabled by default in the default implementation. You can disable it by setting `balance_batch=False` in the config.

**Other Optimizations:**
- Gradient Checkpointing: Trade compute for memory with `activation_checkpointing=True`
- Torch Compile: JIT compilation for faster kernels
- Fused Kernels: Megatron's fused forward/backward kernels
- Flash Attention: Efficient attention implementation
- Prefix Grouper: Groups sequences by shared prefix for redundant computation elimination

## Codebase Structure

```
verl/
├── checkpoint_engine/        # Checkpoint management (NCCL, NIXL)
├── experimental/             # Experimental features (async policy, agent loops, VLA)
├── interactions/             # Task interaction implementations (GSM8K, weather)
├── model_merger/             # Model merging utilities
├── models/                   # Model implementations (Llama, Qwen with Megatron)
├── single_controller/        # Single controller orchestration (Ray-based)
│   ├── base/                 # Worker, WorkerGroup, ResourcePool abstractions
│   └── ray/                  # Ray-specific implementations
├── third_party/              # Third-party integrations (vLLM patches)
├── tools/                    # Tool implementations for agents
├── trainer/                  # Training logic (PPO trainer, SFT trainer)
│   └── ppo/                  # PPO algorithm: ray_trainer.py, core_algos.py
├── utils/                    # Core utilities (distributed, data, metrics)
├── workers/                  # Worker implementations
│   ├── actor/                # BasePPOActor, DataParallelPPOActor, MegatronPPOActor
│   ├── critic/               # Critic workers
│   ├── rollout/              # vLLM, SGLang, TensorRT-LLM rollout engines
│   ├── engine/               # FSDP, Megatron, MindSpeed engines
│   ├── reward_model/         # Reward model workers
│   └── reward_manager/       # Reward dispatch (Naive, PRIME, DAPO)
├── protocol.py               # DataProto - universal data container (1,253 lines)
└── base_config.py            # BaseConfig - frozen dataclass configuration
```

---

## 11. References

- **Repository:** https://github.com/volcengine/verl
- **Documentation:** https://verl.readthedocs.io/
- **Paper:** HybridFlow: A Flexible and Efficient RLHF Framework (EuroSys 2025)

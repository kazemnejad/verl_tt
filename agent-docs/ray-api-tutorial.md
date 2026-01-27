---
summary: How to write distributed code in verl—workers, data dispatch, GPU sharing patterns, and how verl uses Ray.
read_when:
  - Implementing new workers
  - Understanding dispatch decorators
  - Debugging distributed issues
  - GPU resource allocation and colocation
---

# verl Ray API - Single Controller Programming Guide

## Overview

verl uses Ray for single-controller architecture: one driver process orchestrates multiple distributed workers.

**Core Abstractions:**
- **Ray Actors** → stateful remote processes; methods return futures (ObjectRef)
- **ResourcePool** → allocates GPU resources; multiple worker groups can share (colocation)
- **RayWorkerGroup** → collection of workers; execute methods in parallel
- **Dispatch Decorators** → automatic data distribution driver↔workers
- **NVMegatronRayWorkerGroup** → specialized for Megatron TP/PP parallelism

---

## Key Concepts

### ResourcePool

Allocates GPUs. Multiple worker groups can share same pool (colocation pattern).

```python
resource_pool = RayResourcePool([4], use_gpu=True)  # 4 GPUs on one node
```

Key params: GPU count list, `use_gpu`, `name_prefix`, `max_colocate_count`

### Worker Base Class

Workers inherit from `Worker`. Provides `self.rank` and `self.world_size`.

### RayWorkerGroup

Manages workers, dispatches work. Input/output are lists of length `world_size`.

```python
worker_group = RayWorkerGroup(resource_pool, class_with_args)
results = worker_group.execute_all_sync("method", x=[...])
```

---

## Dispatch Decorators

The `@register` decorator automates data dispatch/collection:

```python
@register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True)
def my_method(self, data):
    ...
```

### Predefined Dispatch Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `Dispatch.ONE_TO_ALL` | Broadcast single input to all workers | Config, hyperparams |
| `Dispatch.ALL_TO_ALL` | Pass list directly (no transformation) | Pre-sharded data |
| `Dispatch.DP_COMPUTE` | Expects pre-split list of length world_size | Manual sharding |
| `Dispatch.DP_COMPUTE_PROTO` | Auto-shard DataProto by DP dim with padding | Training batches |
| `Dispatch.DP_COMPUTE_PROTO_WITH_FUNC` | Like DP_COMPUTE_PROTO but first arg is a function | Function + data |
| `Dispatch.DP_COMPUTE_METRIC` | Shard DataProto, collect raw list (no concat) | Metrics collection |
| `Dispatch.DIRECT_ROLLOUT_METHOD` | Special mode for vLLM external executor | vLLM rollout |

**Note:** Additional modes like `MEGATRON_COMPUTE` and `MEGATRON_PP_AS_DP` are registered in Megatron-specific modules.

### Execute Modes

| Mode | Description |
|------|-------------|
| `Execute.ALL` (default) | Execute on all workers |
| `Execute.RANK_ZERO` | Execute only on rank 0 |

### Custom Dispatch Functions

For complex patterns, pass a dict instead of a Dispatch enum:

```python
@register(dispatch_mode={
    "dispatch_fn": my_dispatch_fn,  # (worker_group, *args, **kwargs) -> (args, kwargs)
    "collect_fn": my_collect_fn,    # (worker_group, output) -> result
})
def custom_method(self, data):
    ...
```

Use `register_dispatch_mode(name, dispatch_fn, collect_fn)` to add new named modes globally.

### Blocking vs Non-Blocking Execution

The `blocking` parameter controls whether the driver waits for results:

```python
# Blocking (default): waits for all workers, returns actual results
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=True)
def train_step(self, batch):
    return self.model(batch)

# Non-blocking: returns Ray ObjectRefs immediately (futures)
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, blocking=False)
def train_step_async(self, batch):
    return self.model(batch)
```

| Mode | Behavior | Returns | Use Case |
|------|----------|---------|----------|
| `blocking=True` | Calls `ray.get()` internally | Actual values | Simple synchronous flow |
| `blocking=False` | Returns immediately | `ray.ObjectRef` futures | Overlapping compute/communication |

**Non-blocking pattern:**
```python
# Launch async, do other work, then collect
future = worker_group.train_step_async(batch)  # returns ObjectRef
# ... do other work while workers compute ...
result = ray.get(future)  # collect when needed
```

### Materialize Futures

The `materialize_futures` parameter (default `True`) controls whether `DataProtoFuture` inputs are resolved before dispatch:

- `True`: Calls `.get()` on any `DataProtoFuture` args before dispatching
- `False`: Passes futures through (worker must handle them)

---

## Megatron Integration

`NVMegatronRayWorkerGroup` + `MegatronWorker` for tensor/pipeline parallelism.

- Workers init Megatron parallel state via `mpu.initialize_model_parallel()`
- `MEGATRON_COMPUTE` dispatch handles TP/PP complexity transparently
- Input: list of `dp_size`; broadcasts within TP/PP groups

---

## Common Patterns

| Pattern | Description |
|---------|-------------|
| Actor/Critic colocation | Same ResourcePool → same GPUs |
| Rollout on separate GPUs | Different ResourcePools |
| Async execution | `.remote_async()` returns future; `ray.get()` when needed |

---

## Colocation: GPU-Level vs Process-Level

verl supports two levels of colocation for sharing GPU resources:

### GPU-Level Colocation (RayResourcePool)

Multiple **separate Ray actors** share the same GPU via `max_colocate_count`:

```python
# Allow up to 2 worker groups to share each GPU bundle
resource_pool = RayResourcePool([4], use_gpu=True, max_colocate_count=2)
```

- Each worker is a **separate process**
- GPU shared via CUDA time-slicing/MPS
- Memory is **not shared** (each process has its own copy)
- Communication requires Ray RPC (serialization overhead)
- Workers can run **concurrently**

Internally, fractional GPU allocation: `num_gpus = 1 / max_colocate_count`

### Process-Level Colocation (FusedWorker)

Multiple worker classes fused into the **same Ray actor process**:

```python
from verl.single_controller.ray.base import create_colocated_worker_cls_fused, RayClassWithInitArgs

# Fuse Actor and Critic into one process per GPU
class_dict = {
    "actor": RayClassWithInitArgs(cls=ActorWorker, ...),
    "critic": RayClassWithInitArgs(cls=CriticWorker, ...),
}
fused_cls = create_colocated_worker_cls_fused(class_dict)
worker_group = RayWorkerGroup(resource_pool, fused_cls)

# Access sub-workers via spawn or fuse
wg_dict = worker_group.spawn({"actor", "critic"})
wg_dict["actor"].compute(...)  # calls ActorWorker methods
wg_dict["critic"].compute(...)  # calls CriticWorker methods
```

- One process, one Python interpreter per GPU
- Memory **is shared** (objects passed by reference)
- Direct method calls (no serialization)
- Workers must **take turns** (no concurrency)
- If one fails, all fail together

### Comparison

| Aspect | RayResourcePool | FusedWorker |
|--------|-----------------|-------------|
| Isolation | Separate processes | Same process |
| Memory | Duplicated | Shared |
| Communication | Ray RPC (slow) | Direct calls (fast) |
| Concurrency | Can run in parallel | Must take turns |
| Failure | Independent | All fail together |

### When to Use Which

**Use RayResourcePool colocation when:**
- Workers need to run concurrently
- You want process isolation (fault tolerance)
- Workers have different lifetimes

**Use FusedWorker when:**
- Models don't run simultaneously (e.g., actor/critic take turns in PPO)
- You want to share memory between workers
- You need low-latency cross-worker communication
- GPU memory is constrained (avoid duplicate Python overhead)

---

## Resource Sharing (Driver-Worker Architecture)

| Resource Pool | Shared By |
|---------------|-----------|
| Pool 0 | Actor, Critic |
| Pool 1 | Rollout |
| Pool 2 | Reference Policy, Reward Model |

---

## Where to Look

| Topic | Location |
|-------|----------|
| Worker base class | `verl/single_controller/base/worker.py` |
| Dispatch decorators | `verl/single_controller/base/decorator.py` |
| RayResourcePool, RayWorkerGroup | `verl/single_controller/ray/base.py` |
| FusedWorker, colocation utilities | `verl/single_controller/ray/base.py` (lines 948-1040) |
| MegatronWorker | `verl/workers/megatron_workers.py` |
| FSDPRayWorker | `verl/workers/fsdp_workers.py` |
| EngineWorkers | `verl/workers/engine_workers.py` |
| DataProto | `verl/protocol.py` |

---

## Quick Reference

```python
# Minimal worker
@ray.remote
class MyWorker(Worker):
    @register(Dispatch.ONE_TO_ALL)
    def compute(self, x):
        return x + self.rank

# Setup
pool = RayResourcePool([4], use_gpu=True)
wg = RayWorkerGroup(pool, RayClassWithInitArgs(cls=MyWorker))
results = wg.compute(x=10)  # [10, 11, 12, 13]
```

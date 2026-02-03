---
summary: How sglang's inference engine (SRT) works—process architecture, request lifecycle, scheduling, KV cache, model execution, sampling, and how verl integrates with it.
read_when:
  - Understanding sglang internals
  - Debugging inference/rollout issues
  - Working with sglang weight sync in verl
  - Understanding RadixAttention / KV cache
  - Investigating scheduling or batching behavior
  - Working with tensor parallelism in inference
---

# SGLang Inference Engine (SRT) — Architecture Guide

> Based on sglang v0.5.6.post2 source at `/sgl-workspace/sglang/python/sglang/srt/`.

## 1. Overview

SGLang Runtime (SRT) is a high-performance LLM serving engine. Its core design:

- **Multi-process architecture** with ZeroMQ IPC between processes
- **Continuous batching** — requests dynamically join/leave running batches
- **RadixAttention** — radix-tree prefix cache for automatic KV reuse
- **Overlap scheduling** — CPU prepares next batch while GPU runs current batch
- **Pluggable backends** — FlashInfer, FlashAttention, Triton for attention; multiple quantization methods

The codebase lives under `sglang/srt/` (~300k lines total; Mini-SGLang is a 5k-line educational distillation).

## 2. Process Architecture

SRT runs three independent processes communicating via ZMQ IPC sockets:

```
┌─────────────────────────────────────────────────────────────┐
│                      MAIN PROCESS                           │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │          HTTP/gRPC Server (FastAPI + uvicorn)       │   │
│   │   Endpoints: /v1/completions, /v1/chat/completions, │   │
│   │              /v1/embeddings, /generate, ...         │   │
│   └──────────────────────┬──────────────────────────────┘   │
│                          │                                  │
│   ┌──────────────────────▼──────────────────────────────┐   │
│   │           TokenizerManager (async)                  │   │
│   │   - Tokenize text → token IDs                       │
│   │   - Track request state (ReqState)                  │
│   │   - Session management                              │
│   │   - LoRA adapter loading                            │
│   │   - Multimodal data preprocessing                   │
│   └──────────────────────┬──────────────────────────────┘   │
└──────────────────────────┼──────────────────────────────────┘
                           │ ZMQ IPC
              ┌────────────▼──────────────┐
              │   SCHEDULER SUBPROCESS    │  (one per TP group)
              │                           │
              │  ┌─────────────────────┐  │
              │  │     Scheduler       │  │
              │  │  - Batch formation  │  │
              │  │  - RadixCache       │  │
              │  │  - Memory pools     │  │
              │  │  - Event loop       │  │
              │  └─────────┬───────────┘  │
              │            │              │
              │  ┌─────────▼───────────┐  │
              │  │    ModelRunner      │  │
              │  │  - Forward pass     │  │
              │  │  - CUDA graphs      │  │
              │  │  - TP coordination  │  │
              │  │  - KV cache I/O     │  │
              │  └─────────┬───────────┘  │
              │            │              │
              │  ┌─────────▼───────────┐  │
              │  │      Sampler        │  │
              │  │  - Top-k/p/min-p    │  │
              │  │  - Grammar masks    │  │
              │  │  - Penalties        │  │
              │  └─────────────────────┘  │
              └────────────┬──────────────┘
                           │ ZMQ IPC
              ┌────────────▼──────────────┐
              │  DETOKENIZER SUBPROCESS   │
              │  - Token IDs → text       │
              │  - Incremental decode     │
              │  - Returns to Tokenizer   │
              └───────────────────────────┘
```

### Key files

| File | Lines | Role |
|------|-------|------|
| `entrypoints/engine.py` | ~930 | `Engine` class — orchestrates subprocess launch |
| `entrypoints/http_server.py` | ~2700 | FastAPI app, OpenAI-compatible API |
| `managers/tokenizer_manager.py` | ~1000 | `TokenizerManager` — request tokenization & state |
| `managers/scheduler.py` | ~3000 | `Scheduler` — core scheduling logic |
| `managers/detokenizer_manager.py` | — | Detokenizer subprocess |
| `server_args.py` | ~4700 | All config: `ServerArgs`, `PortArgs` |

### Launch sequence (`Engine._launch_subprocesses()`)

1. Configure `ServerArgs` (model, TP, DP, quantization, etc.)
2. Allocate `PortArgs` (ZMQ IPC socket names)
3. Launch scheduler subprocess(es) — one per TP group
4. Launch detokenizer subprocess
5. Init `TokenizerManager` in main process
6. Wait for model-ready signal via pipe

## 3. Data Structures

Requests transform through a pipeline of progressively GPU-ready structures:

```
HTTP JSON
  → GenerateReqInput          (io_struct.py — parsed request)
  → TokenizedGenerateReqInput (io_struct.py — after tokenization)
  → Req                       (schedule_batch.py — scheduler's request object)
  → ScheduleBatch             (schedule_batch.py — batch of Reqs for scheduling)
  → ModelWorkerBatch          (schedule_batch.py — GPU-ready subset)
  → ForwardBatch              (forward_batch_info.py — actual GPU tensors)
```

### `Req` — the request object (`managers/schedule_batch.py:455+`)

Core fields:

```
rid: str                          # unique request ID
origin_input_ids: List[int]       # original prompt tokens
output_ids: List[int]             # generated tokens (grows during decode)
fill_ids: List[int]               # origin_input_ids + output_ids (what's in KV cache)
sampling_params: SamplingParams   # temperature, top_k, top_p, etc.
kv_committed_len: int             # KV cache positions actually written
kv_allocated_len: int             # KV cache positions allocated
finished_reason: BaseFinishReason # None while active; set on completion
prefix_indices: List[int]         # KV indices from radix cache prefix match
last_node: TreeNode               # pointer into radix tree (for cache management)
```

### `ScheduleBatch` — batch for scheduling (`managers/schedule_batch.py:1081+`)

```
reqs: List[Req]
forward_mode: ForwardMode         # EXTEND, DECODE, MIXED, IDLE, ...
seq_lens: torch.Tensor            # per-request sequence lengths
input_ids: torch.Tensor           # flattened input token IDs
out_cache_loc: torch.Tensor       # KV cache write locations
extend_num_tokens: int            # total tokens in prefill batch
extend_lens: List[int]            # per-request extend lengths
prefix_lens: List[int]            # per-request cached prefix lengths
sampling_info: SamplingBatchInfo  # batched sampling parameters
```

### `ForwardBatch` — GPU tensors (`model_executor/forward_batch_info.py:226+`)

The final GPU-resident structure passed to `model.forward()`:

```
forward_mode: ForwardMode
batch_size: int
input_ids: torch.Tensor
seq_lens: torch.Tensor
positions: torch.Tensor
out_cache_loc: torch.Tensor
req_to_token_pool: ReqToTokenPool
token_to_kv_pool: KVCache
attn_backend: AttentionBackend
sampling_info: SamplingBatchInfo
```

### `ForwardMode` enum

```python
EXTEND = 1         # Prefill — process all input tokens
DECODE = 2         # Autoregressive — 1 token per request
MIXED  = 3         # Chunked prefill — both extend + decode in one forward
IDLE   = 4         # No work (DP workers when idle)
TARGET_VERIFY = 5  # Speculative decoding — verify draft tokens
DRAFT_EXTEND = 6   # Speculative decoding — draft model prefill
SPLIT_PREFILL = 8  # Piecewise multiplexed prefill
```

## 4. Request Lifecycle

```
   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ RECEIVED │────►│ WAITING  │────►│ PREFILL  │────►│ DECODE   │────►│ FINISHED │
   └──────────┘     └──────────┘     └──────────┘     └──────────┘     └──────────┘
                         │                                  │
                         │          prefix cache hit        │
                         └──────────────────────────────────┘
                              (skip some/all prefill)
```

### Stage-by-stage

1. **RECEIVED** — HTTP request parsed into `GenerateReqInput`; tokenized by `TokenizerManager`; sent to scheduler via ZMQ.

2. **WAITING** — `Req` object created; added to `waiting_queue` via `_add_request_to_queue()`.

3. **PREFILL (EXTEND)** — Scheduler pulls from `waiting_queue` via `get_new_batch_prefill()`:
   - `RadixCache.match_prefix()` finds longest cached prefix → saves recomputation
   - `alloc_for_extend()` allocates KV cache for remaining tokens
   - Forward pass in `ForwardMode.EXTEND` processes all non-cached tokens in parallel
   - Output: KV cache populated for entire input sequence

4. **DECODE** — Request moves to `running_batch` for autoregressive generation:
   - Each step: `alloc_for_decode()` allocates 1 new KV slot
   - Forward in `ForwardMode.DECODE` — only last token as input, attends over full KV
   - Sampler produces next token → appended to `output_ids`
   - Repeats until stopping condition

5. **FINISHED** — Stopping condition met (`max_tokens`, EOS, stop string, abort):
   - `finished_reason` set on `Req`
   - KV cache returned to pool
   - `cache_finished_req()` inserts full sequence into radix tree for future reuse
   - Output tokens sent to detokenizer → text returned to client

### Finish reasons

| Reason | Trigger |
|--------|---------|
| `FINISH_LENGTH` | Hit `max_tokens` or `max_new_tokens` |
| `FINISH_MATCHED_TOKEN` | Generated EOS or stop token |
| `FINISH_MATCHED_STR` | Generated text matches stop string |
| `FINISH_ABORT` | Request cancelled or error |

## 5. Scheduler & Event Loop

**File:** `managers/scheduler.py` (~3000 lines)

The Scheduler is a composition of mixins:

| Mixin | Purpose |
|-------|---------|
| `SchedulerOutputProcessorMixin` | Process forward pass results |
| `SchedulerUpdateWeightsMixin` | Online weight updates (for verl integration) |
| `SchedulerProfilerMixin` | CUDA profiling |
| `SchedulerMetricsMixin` | Prometheus metrics |
| `SchedulerDisaggregation{Decode,Prefill}Mixin` | Prefill/decode disaggregation |
| `SchedulerMultiplexMixin` | Request multiplexing |
| `SchedulerPPMixin` | Pipeline parallelism |
| `SchedulerDPAttnMixin` | Data-parallel attention |

### Core state

```python
self.waiting_queue: List[Req]           # requests awaiting execution
self.running_batch: ScheduleBatch       # continuous decode batch
self.cur_batch: Optional[ScheduleBatch] # batch currently on GPU
self.chunked_req: Optional[Req]         # partially-prefilled request
```

### Event loop

**Normal loop** (`event_loop_normal()`):
```python
while True:
    recv_reqs = recv_requests()         # ZMQ recv from TokenizerManager
    process_input_requests(recv_reqs)   # → waiting_queue
    batch = get_next_batch_to_run()     # scheduling decision
    result = run_batch(batch)           # GPU forward pass
    process_batch_result(batch, result) # update Reqs, filter finished
```

**Overlap loop** (`event_loop_overlap()`):
```
Time:  ─────────────────────────────────────────────────────►
GPU:   [  Batch N forward  ][  Batch N+1 forward  ][  Batch N+2  ]
CPU:        [Prep N+1]  [Process N result + Prep N+2]
```
- Result processing for batch N happens while batch N+1 runs on GPU
- Uses `result_queue` deque to pipeline
- `FutureMap` resolves placeholder tokens before GPU execution
- Impact: GPU utilization ~78% → ~97%

### Batch scheduling decision (`get_next_batch_to_run()`)

```
1. If new requests in waiting_queue:
   a. Try to form prefill batch via PrefillAdder
   b. PrefillAdder checks: enough GPU memory? fits budget?
   c. For each request: match prefix in RadixCache → compute extend_len
   d. If prefill batch formed → run EXTEND
2. Else if running_batch has requests:
   → run DECODE on running_batch
3. Else → IDLE
```

**Prefill budget:** controlled by `max_prefill_tokens` — limits total tokens in a single prefill batch to prevent GPU OOM.

**Chunked prefill:** If a request's extend length exceeds `chunked_prefill_size`, it's split across multiple prefill batches. The partially-prefilled request is stored in `self.chunked_req`.

### Scheduling policies (`managers/schedule_policy.py`)

**Cache-aware** (when radix tree enabled):
- **LPM** (Longest Prefix Match) — default; sort by `len(prefix_indices)` descending; maximizes cache hits
- **DFS-Weight** — depth-first tree traversal; schedules subtrees with most pending requests first

**Cache-agnostic:**
- **FCFS** — first-come-first-served
- **LOF** — longest-output-first
- **RANDOM** — shuffle

**Priority scheduling** (optional):
- Requests carry `priority` field; sorted before other criteria
- Supports preemption: low-priority requests can be evicted to make room for high-priority

## 6. Memory Management & KV Cache

### Two-level pool hierarchy

```
ReqToTokenPool
  Maps: req_pool_idx × position → token_kv_idx
  Shape: [max_requests, max_context_len] of int32

TokenToKVPoolAllocator
  Allocates/frees indices into physical KV tensors
  Interfaces with RadixCache for prefix reuse

KVCache (physical GPU memory)
  ├── MHATokenToKVPool     — standard multi-head attention
  ├── MLATokenToKVPool     — multi-latent-head (DeepSeek-style compressed KV)
  ├── NSATokenToKVPool     — nested sparse attention
  └── SWAKVPool            — sliding window attention
```

**File:** `mem_cache/memory_pool.py` (~69KB)

### Allocation flow

**Prefill (extend):**
```
prefix_len = RadixCache.match_prefix(fill_ids)   # how many tokens cached
extend_len = len(fill_ids) - prefix_len           # how many to compute
allocate extend_len slots from TokenToKVPoolAllocator
→ req.kv_allocated_len = prefix_len + extend_len
```

**Decode (per step):**
```
allocate 1 slot from TokenToKVPoolAllocator
→ req.kv_allocated_len += 1
```

**Eviction:** When pool is full, the radix cache evicts least-recently-used leaf nodes, freeing their KV indices back to the allocator.

## 7. RadixAttention — Prefix Cache

**File:** `mem_cache/radix_cache.py` (~1000 lines)

RadixAttention is sglang's signature innovation — a radix tree (compressed trie) storing KV cache indexed by token sequences. This enables **automatic** KV reuse across requests sharing common prefixes (system prompts, few-shot examples, etc.).

### Structure

```
                        ROOT (empty)
                       /            \
              [system prompt A]    [system prompt B]
              /          \                  |
       [user msg 1]  [user msg 2]    [user msg 3]
           |
    [assistant response]
```

Each `TreeNode` stores:
- `key`: token subsequence
- `value`: corresponding KV cache indices (int32 array)
- `children`: dict mapping next-token → child node
- `lock_ref`: reference count preventing eviction while in use
- `last_access_time`: for LRU eviction

### Match prefix (`match_prefix()`)

1. Start at root
2. Walk tree following input tokens
3. If match ends mid-node → split node at boundary
4. Return `MatchResult`:
   - `device_indices` — KV cache indices for matched prefix
   - `last_device_node` — terminal node of match
5. Lock matched nodes (`inc_lock_ref()`) to prevent eviction

### Insert (`cache_finished_req()`)

When a request finishes:
1. Extract full sequence: `origin_input_ids + output_ids[:kv_committed_len]`
2. Get corresponding KV indices from `req_to_token_pool`
3. Insert into radix tree
4. If prefix already cached → free duplicate indices back to pool
5. Unlock nodes

### Eviction strategies

| Strategy | Behavior |
|----------|----------|
| LRU | Evict least recently accessed leaves first |
| LFU | Evict least frequently accessed |
| FIFO | Evict oldest insertions |
| MRU | Evict most recently used (anti-thrashing) |
| Priority | Custom priority ordering |

Only unlocked leaf nodes can be evicted. Eviction frees KV indices back to `TokenToKVPoolAllocator`.

### Performance impact

- Multi-turn chat: ~67% cache hit rate (vs ~32% with round-robin)
- Shared system prompts: near-100% prefix reuse
- Few-shot prompting: automatic example caching

## 8. Model Execution Layer

**File:** `model_executor/model_runner.py` (~3800 lines)

### ModelRunner responsibilities

1. **Model loading** — via `DefaultModelLoader` from `model_loader/loader.py`
   - Supports: safetensors, pytorch, GGUF, GPTQ, AWQ
   - TP-aware weight sharding at load time
2. **Forward pass dispatch** — routes to `forward_extend()`, `forward_decode()`, or `forward_mixed()`
3. **CUDA graph management** — captures static decode graphs for reduced Python overhead (~90% less)
4. **KV cache pool init** — allocates GPU memory for token pools
5. **Attention backend init** — selects FlashInfer, FlashAttention, Triton, etc.

### Forward pass

**Extend (prefill):**
```
Input:  all non-cached tokens for each request (variable length)
Output: logits for last position; KV cache written for all positions
Kernel: FlashAttention-3 or equivalent (compute-bound)
```

**Decode:**
```
Input:  1 token per request (previous output)
Output: logits for next position; 1 KV slot written per request
Kernel: FlashInfer paged decode (memory-bound)
CUDA graphs used for static shapes → minimal Python overhead
```

**Mixed (chunked prefill):**
```
Both extend and decode requests in single forward pass
Controlled by enable_mixed_chunk=True
Single kernel handles heterogeneous sequence lengths
```

### CUDA graph runners

| Runner | Use case |
|--------|----------|
| `CudaGraphRunner` | Standard decode with fixed batch sizes |
| `CpuGraphRunner` | CPU-only execution |
| `PiecewiseCudaGraphRunner` | Large prefill with graph segments |

Graphs captured for common batch sizes at startup. During decode, the graph replays with minimal launch overhead.

### Data flow through model

```
ForwardBatch
  → Embedding lookup (input_ids → hidden_states)
  → N × TransformerLayer:
      → LayerNorm
      → RadixAttention (self-attention with KV cache)
        → Attention backend (FlashInfer/FA3/Triton)
        → KV cache read (past) + write (current)
      → MLP (with optional MoE routing)
  → Final LayerNorm
  → LogitsProcessor (lm_head projection)
  → Sampler
```

## 9. Sampling

**Files:** `layers/sampler.py` (~24KB), `sampling/sampling_batch_info.py`

### `SamplingBatchInfo` — per-batch parameters

```
temperatures: torch.Tensor      # per-request temperature
top_ps: torch.Tensor            # top-p (nucleus) threshold
top_ks: torch.Tensor            # top-k cutoff
min_ps: torch.Tensor            # min-p threshold
is_all_greedy: bool             # fast path when all temps=0
vocab_mask: torch.Tensor        # grammar constraint mask
penalizer_orchestrator          # frequency/repetition/presence penalties
logit_bias: torch.Tensor        # per-token logit adjustments
```

### Sampling pipeline

```
raw logits from model
  → custom logit processors (if any)
  → NaN detection & correction
  → penalty application (frequency, repetition, presence)
  → grammar constraint masking (vocab_mask → set invalid tokens to -inf)
  → logit bias addition
  → temperature scaling: logits /= temperature
  → softmax → probability distribution
  → top-k filtering → top-p (nucleus) filtering → min-p filtering
  → categorical sampling → token ID
  → (optional) log probability computation
```

**Fast paths:**
- All-greedy: `torch.argmax(logits, -1)` — skip softmax/sampling entirely
- FlashInfer sampling kernel: fused top-k/p on GPU (faster than PyTorch fallback)

### Grammar-constrained generation

Integration with xgrammar / outlines / llguidance:
```
JSON Schema → FSM (finite state machine) → per-state token bitmask → logit masking
```
The FSM tracks which tokens are valid at each generation step. Invalid tokens get `-inf` logits before sampling. Can skip 30-50% of generation steps for highly structured outputs via jump-forward optimization.

## 10. Attention Backends

**File:** `layers/attention/` (9+ backend implementations)

The attention computation is abstracted behind `AttentionBackend`:

```python
class AttentionBackend(ABC):
    def init_forward_metadata(forward_batch)  # prepare indices/buffers
    def forward_extend(q, k, v, layer, ...)   # prefill attention
    def forward_decode(q, k, v, layer, ...)   # decode attention
    def forward_mixed(q, k, v, layer, ...)    # chunked prefill
```

### Available backends

| Backend | File | Best for |
|---------|------|----------|
| FlashInfer | `flashinfer_backend.py` (111KB) | Default; paged KV; Hopper decode |
| FlashAttention | `flashattention_backend.py` | FA2/FA3; Hopper prefill |
| Triton | `triton_backend.py` | Custom Triton kernels |
| MLA | `mla_backend.py` | Multi-latent attention (DeepSeek) |
| NSA | `nsa_backend.py` | Nested sparse attention |
| Mamba | `mamba/` | State-space models |
| Ascend | `ascend_backend.py` | Huawei NPU |

**RadixAttention layer** (`layers/radix_attention.py`) wraps the backend selection:
```python
class RadixAttention(nn.Module):
    def forward(self, q, k, v, forward_batch, save_kv_cache=True):
        if forward_batch.forward_mode.is_extend():
            return attn_backend.forward_extend(q, k, v, ...)
        else:
            return attn_backend.forward_decode(q, k, v, ...)
```

## 11. Tensor Parallelism

**Files:** `layers/linear.py` (57KB), `distributed/parallel_state.py` (74KB)

### TP-aware layers

| Layer | Sharding | Communication |
|-------|----------|---------------|
| `ColumnParallelLinear` | Output dim split across ranks | All-gather (if `gather_output=True`) |
| `RowParallelLinear` | Input dim split across ranks | All-reduce after forward |
| `QKVParallelLinear` | Fused Q,K,V column-parallel | None (local per rank) |

### Weight loading with TP

```python
# Each rank loads only its shard:
tp_rank = get_tensor_model_parallel_rank()
tp_size = get_tensor_model_parallel_world_size()
shard_size = total_size // tp_size
start_idx = tp_rank * shard_size
loaded_weight = full_weight[start_idx : start_idx + shard_size]
```

### Communication groups (`parallel_state.py`)

| Group | Purpose |
|-------|---------|
| `tp_group` | Tensor parallel all-reduce/all-gather |
| `pp_group` | Pipeline parallel P2P |
| `dp_group` | Data parallel gradient sync |
| `moe_group` | MoE expert routing all-to-all |

## 12. Advanced Features

### Speculative decoding (`speculative/`)

Draft model generates K tokens ahead; target model verifies in parallel:
```
Draft: [t1, t2, t3, t4, t5]  (cheap model, fast)
Target: verify all 5 in one forward pass
Accept: [t1, t2, t3] ✓  reject: [t4, t5] ✗
→ save 2 forward passes
```
Tree-structured attention handles branching verification paths. Supports EAGLE and EAGLE3 draft models.

### Prefill-decode disaggregation (`disaggregation/`)

Separate GPU pools for prefill vs decode:
- **Prefill instances** — compute-bound; optimized for throughput
- **Decode instances** — memory-bound; optimized for latency
- KV cache transferred between instances via RDMA/NCCL
- 8 backend implementations (Mooncake, NIXL, P2P, etc.)

### LoRA support (`lora/`)

- Per-request LoRA adapter selection
- Adapter loading/unloading at runtime
- Integrated into scheduler and model forward pass
- Radix cache keys include LoRA ID for correct caching

### Multimodal (`multimodal/`)

- Image, video, audio processing
- Per-model processors in `multimodal/processors/`
- `AsyncMMDataProcessor` in TokenizerManager pipeline
- Multimodal embeddings injected into model input

### Constrained generation (`constraint/`)

- xgrammar, outlines, llguidance backends
- FSM-based token masking
- Jump-forward optimization for deterministic segments

## 13. verl Integration

verl wraps sglang for rollout (trajectory generation) during RL training. This section documents how verl launches, communicates with, and syncs weights to sglang.

### Integration files (in verl codebase)

| File | Class | Purpose |
|------|-------|---------|
| `verl/workers/rollout/sglang_rollout/sglang_rollout.py` | `ServerAdapter` | Primary wrapper; weight sync entry point |
| `verl/workers/rollout/sglang_rollout/async_sglang_server.py` | `SGLangHttpServer`, `SGLangReplica` | Server lifecycle management |
| `verl/workers/rollout/sglang_rollout/http_server_engine.py` | `AsyncHttpServerAdapter` | Low-level async HTTP client |
| `verl/workers/rollout/sglang_rollout/utils.py` | — | Tensor bucketization & broadcast |

### Rollout modes

| Mode | GPU sharing | Weight sync | Use case |
|------|-------------|-------------|----------|
| **HYBRID** | Training & rollout share GPU | `sgl_update_weights()` | On-policy (default) |
| **COLOCATED** | Same placement group, separate process | CPU-backed (no sync needed) | LLM-as-judge |
| **STANDALONE** | Dedicated GPU pool | Async HTTP updates | Off-policy |

### Launch flow (HYBRID mode)

```
ActorRolloutRefWorker.init_model()
  → SGLangReplica.launch_servers()
    → Per node: Ray remote SGLangHttpServer actor
    → SGLangHttpServer.launch_server()
      → sglang.srt.entrypoints.http_server.launch_server()
        → Engine._launch_subprocesses()
          → Scheduler + ModelRunner + Detokenizer
```

### Weight sync flow

After each training step, verl streams updated weights to sglang:

```
ServerAdapter.update_weights(weight_generator)
  → Bucketize tensors (configurable MB chunks)
  → Optional FP8 quantization
  → sgl_update_weights(engine, params_batch, device_mesh)
    → HTTP POST to /update_weights_from_tensor
  → flush_cache()  (invalidate stale KV cache)
```

Only TP rank 0 performs the sync; device mesh coordinates TP-aware distribution.

### Memory lifecycle (HYBRID mode)

```
Before rollout:  wake_up()   → resume_memory_occupation(["weights", "kv_cache"])
                 update_weights() → stream new policy weights
                 generate()  → run inference
After rollout:   sleep()     → release_memory_occupation(["weights", "kv_cache"])
                              → GPU memory freed for training
```

### Generation interface

All modes expose the same API via `BaseRollout`:
```python
async def generate(
    prompt_ids,
    sampling_params,
    request_id,
    image_data=None,
) -> TokenOutput  # token_ids, log_probs, routed_experts, stop_reason
```

Communication is async HTTP to sglang's `/generate` endpoint.

## 14. Configuration Reference

### Key `ServerArgs` parameters

| Parameter | Description |
|-----------|-------------|
| `model_path` | HuggingFace model path or local dir |
| `tp_size` | Tensor parallelism degree |
| `dp_size` | Data parallelism degree |
| `ep_size` | Expert parallelism (MoE) |
| `quantization` | `fp8`, `int8`, `gptq`, `awq`, etc. |
| `attention_backend` | `flashinfer`, `fa3`, `triton` |
| `schedule_policy` | `lpm`, `fcfs`, `lof`, `random` |
| `max_running_requests` | Max concurrent requests |
| `max_prefill_tokens` | Budget per prefill batch |
| `chunked_prefill_size` | Max tokens per prefill chunk |
| `mem_fraction_static` | Fraction of GPU mem for KV cache |
| `enable_mixed_chunk` | Allow mixed prefill+decode batches |
| `load_format` | `auto`, `safetensors`, `dummy` |
| `disaggregation_mode` | `null`, `prefill`, `decode` |

### ZMQ IPC ports (`PortArgs`)

| Socket | Direction |
|--------|-----------|
| `tokenizer_ipc_name` | Detokenizer → TokenizerManager |
| `scheduler_input_ipc_name` | TokenizerManager → Scheduler |
| `detokenizer_ipc_name` | Scheduler → Detokenizer |

## 15. Directory Map

```
sglang/srt/
├── entrypoints/          # HTTP/gRPC servers, Engine orchestration
├── managers/             # Scheduler, TokenizerManager, Detokenizer, IO structs
├── model_executor/       # ModelRunner, ForwardBatch, CUDA graphs
├── models/               # 100+ model implementations (Llama, Qwen, Mistral, ...)
├── layers/               # Attention, linear, sampler, logits, RoPE, MoE
│   ├── attention/        # FlashInfer, FA, Triton, MLA, NSA, Mamba backends
│   ├── quantization/     # FP8, INT8, INT4, GPTQ, AWQ, GGUF
│   └── moe/              # Mixture-of-experts routing
├── mem_cache/            # RadixCache, memory pools, KV cache
├── sampling/             # SamplingBatchInfo, penalty library
├── configs/              # ModelConfig (architecture definitions)
├── distributed/          # Parallel state, TP/PP/EP groups
├── disaggregation/       # Prefill-decode separation (8 backends)
├── speculative/          # EAGLE speculative decoding
├── constraint/           # Grammar-constrained generation
├── lora/                 # LoRA adapter management
├── multimodal/           # Image/video/audio processors
├── tokenizer/            # Tokenization utilities
├── model_loader/         # Weight loading, TP sharding
├── weight_sync/          # Online weight updates (verl integration)
├── connector/            # External backends (S3, Redis, remote)
├── grpc/                 # gRPC server implementation
├── function_call/        # Function calling support
├── tracing/              # Distributed tracing
├── metrics/              # Prometheus metrics
├── compilation/          # Kernel compilation
├── batch_overlap/        # Overlap scheduling implementation
├── checkpoint_engine/    # Checkpointing
├── eplb/                 # Expert load balancing
├── elastic_ep/           # Elastic expert parallelism
├── dllm/                 # Diffusion LLM
├── multiplex/            # Request multiplexing
├── utils/                # Common utilities
└── debug_utils/          # Debugging tools
```

## 16. External Resources

- [Mini-SGLang blog post (LMSYS, Dec 2025)](https://lmsys.org/blog/2025-12-17-minisgl/) — 5k-line educational distillation
- [SGLang deep dive (SugiV Blog)](https://blog.sugiv.fyi/sglang-deep-dive-inside-sglang) — detailed architecture walkthrough
- [DeepWiki reference](https://deepwiki.com/sgl-project/sglang) — component-level documentation
- [SGLang GitHub](https://github.com/sgl-project/sglang) — source of truth

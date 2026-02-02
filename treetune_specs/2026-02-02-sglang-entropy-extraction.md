# SGLang Per-Token Entropy Extraction

## Problem

RL training algorithms (PPO, GRPO) benefit from per-token entropy of the policy distribution — for regularization, monitoring, and diagnostics. sglang's sampler computes logits but discards the full distribution after sampling. The `output_token_entropy_val` field exists in sglang's output structs (`BatchTokenIDOutput`, `BatchStrOutput`) but is hardcoded to `None` and never populated. There is no way to get per-token entropy through sglang's API.

## Constraints

- **sglang 0.5.6.post2** — only version we target.
- **`spawn` multiprocessing** — sglang scheduler runs in a subprocess with fresh imports. Parent-process monkey-patches don't propagate.
- **Upstream isolation** — no direct modifications to sglang or verl source. All patches via subclassing, monkey-patching, or our own treetune code.
- **Pre-temperature entropy** — compute from raw logits before temperature scaling. Post-temperature support deferred but design should not preclude it.
- **Decode only** — no entropy for prefill/prompt tokens.
- **Always compute** — no `return_entropy` flag wiring. Entropy is computed whenever our custom scheduler process is active.
- **Configurable scope** — full-vocab or top-k entropy, with adjustable k. Configured via env var to cross the `spawn` boundary.

## Solution

Inject a custom `run_scheduler_process` function that runs inside the sglang scheduler subprocess, applying targeted patches to the Sampler and Scheduler before they are instantiated. Entropy flows through sglang's existing output structs, detokenizer, and tokenizer manager into `meta_info`, then through verl's existing output pipeline into `DataProto`.

## Configuration

Single env var crosses the `spawn` boundary:

```
TREETUNE_ENTROPY_TOP_K=0    → full-vocab entropy (default)
TREETUNE_ENTROPY_TOP_K=50   → top-k entropy, k=50
```

Set in parent from Hydra config before `_launch_subprocesses`. Read in `EntropySampler.__init__` inside the subprocess.

**Full-vocab**: `log_softmax` over entire vocab (~128k), then `-(exp(lp) * lp).sum(-1)`. Cost: ~0.5ms/step on A100. Captures total distributional uncertainty.

**Top-k**: `torch.topk(logits, k)` (cheap partial sort), then `log_softmax` over k values (renormalized), then entropy. Orders of magnitude cheaper. Captures effective uncertainty among plausible tokens. Bounded by `log(k)`.

Hydra config shape:

```yaml
rollout:
  entropy_top_k: 0  # 0 = full vocab, >0 = top-k
```

## Architecture

```
SUBPROCESS (scheduler process)
  custom_run_scheduler_process()          ← our function, pickled by reference
    → _apply_subprocess_patches()         ← patches Sampler, Scheduler methods
    → original run_scheduler_process()    ← creates Scheduler → ModelRunner → Sampler

  EntropySampler.forward()
    → reads TREETUNE_ENTROPY_TOP_K from env
    → full: log_softmax(raw_logits), entropy over vocab
    → top-k: topk(raw_logits, k), log_softmax over k, entropy over k
    → logits_output.next_token_entropy = entropy
    → super().forward()                   ← normal sampling proceeds

  patched process_batch_result_decode()
    → super()                             ← normal logprob processing
    → _entropy_store[rid].append(entropy) ← accumulate per-token entropy

  patched send_to_detokenizer.send_output()
    → populate output.output_token_entropy_val from _entropy_store
    → original send_output()              ← sends via ZMQ

DETOKENIZER SUBPROCESS
  → copies output_token_entropy_val through (already works, no patch needed)

PARENT PROCESS (tokenizer manager)
  patched convert_logprob_style()
    → accumulate entropy into ReqState
  patched add_logprob_to_meta_info()
    → meta_info["output_token_entropy"] = accumulated values

VERL (our code, no monkey-patching)
  async_sglang_server.py generate()
    → extract meta_info["output_token_entropy"]
    → return in TokenOutput

  agent_loop.py
    → pad to response_length, convert to tensor
    → stack into DataProto["rollout_entropy"]
```

## Injection mechanism

verl calls `_launch_subprocesses()` in `async_sglang_server.py:235`. Before that call, we monkey-patch:

```python
sglang.srt.entrypoints.engine.run_scheduler_process = custom_run_scheduler_process
```

This works because:
1. `_launch_subprocesses` looks up `run_scheduler_process` in `engine` module globals at call time (line 838: `target=run_scheduler_process`).
2. With `spawn`, `mp.Process` pickles our function by reference (module + qualname).
3. Child subprocess imports `treetune_verl.sglang.entropy`, gets our function.
4. Our function calls `_apply_subprocess_patches()` then delegates to the original.

Same pattern as verl's existing `_set_envs_and_config` patch at line 231.

**Upgrade path**: When sglang >= 0.5.7, `_launch_subprocesses` accepts `run_scheduler_process_func` as a parameter. verl already passes it (line 238). Switch from monkey-patching to parameter passing — no other changes needed.

## Subprocess patches (3)

All applied inside `_apply_subprocess_patches()`, which runs in the scheduler subprocess before `Scheduler` is instantiated.

### Patch 1: Sampler → EntropySampler

```python
import sglang.srt.layers.sampler as sampler_mod
sampler_mod.Sampler = EntropySampler
```

Applied before `model_runner.py` is imported (it's a lazy import inside `Scheduler.__init__`), so `from sglang.srt.layers.sampler import Sampler` in model_runner picks up our subclass.

`EntropySampler(Sampler)` overrides `forward()`:

```python
class EntropySampler(Sampler):
    def __init__(self):
        super().__init__()
        k = int(os.environ.get("TREETUNE_ENTROPY_TOP_K", "0"))
        self.entropy_top_k = k if k > 0 else None

    def forward(self, logits_output, sampling_info, return_logprob, ...):
        logits = logits_output.next_token_logits
        with torch.no_grad():
            if self.entropy_top_k is None:
                # Full-vocab entropy
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
            else:
                # Top-k entropy (renormalized)
                top_vals, _ = torch.topk(logits, self.entropy_top_k, dim=-1)
                log_probs = F.log_softmax(top_vals, dim=-1)
                entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        logits_output.next_token_entropy = entropy  # dynamic attr, no __slots__
        return super().forward(logits_output, sampling_info, return_logprob, ...)
```

Pre-temperature: operates on raw logits before `super().forward()` applies `div_(temperatures)` and in-place softmax.

**Post-temperature support (future)**: After `super().forward()`, the probs tensor is gone (`del logits` inside parent). Would require retaining probs before deletion — a deeper patch. Deferred.

### Patch 2: process_batch_result_decode

Wrap `Scheduler.process_batch_result_decode` to accumulate entropy per request:

```python
_entropy_store = {}  # rid → {'vals': List[float], 'offset': int}

_orig = Scheduler.process_batch_result_decode

def _patched(self, batch, result):
    _orig(self, batch, result)
    if hasattr(result, 'next_token_entropy') and result.next_token_entropy is not None:
        for i, req in enumerate(batch.reqs):
            rid = req.rid
            if rid not in _entropy_store:
                _entropy_store[rid] = {'vals': [], 'offset': 0}
            _entropy_store[rid]['vals'].append(result.next_token_entropy[i].item())

Scheduler.process_batch_result_decode = _patched
```

`result` is the `LogitsProcessorOutput` returned by the model runner. Our `EntropySampler` stored `next_token_entropy` on it.

Module-level `_entropy_store` lives in the subprocess. Keyed by `rid` (request ID). Offset tracks how many values have been sent (for streaming chunking, matching the logprobs pattern).

### Patch 3: send_to_detokenizer intercept

Wrap `Scheduler.__init__` to intercept `self.send_to_detokenizer.send_output`:

```python
_orig_init = Scheduler.__init__

def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)

    _orig_send = self.send_to_detokenizer.send_output

    def _entropy_send(output):
        if hasattr(output, 'output_token_entropy_val') and hasattr(output, 'rids'):
            entropy_per_rid = []
            for rid in output.rids:
                store = _entropy_store.get(rid)
                if store:
                    entropy_per_rid.append(store['vals'][store['offset']:])
                    store['offset'] = len(store['vals'])
                else:
                    entropy_per_rid.append([])
            output.output_token_entropy_val = entropy_per_rid

            # Cleanup finished requests
            if hasattr(output, 'finished_reasons'):
                for i, rid in enumerate(output.rids):
                    if output.finished_reasons[i] is not None:
                        _entropy_store.pop(rid, None)

        return _orig_send(output)

    self.send_to_detokenizer.send_output = _entropy_send

Scheduler.__init__ = _patched_init
```

This populates the existing `output_token_entropy_val` field on `BatchTokenIDOutput` (currently hardcoded to `None` at `scheduler_output_processor_mixin.py:995`) with actual per-token entropy values, sliced with offset tracking to match the logprobs chunking pattern.

Avoids copying the ~270-line `stream_output_generation` method entirely.

## Parent-process patches (2)

Applied in `async_sglang_server.py` alongside the existing `_set_envs_and_config` and `run_scheduler_process` patches. These target `TokenizerManager`, which runs in the parent process.

### Patch 4a: convert_logprob_style — entropy accumulation

Wrap to accumulate entropy from `BatchStrOutput` into `ReqState`:

```python
_orig_convert = TokenizerManager.convert_logprob_style

def _patched_convert(self, recv_obj, state, meta_info, recv_obj_index, ...):
    _orig_convert(self, recv_obj, state, meta_info, recv_obj_index, ...)

    entropy_vals = getattr(recv_obj, 'output_token_entropy_val', None)
    if entropy_vals is not None:
        per_req = entropy_vals[recv_obj_index] if isinstance(entropy_vals[recv_obj_index], list) else []
        if per_req:
            if not hasattr(state, 'output_token_entropy_val'):
                state.output_token_entropy_val = []
            state.output_token_entropy_val.extend(per_req)

TokenizerManager.convert_logprob_style = _patched_convert
```

Uses `getattr`/`hasattr` defensively — no need to patch `ReqState` dataclass `__init__`.

**Note on conditionality**: `convert_logprob_style` is only called when `return_logprob=True`. For RL training, logprobs are always requested (needed for importance sampling). If entropy is ever needed without logprobs, the accumulation would need to move to `_handle_batch_output` directly. Not a concern for the current use case.

### Patch 4b: add_logprob_to_meta_info — inject into meta_info

Wrap to add entropy to the response metadata dict:

```python
_orig_add = TokenizerManager.add_logprob_to_meta_info

def _patched_add(self, state, meta_info, ...):
    _orig_add(self, state, meta_info, ...)
    entropy = getattr(state, 'output_token_entropy_val', None)
    if entropy:
        meta_info["output_token_entropy"] = list(entropy)

TokenizerManager.add_logprob_to_meta_info = _patched_add
```

## verl integration (subclass chain — zero upstream changes)

All integration lives in `treetune_verl/`. No modifications to `verl/` files.

### EntropyTokenOutput

`treetune_verl/sglang/server.py` — subclass of `TokenOutput`:

```python
class EntropyTokenOutput(TokenOutput):
    entropy: Optional[list[float]] = None
```

### EntropySGLangHttpServer

`treetune_verl/sglang/server.py` — subclass of `SGLangHttpServer`:

**`launch_server()`** — apply entropy patches before calling `super().launch_server()`:

```python
async def launch_server(self, master_address=None, master_port=None):
    import sglang.srt.entrypoints.engine
    from treetune_verl.sglang.entropy import apply_parent_patches, custom_run_scheduler_process

    entropy_top_k = getattr(self.config, "entropy_top_k", 0)
    os.environ["TREETUNE_ENTROPY_TOP_K"] = str(entropy_top_k if entropy_top_k else 0)
    sglang.srt.entrypoints.engine.run_scheduler_process = custom_run_scheduler_process
    apply_parent_patches()

    await super().launch_server(master_address, master_port)
```

**`generate()`** — reimplements parent to extract entropy from `output["meta_info"]`:

```python
async def generate(self, prompt_ids, sampling_params, request_id, ...):
    # Same setup as parent: build GenerateReqInput, call tokenizer_manager
    output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()

    if return_logprob:
        # ... existing logprob extraction (same as parent) ...
        entropy = output["meta_info"].get("output_token_entropy", None)
    else:
        entropy = None

    return EntropyTokenOutput(token_ids=token_ids, log_probs=log_probs, entropy=entropy, ...)
```

Why reimplement: `super().generate()` consumes the raw output dict and discards
`meta_info["output_token_entropy"]`. No hook point to intercept.

### EntropySGLangReplica

`treetune_verl/sglang/server.py` — subclass of `SGLangReplica`:

```python
class EntropySGLangReplica(SGLangReplica):
    def __init__(self, ...):
        super().__init__(...)
        self.server_class = ray.remote(EntropySGLangHttpServer)
```

### Entropy agent loops

`treetune_verl/agent_loop/entropy_single_turn_agent_loop.py` — subclass of `SingleTurnAgentLoop`:

Reimplements `run()` to extract `output.entropy` from `EntropyTokenOutput` and store
in `AgentLoopOutput.extra_fields["response_entropy"]`. Registered as `"entropy_single_turn_agent"`.

`treetune_verl/agent_loop/entropy_tool_agent_loop.py` — subclass of `ToolAgentLoop`:

Reimplements `_handle_generating_state()`, `_handle_processing_tools_state()`,
`_handle_interacting_state()` to accumulate entropy alongside logprobs. Stores entropy
in `extra_fields["response_entropy"]`. Registered as `"entropy_tool_agent"`.

Entropy rides `extra_fields: dict[str, Any]` — the built-in extension point in `AgentLoopOutput`.
No need to subclass `AgentLoopOutput` or `_InternalAgentLoopOutput`.

### EntropyAgentLoopWorker

`treetune_verl/agent_loop/entropy_worker.py` — subclass of `AgentLoopWorker`:

**`_agent_loop_postprocess()`** — calls `super()`, then pads entropy to tensor:

```python
async def _agent_loop_postprocess(self, output, **kwargs):
    result = await super()._agent_loop_postprocess(output, **kwargs)
    entropy = output.extra_fields.get("response_entropy")
    if entropy is not None:
        response_length = self.config.actor_rollout_ref.rollout.response_length
        pad_size = response_length - len(entropy)
        result.extra_fields["response_entropy"] = torch.tensor(
            entropy + [0.0] * pad_size
        ).unsqueeze(0)
    return result
```

**`_postprocess()`** — pops entropy tensors before `super()`, then adds to DataProto:

```python
def _postprocess(self, inputs):
    entropy_tensors = None
    if "response_entropy" in inputs[0].extra_fields \
       and isinstance(inputs[0].extra_fields["response_entropy"], torch.Tensor):
        entropy_tensors = [inp.extra_fields.pop("response_entropy") for inp in inputs]

    data_proto = super()._postprocess(inputs)

    if entropy_tensors is not None:
        data_proto.batch["rollout_entropy"] = torch.cat(entropy_tensors, dim=0)
    return data_proto
```

## Directory layout

```
treetune_verl/
├── sglang/
│   ├── __init__.py
│   ├── entropy.py            # compute_entropy, EntropyStore, EntropySampler,
│   │                         #   custom_run_scheduler_process,
│   │                         #   _apply_subprocess_patches, apply_parent_patches
│   └── server.py             # EntropyTokenOutput, EntropySGLangHttpServer,
│                              #   EntropySGLangReplica
├── agent_loop/
│   ├── __init__.py
│   ├── entropy_single_turn_agent_loop.py  # EntropySingleTurnAgentLoop
│   ├── entropy_tool_agent_loop.py         # EntropyToolAgentLoop
│   └── entropy_worker.py                  # EntropyAgentLoopWorker
└── ...
```

Contents of `entropy.py`:
- `compute_entropy()` — full-vocab and top-k entropy computation
- `EntropyStore` — per-request entropy accumulation with offset tracking
- `EntropySampler(Sampler)` — sampler subclass with full/top-k support
- `_entropy_store` — module-level `EntropyStore` for subprocess entropy accumulation
- `_apply_subprocess_patches()` — applies patches 1-3 inside subprocess
- `custom_run_scheduler_process()` — subprocess entry point (calls `_apply_subprocess_patches` then original)
- `apply_parent_patches()` — applies patches 4a-4b on TokenizerManager in parent process

Patches applied in `EntropySGLangHttpServer.launch_server()` before `super().launch_server()`.

## Data flow summary

```
EntropySampler.forward()
  → logits_output.next_token_entropy: Tensor [batch_size]

process_batch_result_decode (patched)
  → _entropy_store.append(rid, float)

send_to_detokenizer (intercepted)
  → BatchTokenIDOutput.output_token_entropy_val: List[List[float]]  (per-rid sublists)

DetokenizerManager (no patch)
  → BatchStrOutput.output_token_entropy_val: List[List[float]]

convert_logprob_style (patched)
  → state.output_token_entropy_val: List[float]  (accumulated via extend)

add_logprob_to_meta_info (patched)
  → meta_info["output_token_entropy"]: List[float]

EntropySGLangHttpServer.generate()                     [treetune_verl/sglang/server.py]
  → EntropyTokenOutput.entropy: List[float]

Entropy{SingleTurn,Tool}AgentLoop                      [treetune_verl/agent_loop/]
  → AgentLoopOutput.extra_fields["response_entropy"]: List[float]

EntropyAgentLoopWorker._agent_loop_postprocess()       [treetune_verl/agent_loop/entropy_worker.py]
  → extra_fields["response_entropy"]: Tensor [1, response_length]

EntropyAgentLoopWorker._postprocess()
  → DataProto["rollout_entropy"]: Tensor [batch_size, response_length]
```

## Tensor key

`"rollout_entropy"` in DataProto, shape `[batch_size, response_length]`, padded with 0.0 for tokens beyond sequence length. Masked by existing `"response_mask"`.

## Testing

- **Unit**: Mock `LogitsProcessorOutput` with known logits. Verify `EntropySampler.forward()` produces correct entropy for both full and top-k modes (compare against manual `-(p*log p).sum()`). Verify `_entropy_store` accumulation logic. Verify `send_to_detokenizer` intercept populates and cleans up correctly.
- **Integration**: Small model, run sglang generation with entropy patches active. Verify `meta_info["output_token_entropy"]` appears with correct length and reasonable values (non-negative, bounded by `log(vocab_size)` or `log(k)`).
- **E2E**: Run a tiny GRPO step end-to-end. Verify `DataProto["rollout_entropy"]` tensor exists with expected shape.

## Risks

- **sglang version coupling**: Patches depend on method signatures and internal data flow of sglang 0.5.6.post2. Any sglang upgrade requires re-verification.
- **`send_to_detokenizer` intercept**: Assumes `send_output` is the sole send path for `BatchTokenIDOutput`. If sglang adds alternative paths, entropy would be missing.
- **`_entropy_store` memory**: Accumulates in subprocess memory. Cleaned up on request finish. If requests are abandoned without finishing, entries leak. Acceptable for RL batch generation (bounded request count).
- **Performance**: Full-vocab mode: one `log_softmax` over vocab (~128k) per decode step per batch, ~0.5ms on A100, negligible vs model forward. Top-k mode: `topk` + small `log_softmax`, even cheaper.

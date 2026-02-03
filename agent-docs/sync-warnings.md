# Upstream Sync Warnings

This doc catalogs every sync-sensitive site across the codebase -- places where we reimplement, copy, or monkey-patch upstream methods. These are **brittle**: if the upstream changes, our copies silently drift.

When upgrading verl or sglang, **diff each upstream source against the listed lines** and propagate changes.

## Tag convention

Every sync-sensitive site in our code is marked with:
```
# SYNC WARNING: <upstream_file>:<method> — see agent-docs/sync-warnings.md
```

---

## Entropy extraction feature

The entropy extraction feature (`treetune_verl/sglang/` and `treetune_verl/agent_loop/`) relies on reimplementing or monkey-patching several upstream methods.

### 1. `EntropySGLangHttpServer.generate()`

| | |
|---|---|
| **Our file** | `treetune_verl/sglang/server.py` — `generate()` |
| **Upstream** | `verl/workers/rollout/sglang_rollout/async_sglang_server.py` — `SGLangHttpServer.generate()` |
| **Why copied** | Parent's `generate()` consumes the raw output dict and discards `meta_info["output_token_entropy"]`. No hook to intercept. |
| **Our additions** | Lines extracting `entropy = output["meta_info"].get("output_token_entropy")` and returning `EntropyTokenOutput` instead of `TokenOutput`. |

### 2. `EntropySingleTurnAgentLoop.run()`

| | |
|---|---|
| **Our file** | `treetune_verl/agent_loop/entropy_single_turn_agent_loop.py` — `run()` |
| **Upstream** | `verl/experimental/agent_loop/single_turn_agent_loop.py` — `SingleTurnAgentLoop.run()` |
| **Why copied** | Parent returns `AgentLoopOutput` directly; no way to inject entropy into `extra_fields` after the fact. |
| **Our additions** | `getattr(token_output, "entropy", None)` extraction and `extra_fields["response_entropy"]` storage. |

### 3. `EntropyToolAgentLoop.run()`

| | |
|---|---|
| **Our file** | `treetune_verl/agent_loop/entropy_tool_agent_loop.py` — `run()` |
| **Upstream** | `verl/experimental/agent_loop/tool_agent_loop.py` — `ToolAgentLoop.run()` |
| **Why copied** | `agent_data` is local to `run()`; we need to init `agent_data.response_entropy = []` and read it at the end. |
| **Our additions** | `agent_data.response_entropy = []` init (line 92) and `extra_fields["response_entropy"]` storage at end. |

### 4. `EntropyToolAgentLoop._handle_generating_state()`

| | |
|---|---|
| **Our file** | `treetune_verl/agent_loop/entropy_tool_agent_loop.py` — `_handle_generating_state()` |
| **Upstream** | `verl/experimental/agent_loop/tool_agent_loop.py` — `ToolAgentLoop._handle_generating_state()` |
| **Why copied** | `output` (the `TokenOutput`) is consumed inside the method and not returned. Must access `output.entropy` during execution. |
| **Our additions** | Lines 172-177: `getattr(output, "entropy", None)` accumulation into `agent_data.response_entropy`. |

### 5. Sglang monkey-patches (signature-dependent)

| | |
|---|---|
| **Our file** | `treetune_verl/sglang/entropy.py` — `_apply_subprocess_patches()` and `apply_parent_patches()` |
| **Upstream** | sglang 0.5.6.post2 internals |
| **Methods patched** | `Sampler.forward` (6 params), `Scheduler.process_batch_result_decode`, `Scheduler.process_batch_result_prefill`, `Scheduler.__init__`, `SenderWrapper.send_output`, `TokenizerManager.convert_logprob_style` (7 params) |
| **Why brittle** | Any signature change in these methods breaks the patches silently. |
| **Version pinned** | sglang 0.5.6.post2 |

### 6. `custom_run_scheduler_process` signature

| | |
|---|---|
| **Our file** | `treetune_verl/sglang/entropy.py` — `custom_run_scheduler_process()` |
| **Upstream** | `sglang/srt/managers/scheduler.py` — `run_scheduler_process()` |
| **Why brittle** | Must match exact parameter signature (8 params). |

### Entropy upgrade checklist

When upgrading **verl**:
1. Diff `SGLangHttpServer.generate()` against our `EntropySGLangHttpServer.generate()`
2. Diff `SingleTurnAgentLoop.run()` against our `EntropySingleTurnAgentLoop.run()`
3. Diff `ToolAgentLoop.run()` and `._handle_generating_state()` against our versions
4. Check `AgentLoopWorker._agent_loop_postprocess()` and `._postprocess()` — we override both

When upgrading **sglang**:
1. Check `Sampler.forward` signature
2. Check `Scheduler.process_batch_result_decode` and `process_batch_result_prefill` signatures
3. Check `Scheduler.__init__` and `send_to_detokenizer` pattern
4. Check `TokenizerManager.convert_logprob_style` and `add_logprob_to_meta_info` signatures
5. Check `run_scheduler_process` signature
6. Check `BatchTokenIDOutput.output_token_entropy_val` field still exists

---

<!-- Add new feature sections below this line -->

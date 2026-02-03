# SGLang Per-Token Entropy Extraction — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-ng:executing-plans to implement this plan task-by-task.

**Goal:** Extract per-token policy entropy from sglang's sampler and flow it into verl's `DataProto["rollout_entropy"]` tensor for RL training.

**Architecture:** A custom `run_scheduler_process` function injected via monkey-patch runs inside sglang's scheduler subprocess, applying targeted patches to the Sampler and Scheduler before they are instantiated. Entropy flows through sglang's existing output structs, detokenizer, and tokenizer manager into `meta_info`, then through verl's output pipeline into `DataProto`. This is a **brittle implementation** — it patches sglang internals and depends on exact method signatures of sglang 0.5.6.post2. Thorough testing at every layer is critical.

**Tech Stack:** sglang 0.5.6.post2, PyTorch, verl (Ray-based RL framework), Hydra config

**Spec:** `treetune_specs/2026-02-02-sglang-entropy-extraction.md` — read the full spec before starting any task.

**Acceptance criteria:** All unit tests pass. Smoke test with sglang (no verl) shows entropy in output. E2E test shows `DataProto["rollout_entropy"]` tensor with correct shape.

**Key docs to read before starting:**
- `agent-docs/development-guide.md` — extension strategy, license headers, directory layout
- `agent-docs/sglang-engine-guide.md` — sglang internals, process architecture
- `agent-docs/testing-guide.md` — test naming, directory conventions
- `treetune_specs/2026-02-02-sglang-entropy-extraction.md` — full spec with all patch details

---

## Testing Strategy

Three test layers, each catching different failure modes:

### Layer 1: Unit tests (CPU, no sglang server)
File: `treetune_tests/treetune_verl/sglang/test_entropy_on_cpu.py`

Tests for:
- **`compute_entropy`**: uniform → log(N), deterministic → ~0, batch shape, bounded by log(vocab), top-k bounded by log(k), top-k picks highest logits
- **`EntropyStore`**: append/get, offset tracking (incremental reads), cleanup, missing rid returns empty, multi-rid isolation

These are the **acceptance criteria** — all must pass before proceeding to integration.

### Layer 2: Smoke test (GPU, sglang server, no verl)
File: `treetune_tests/treetune_verl/sglang/test_entropy_smoke.py`

Launches a real sglang server with a tiny model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`), applies entropy patches, sends a generation request with `return_logprob=True`, and verifies:
- `meta_info["output_token_entropy"]` exists
- Length matches number of generated tokens
- Values are non-negative and bounded by log(vocab_size)
- Both full-vocab and top-k modes work

Uses `sglang.Engine` directly — no verl involved.

### Layer 3: E2E test (GPU, full verl pipeline)
File: `treetune_tests/treetune_verl/sglang/test_entropy_e2e.sh`

Runs a tiny GRPO step end-to-end with the GSM8K task system. Verifies `DataProto["rollout_entropy"]` tensor exists with shape `[batch_size, response_length]`.

---

### Task 1: Package scaffolding and test file structure

**Files:**
- Create: `treetune_verl/sglang/__init__.py` (license header only)
- Create: `treetune_verl/sglang/entropy.py` (license header + module docstring)
- Create: `treetune_tests/treetune_verl/sglang/__init__.py` (license header only)
- Create: `treetune_tests/treetune_verl/sglang/test_entropy_on_cpu.py` (license header only)

**Steps:**
1. Create all four files with license headers per `development-guide.md`
2. `entropy.py` gets a docstring referencing the spec
3. Verify import: `python -c "import treetune_verl.sglang.entropy"`
4. Commit: `feat(sglang): scaffold entropy extraction package`

---

### Task 2: `compute_entropy` — TDD

**Files:**
- Modify: `treetune_tests/treetune_verl/sglang/test_entropy_on_cpu.py`
- Modify: `treetune_verl/sglang/entropy.py`

**Step 1: Write failing tests for `compute_entropy`**

Test class `TestComputeEntropy` with these test cases:
- `test_full_vocab_uniform`: uniform logits over 4 tokens → entropy = log(4)
- `test_full_vocab_deterministic`: one-hot-ish logits → entropy ≈ 0
- `test_top_k_uniform`: top-2 from uniform-4 → renormalized → entropy = log(2)
- `test_top_k_picks_largest`: verify top-k selects the k highest logits (compare against manual computation)
- `test_batch`: shape `(3, 100)` → output shape `(3,)`, all non-negative
- `test_bounded_by_log_vocab`: full-vocab entropy ≤ log(vocab_size)
- `test_top_k_bounded_by_log_k`: top-k entropy ≤ log(k)

Signature: `compute_entropy(logits: Tensor, top_k: Optional[int] = None) -> Tensor`

**Step 2: Run tests — verify ImportError**

Run: `pytest treetune_tests/treetune_verl/sglang/test_entropy_on_cpu.py -v`

**Step 3: Implement `compute_entropy`**

Per spec: full-vocab uses `log_softmax` over entire vocab, top-k uses `torch.topk` then `log_softmax` over k values. Both compute `-(exp(lp) * lp).sum(-1)`. Wrapped in `torch.no_grad()`.

**Step 4: Run tests — all 7 pass**

**Step 5: Commit:** `feat(sglang): add compute_entropy with full-vocab and top-k modes`

---

### Task 3: `EntropyStore` — TDD

**Files:**
- Modify: `treetune_tests/treetune_verl/sglang/test_entropy_on_cpu.py`
- Modify: `treetune_verl/sglang/entropy.py`

**Step 1: Write failing tests for `EntropyStore`**

Test class `TestEntropyStore`:
- `test_append_and_get`: append two values, get returns both
- `test_offset_tracking`: first get returns all, second get returns empty, append more, third get returns only new
- `test_cleanup`: after cleanup, get returns empty
- `test_missing_rid`: nonexistent rid returns empty list
- `test_multiple_rids`: values isolated per rid

Methods: `append(rid, value)`, `get_since_offset(rid) -> list[float]`, `cleanup(rid)`

**Step 2: Run tests — verify ImportError**

**Step 3: Implement `EntropyStore`**

Per spec: dict of `{rid: {vals: list, offset: int}}`. `get_since_offset` returns `vals[offset:]` and advances offset.

**Step 4: Run tests — all pass (both TestComputeEntropy and TestEntropyStore)**

**Step 5: Commit:** `feat(sglang): add EntropyStore for per-request entropy accumulation`

---

### Task 4: Subprocess patches and `custom_run_scheduler_process`

**Files:**
- Modify: `treetune_verl/sglang/entropy.py`

These cannot be unit-tested without a running sglang instance — correctness verified via smoke test (Task 7).

**Step 1: Implement `_apply_subprocess_patches()`**

Three patches per spec:
1. **Patch 1 (Sampler → EntropySampler):** Replace `sampler_mod.Sampler` with a subclass that calls `compute_entropy` on raw logits before `super().forward()`. Reads `TREETUNE_ENTROPY_TOP_K` from env in `__init__`. Important: must run before `model_runner.py` import.
2. **Patch 2 (process_batch_result_decode):** Wrap `Scheduler.process_batch_result_decode` to append `result.next_token_entropy[i].item()` to `_entropy_store` per request.
3. **Patch 3 (send_to_detokenizer intercept):** Wrap `Scheduler.__init__` to intercept `self.send_to_detokenizer.send_output`. Populate `output.output_token_entropy_val` from `_entropy_store` using offset tracking. Cleanup finished requests.

Reference: spec sections "Subprocess patches (3)" for exact method signatures and data flow.

**Step 2: Implement `custom_run_scheduler_process()`**

Top-level function (must be importable for `spawn` pickle). Calls `_apply_subprocess_patches()` then delegates to original `run_scheduler_process`. Must match the exact parameter signature — check `scheduler.py:2624`.

**Step 3: Verify module imports cleanly** (sglang imports are lazy, inside functions only)

Run: `python -c "from treetune_verl.sglang.entropy import custom_run_scheduler_process; print('OK')"`

**Step 4: Run existing unit tests — still pass**

**Step 5: Commit:** `feat(sglang): add subprocess patches and custom_run_scheduler_process`

---

### Task 5: Parent-process patches — `apply_parent_patches`

**Files:**
- Modify: `treetune_verl/sglang/entropy.py`

Also deferred to smoke test for correctness verification.

**Step 1: Implement `apply_parent_patches()`**

Two patches per spec:
1. **Patch 4a (convert_logprob_style):** Wrap `TokenizerManager.convert_logprob_style` to accumulate entropy from `recv_obj.output_token_entropy_val` into `state.output_token_entropy_val` via `extend`. Defensive `getattr`/`hasattr` — no ReqState modification.
2. **Patch 4b (add_logprob_to_meta_info):** Wrap `TokenizerManager.add_logprob_to_meta_info` to set `meta_info["output_token_entropy"]` from `state.output_token_entropy_val`.

Reference: spec sections "Parent-process patches (2)".

**Step 2: Verify import**

Run: `python -c "from treetune_verl.sglang.entropy import apply_parent_patches; print('OK')"`

**Step 3: Commit:** `feat(sglang): add parent-process patches for TokenizerManager entropy`

---

### Task 6a: Custom SGLang server and replica

**Files:**
- Create: `treetune_verl/sglang/server.py`

**Zero upstream changes.** All via subclassing.

**Step 1: Implement `EntropyTokenOutput(TokenOutput)`**

Subclass with one new field: `entropy: Optional[list[float]] = None`.

**Step 2: Implement `EntropySGLangHttpServer(SGLangHttpServer)`**

Two overrides:

`launch_server()` — apply entropy patches before `super().launch_server()`:
- Set `TREETUNE_ENTROPY_TOP_K` env var from `self.config.entropy_top_k` (default 0)
- Monkey-patch `sglang.srt.entrypoints.engine.run_scheduler_process = custom_run_scheduler_process`
- Call `apply_parent_patches()`
- Then `await super().launch_server(master_address, master_port)`

`generate()` — reimplements parent method to access raw output dict and extract entropy:
- Same logic as parent for prompt length validation, sampling_params setup, GenerateReqInput construction
- Calls `self.tokenizer_manager.generate_request(generate_request, None).__anext__()`
- Extracts logprobs (same as parent)
- Extracts `entropy = output["meta_info"].get("output_token_entropy", None)` when `return_logprob`
- Returns `EntropyTokenOutput` instead of `TokenOutput`
- WHY reimplement: `super().generate()` consumes the raw output dict and discards `meta_info["output_token_entropy"]`. No hook point to intercept.

**Step 3: Implement `EntropySGLangReplica(SGLangReplica)`**

Override `__init__()` to set `self.server_class = ray.remote(EntropySGLangHttpServer)`.

**Step 4: Verify imports**

```bash
python -c "from treetune_verl.sglang.server import EntropySGLangHttpServer, EntropySGLangReplica, EntropyTokenOutput; print('OK')"
```

**Step 5: Run existing unit tests — still pass**

**Step 6: Commit:** `feat(sglang): add entropy-aware SGLang server and replica subclasses`

---

### Task 6b: Entropy-aware single-turn agent loop

**Files:**
- Create: `treetune_verl/agent_loop/__init__.py` (license header only)
- Create: `treetune_verl/agent_loop/entropy_single_turn_agent_loop.py`

**Step 1: Implement `EntropySingleTurnAgentLoop(SingleTurnAgentLoop)`**

Reimplements `run()` (~30 lines). Same logic as parent:
- `apply_chat_template`, generate via `server_manager`, build `AgentLoopOutput`
- Added: extracts `output.entropy` from `EntropyTokenOutput` (with `getattr` defensive)
- Stores entropy in `AgentLoopOutput.extra_fields["response_entropy"]`

Register as `"entropy_single_turn_agent"` via `@register()`.

**Step 2: Verify import**

```bash
python -c "from treetune_verl.agent_loop.entropy_single_turn_agent_loop import EntropySingleTurnAgentLoop; print('OK')"
```

**Step 3: Run existing unit tests — still pass**

**Step 4: Commit:** `feat(agent_loop): add entropy-aware single-turn agent loop`

---

### Task 6c: Entropy-aware tool agent loop

**Files:**
- Create: `treetune_verl/agent_loop/entropy_tool_agent_loop.py`

**Step 1: Implement `EntropyToolAgentLoop(ToolAgentLoop)`**

Reimplements 3 state handlers + output construction. Same logic as parent with entropy additions:

- `_handle_generating_state()`: same + `agent_data.response_entropy += output.entropy`
  (attribute added dynamically on first use, defaults to `[]`)
- `_handle_processing_tools_state()`: same + `agent_data.response_entropy += [0.0] * len(response_ids)`
- `_handle_interacting_state()`: same + `agent_data.response_entropy += [0.0] * len(response_ids)`
- `run()` output construction: includes `extra_fields={"response_entropy": agent_data.response_entropy[:response_length]}`

Register as `"entropy_tool_agent"` via `@register()`.

**Step 2: Verify import**

**Step 3: Run existing unit tests — still pass**

**Step 4: Commit:** `feat(agent_loop): add entropy-aware tool agent loop`

---

### Task 6d: Entropy-aware agent loop worker

**Files:**
- Create: `treetune_verl/agent_loop/entropy_worker.py`

**Step 1: Implement `EntropyAgentLoopWorker(AgentLoopWorker)`**

Two overrides (both CAN use `super()`):

`_agent_loop_postprocess(output, **kwargs)`:
- `result = await super()._agent_loop_postprocess(output, **kwargs)`
- Extract `entropy = output.extra_fields.get("response_entropy")`
- Pad to `response_length`, convert to tensor `[1, response_length]`
- Store: `result.extra_fields["response_entropy"] = tensor`
- Return result

`_postprocess(inputs)`:
- Pop entropy tensors from each `input.extra_fields` (before super processes them into np.array)
- `data_proto = super()._postprocess(inputs)`
- `data_proto.batch["rollout_entropy"] = torch.cat(entropy_tensors, dim=0)`
- Return data_proto

**Step 2: Verify import**

**Step 3: Run existing unit tests — still pass**

**Step 4: Commit:** `feat(agent_loop): add entropy-aware agent loop worker`

---

### Task 7: Smoke test — sglang server with entropy

**Files:**
- Create: `treetune_tests/treetune_verl/sglang/test_entropy_smoke.py`

**Requires:** GPU. Skip gracefully if unavailable.

**Step 1: Write the smoke test**

Uses `sglang.Engine` directly (not verl). Test flow:
1. Apply patches: monkey-patch `engine.run_scheduler_process` and call `apply_parent_patches()`
2. Set `TREETUNE_ENTROPY_TOP_K=0` env var
3. Launch `sglang.Engine` with a small model (`Qwen/Qwen2.5-0.5B-Instruct`, tp=1)
4. Send a generation request with `return_logprob=True`, short `max_new_tokens` (e.g. 16)
5. Assert `meta_info["output_token_entropy"]` exists in the response
6. Assert length matches number of generated tokens
7. Assert all values non-negative
8. Assert all values ≤ log(vocab_size) (vocab_size from model config or generous upper bound)
9. Repeat with `TREETUNE_ENTROPY_TOP_K=10` — assert values ≤ log(10)
10. Shutdown engine

Reference sglang Engine API. Mark test with `@pytest.mark.skipif(not torch.cuda.is_available())`.

**Step 2: Run it**

Run: `pytest treetune_tests/treetune_verl/sglang/test_entropy_smoke.py -v -s`

This is the critical integration checkpoint — if entropy doesn't appear here, patches 1-5 have a bug. Debug with logging.

**Step 3: Commit:** `test(sglang): add entropy smoke test with sglang Engine`

---

### Task 8: E2E test — entropy through full verl pipeline

**Files:**
- Create: `treetune_tests/treetune_verl/sglang/test_entropy_e2e.sh`

**Requires:** GPU. Uses the existing GSM8K GRPO task system e2e test as a template.

**Step 1: Write E2E test script**

Reference `treetune_tests/treetune_verl/tasks/test_grpo_gsm8k_e2e.sh` for the pattern. Modifications:
- Use `EntropySGLangReplica` instead of `SGLangReplica` (config or code injection)
- Use `EntropyAgentLoopWorker` instead of `AgentLoopWorker`
- Use `"entropy_single_turn_agent"` or `"entropy_tool_agent"` as agent loop name
- Add `rollout.entropy_top_k=0` to Hydra overrides
- After training completes, verify `DataProto["rollout_entropy"]` exists with shape `[batch_size, response_length]`

Use `set -xeuo pipefail` and `PYTHONPATH` export per testing guide.

**Step 2: Run it**

Run: `bash treetune_tests/treetune_verl/sglang/test_entropy_e2e.sh`

**Step 3: Commit:** `test(sglang): add entropy E2E test through full verl pipeline`

---

### Task 9: Lint, final verification

**Files:** All modified and created files.

**Step 1: Run pre-commit hooks**

```bash
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always check-license
```

Fix any issues.

**Step 2: Run full unit test suite**

```bash
pytest treetune_tests/treetune_verl/sglang/test_entropy_on_cpu.py -v
```

All must pass — this is the acceptance criterion.

**Step 3: Verify public API imports**

```bash
python -c "from treetune_verl.sglang.entropy import compute_entropy, EntropyStore, custom_run_scheduler_process, apply_parent_patches; print('OK')"
python -c "from treetune_verl.sglang.server import EntropySGLangHttpServer, EntropySGLangReplica; print('OK')"
python -c "from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker; print('OK')"
```

**Step 4: Commit any lint fixes:** `style(sglang): fix lint issues in entropy extraction`

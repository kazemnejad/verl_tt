# Copyright 2025 Individual Contributor: Amirhossein Kazemnejad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-token entropy extraction from sglang's sampler.

Spec: treetune_specs/2026-02-02-sglang-entropy-extraction.md
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    """Compute per-row entropy from logits.

    Args:
        logits: Raw logits tensor of shape ``(..., vocab_size)``.
        top_k: If *None*, compute full-vocab entropy.  If a positive int,
            take the *k* largest logits per row, renormalize via softmax,
            and compute entropy over those *k* values.

    Returns:
        Entropy tensor of shape ``(...)``, i.e. the last dimension is
        reduced.
    """
    with torch.no_grad():
        if top_k is not None:
            values, _ = torch.topk(logits, top_k, dim=-1)
            log_probs = F.log_softmax(values, dim=-1)
        else:
            log_probs = F.log_softmax(logits, dim=-1)
        return -(torch.exp(log_probs) * log_probs).sum(dim=-1)


class EntropyStore:
    """Per-request entropy accumulation with offset tracking."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def append(self, rid: str, value: float) -> None:
        if rid not in self._store:
            self._store[rid] = {"vals": [], "offset": 0}
        self._store[rid]["vals"].append(value)

    def get_since_offset(self, rid: str) -> list[float]:
        entry = self._store.get(rid)
        if entry is None:
            return []
        result = entry["vals"][entry["offset"] :]
        entry["offset"] = len(entry["vals"])
        return result

    def cleanup(self, rid: str) -> None:
        self._store.pop(rid, None)


# ---------------------------------------------------------------------------
# Module-level store (one per scheduler subprocess)
# ---------------------------------------------------------------------------

_entropy_store = EntropyStore()


# ---------------------------------------------------------------------------
# Subprocess monkey-patches
# ---------------------------------------------------------------------------


def _apply_subprocess_patches() -> None:
    """Patch sglang internals to compute/propagate per-token entropy.

    Must be called **inside** the scheduler subprocess, **before**
    ``run_scheduler_process`` creates the ``Scheduler`` instance.

    All sglang imports are intentionally lazy so that importing this module
    at the top level never triggers heavy sglang loads.
    """

    # ---- Patch 1: Sampler -> EntropySampler ----------------------------------
    import sglang.srt.layers.sampler as sampler_mod
    from sglang.srt.layers.sampler import Sampler

    class EntropySampler(Sampler):
        """Sampler that also computes per-token entropy."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _top_k_env = os.environ.get("TREETUNE_ENTROPY_TOP_K")
            # 0 or unset → None (full-vocab); positive int → top-k
            self.entropy_top_k: int | None = int(_top_k_env) if _top_k_env else None
            if self.entropy_top_k is not None and self.entropy_top_k <= 0:
                self.entropy_top_k = None

        def forward(
            self,
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
        ):
            if logits_output.next_token_logits is not None:
                entropy = compute_entropy(logits_output.next_token_logits, top_k=self.entropy_top_k)
                logits_output.next_token_entropy = entropy
            return super().forward(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                positions,
            )

    sampler_mod.Sampler = EntropySampler

    # Also patch the already-imported reference in model_runner so that
    # ``ModelRunner.__init__`` instantiates our ``EntropySampler``.
    import sglang.srt.model_executor.model_runner as model_runner_mod

    model_runner_mod.Sampler = EntropySampler

    # ---- Patch 2: process_batch_result_decode --------------------------------
    from sglang.srt.managers.scheduler import Scheduler

    _orig_process = Scheduler.process_batch_result_decode

    def _patched_process(self, batch, result):
        # Store entropy BEFORE calling the original, because the original
        # calls stream_output which drains the store.
        logits_output = result.logits_output
        if (
            logits_output is not None
            and hasattr(logits_output, "next_token_entropy")
            and logits_output.next_token_entropy is not None
        ):
            for i, req in enumerate(batch.reqs):
                _entropy_store.append(req.rid, logits_output.next_token_entropy[i].item())
        _orig_process(self, batch, result)

    Scheduler.process_batch_result_decode = _patched_process

    # ---- Patch 2b: process_batch_result_prefill (first generated token) ------
    _orig_process_prefill = Scheduler.process_batch_result_prefill

    def _patched_process_prefill(self, batch, result):
        # Store entropy BEFORE calling the original, because the original
        # calls stream_output which drains the store.
        if self.is_generation:
            logits_output = result.logits_output
            if (
                logits_output is not None
                and hasattr(logits_output, "next_token_entropy")
                and logits_output.next_token_entropy is not None
            ):
                for i, req in enumerate(batch.reqs):
                    if req.finished() or req.is_retracted:
                        continue
                    if getattr(req, "is_chunked", 0) <= 0:
                        _entropy_store.append(req.rid, logits_output.next_token_entropy[i].item())
        _orig_process_prefill(self, batch, result)

    Scheduler.process_batch_result_prefill = _patched_process_prefill

    # ---- Patch 3: intercept send_to_detokenizer via Scheduler.__init__ -------
    _orig_init = Scheduler.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)

        _orig_send = self.send_to_detokenizer.send_output

        def _entropy_send(output, recv_obj=None):
            if hasattr(output, "output_token_entropy_val") and hasattr(output, "rids"):
                entropy_per_rid: list[list[float]] = []
                for rid in output.rids:
                    entropy_per_rid.append(_entropy_store.get_since_offset(rid))
                output.output_token_entropy_val = entropy_per_rid

                # Cleanup finished requests
                if hasattr(output, "finished_reasons"):
                    for i, rid in enumerate(output.rids):
                        if output.finished_reasons[i] is not None:
                            _entropy_store.cleanup(rid)

            return _orig_send(output, recv_obj)

        self.send_to_detokenizer.send_output = _entropy_send

    Scheduler.__init__ = _patched_init


# ---------------------------------------------------------------------------
# Parent-process monkey-patches (TokenizerManager)
# ---------------------------------------------------------------------------


def apply_parent_patches():
    """Apply parent-process patches to TokenizerManager for entropy propagation.

    Patches convert_logprob_style to accumulate entropy from recv_obj into
    state and inject it directly into meta_info.
    """
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    # Patch 4: convert_logprob_style — accumulate entropy from recv_obj
    # and inject directly into meta_info.
    #
    # Why not use add_logprob_to_meta_info?  Because the original
    # convert_logprob_style calls add_logprob_to_meta_info internally
    # *before* we get a chance to accumulate entropy into state.  So we
    # inject entropy into meta_info right here after the original call.
    _orig_convert = TokenizerManager.convert_logprob_style

    def _patched_convert(
        self, meta_info, state, top_logprobs_num, token_ids_logprob, return_text_in_logprobs, recv_obj, recv_obj_index
    ):
        _orig_convert(
            self,
            meta_info,
            state,
            top_logprobs_num,
            token_ids_logprob,
            return_text_in_logprobs,
            recv_obj,
            recv_obj_index,
        )

        entropy_vals = getattr(recv_obj, "output_token_entropy_val", None)
        if entropy_vals is not None:
            per_req = entropy_vals[recv_obj_index] if isinstance(entropy_vals[recv_obj_index], list) else []
            if per_req:
                if not hasattr(state, "output_token_entropy_val"):
                    state.output_token_entropy_val = []
                state.output_token_entropy_val.extend(per_req)
                # Inject directly into meta_info (add_logprob_to_meta_info
                # has already been called by _orig_convert, so we must do
                # this ourselves).
                meta_info["output_token_entropy"] = list(state.output_token_entropy_val)

    TokenizerManager.convert_logprob_style = _patched_convert


# ---------------------------------------------------------------------------
# Custom scheduler entry-point (must be top-level for spawn/pickle)
# ---------------------------------------------------------------------------


def custom_run_scheduler_process(
    server_args,
    port_args,
    gpu_id,
    tp_rank,
    moe_ep_rank,
    pp_rank,
    dp_rank,
    pipe_writer,
):
    """Drop-in replacement for ``sglang.srt.managers.scheduler.run_scheduler_process``.

    Applies entropy patches first, then delegates to the original function.
    """
    _apply_subprocess_patches()
    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        pipe_writer,
    )

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
            self.entropy_top_k: int | None = int(_top_k_env) if _top_k_env is not None else None

        def forward(
            self,
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
        ):
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

    # ---- Patch 2: process_batch_result_decode --------------------------------
    from sglang.srt.managers.scheduler import Scheduler

    _orig_process = Scheduler.process_batch_result_decode

    def _patched_process(self, batch, result):
        _orig_process(self, batch, result)
        logits_output = result.logits_output
        if (
            logits_output is not None
            and hasattr(logits_output, "next_token_entropy")
            and logits_output.next_token_entropy is not None
        ):
            for i, req in enumerate(batch.reqs):
                _entropy_store.append(req.rid, logits_output.next_token_entropy[i].item())

    Scheduler.process_batch_result_decode = _patched_process

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

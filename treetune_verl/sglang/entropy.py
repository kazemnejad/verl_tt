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

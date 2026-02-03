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

"""Smoke test: launch a real sglang Engine with entropy patches and verify output."""

from __future__ import annotations

import math
import os

import pytest
import torch

# Skip entire module if no GPU
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def engine():
    """Launch sglang Engine with entropy patches applied."""
    import sglang.srt.entrypoints.engine as engine_mod
    from sglang.srt.entrypoints.engine import Engine
    from treetune_verl.sglang.entropy import (
        apply_parent_patches,
        custom_run_scheduler_process,
    )

    # Configure full-vocab entropy
    os.environ["TREETUNE_ENTROPY_TOP_K"] = "0"

    # Monkey-patch scheduler entry-point so the subprocess uses our code
    engine_mod.run_scheduler_process = custom_run_scheduler_process

    # Apply parent-process patches (TokenizerManager)
    apply_parent_patches()

    eng = Engine(model_path=MODEL, tp_size=1)
    yield eng
    eng.shutdown()


class TestEntropySmoke:
    """End-to-end tests verifying per-token entropy flows through sglang."""

    def test_entropy_exists_in_meta_info(self, engine):
        """Entropy list must appear in meta_info when logprobs are requested."""
        output = engine.generate(
            prompt="What is 2+2?",
            sampling_params={"max_new_tokens": 16, "temperature": 0.7},
            return_logprob=True,
        )
        meta = output["meta_info"]
        assert "output_token_entropy" in meta, f"Keys present: {list(meta.keys())}"
        entropy = meta["output_token_entropy"]
        assert len(entropy) > 0, "Entropy list is empty"

    def test_entropy_length_matches_logprobs(self, engine):
        """Entropy list length must equal output_token_logprobs length."""
        output = engine.generate(
            prompt="Count to five:",
            sampling_params={"max_new_tokens": 32, "temperature": 0.7},
            return_logprob=True,
        )
        entropy = output["meta_info"]["output_token_entropy"]
        logprobs = output["meta_info"]["output_token_logprobs"]
        assert len(entropy) == len(logprobs), f"entropy len {len(entropy)} != logprobs len {len(logprobs)}"

    def test_entropy_length_matches_output_ids(self, engine):
        """Entropy list length must equal output_ids length (one entropy per sampled token)."""
        output = engine.generate(
            prompt="Write a haiku:",
            sampling_params={"max_new_tokens": 48, "temperature": 0.7},
            return_logprob=True,
        )
        entropy = output["meta_info"]["output_token_entropy"]
        output_ids = output["output_ids"]
        assert len(entropy) == len(output_ids), (
            f"entropy len {len(entropy)} != output_ids len {len(output_ids)}; "
            f"each entropy should correspond to one sampled token"
        )

    def test_entropy_non_negative(self, engine):
        """All per-token entropy values must be >= 0."""
        output = engine.generate(
            prompt="The meaning of life is",
            sampling_params={"max_new_tokens": 16, "temperature": 0.7},
            return_logprob=True,
        )
        entropy = output["meta_info"]["output_token_entropy"]
        assert all(v >= 0 for v in entropy), f"Negative entropy found: {[v for v in entropy if v < 0]}"

    def test_entropy_bounded_by_log_vocab(self, engine):
        """Entropy must be <= log(vocab_size); use generous upper bound."""
        output = engine.generate(
            prompt="Summarize gravity:",
            sampling_params={"max_new_tokens": 16, "temperature": 0.7},
            return_logprob=True,
        )
        entropy = output["meta_info"]["output_token_entropy"]
        # Qwen2.5-0.5B vocab ~151936; generous bound
        upper = math.log(200000)
        assert all(v <= upper for v in entropy), f"Max entropy {max(entropy):.4f} exceeds bound {upper:.4f}"

    def test_no_entropy_without_logprob(self, engine):
        """Without return_logprob, entropy should be absent or None."""
        output = engine.generate(
            prompt="Hello",
            sampling_params={"max_new_tokens": 8, "temperature": 0.7},
            return_logprob=False,
        )
        meta = output["meta_info"]
        ent = meta.get("output_token_entropy")
        assert ent is None or ent == [], f"Expected no entropy without logprobs, got: {ent}"

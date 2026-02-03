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

"""GPU test: validate sglang per-token entropy against HuggingFace forward pass.

Generates a single sequence with sglang (entropy-patched Engine), then runs
the same prompt+response through HF to get reference logits.  Computes entropy
from both and asserts they match within tolerance.

Requires: GPU, sglang, transformers.  Both engines use eager attention (no CUDA graphs).
"""

from __future__ import annotations

import os

import pytest
import torch

# Skip entire module if no GPU
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
PROMPT = "Explain why the sky is blue in one sentence."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-position entropy from raw logits.

    Mirrors ``treetune_verl.sglang.entropy.compute_entropy`` exactly:
        probs = softmax(logits.float(), dim=-1)
        log_probs = log(probs + 1e-12)   [equivalent to log_softmax for full vocab]
        entropy = -sum(probs * log_probs, dim=-1)

    We use the numerically cleaner log_softmax formulation which is
    mathematically identical and matches the production code.
    """
    with torch.no_grad():
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        return -(probs * log_probs).sum(dim=-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer (shared by sglang and HF fixtures)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def prompt_ids(tokenizer):
    """Tokenize the prompt once; used by both sglang and HF."""
    return tokenizer.encode(PROMPT)


@pytest.fixture(scope="module")
def engine():
    """Launch sglang Engine with entropy patches (full-vocab mode)."""
    import sglang.srt.entrypoints.engine as engine_mod
    from sglang.srt.entrypoints.engine import Engine
    from treetune_verl.sglang.entropy import (
        apply_parent_patches,
        custom_run_scheduler_process,
    )

    # Full-vocab entropy (top_k disabled)
    os.environ["TREETUNE_ENTROPY_TOP_K"] = "0"

    # Monkey-patch scheduler entry-point so the subprocess uses our code
    engine_mod.run_scheduler_process = custom_run_scheduler_process

    # Apply parent-process patches (TokenizerManager)
    apply_parent_patches()

    eng = Engine(
        model_path=MODEL,
        tp_size=1,
        dtype="float16",
        disable_cuda_graph=True,
        attention_backend="torch_native",
    )
    yield eng
    eng.shutdown()


@pytest.fixture(scope="module")
def sglang_result(engine, prompt_ids):
    """Generate a single sequence with entropy via sglang, using token IDs."""
    output = engine.generate(
        input_ids=prompt_ids,
        sampling_params={
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        return_logprob=True,
    )
    return output


@pytest.fixture(scope="module")
def hf_model():
    """Load HF model for reference forward pass."""
    from transformers import AutoModelForCausalLM

    # Use sdpa attention — same as sglang's torch_native (both use F.scaled_dot_product_attention)
    # float16 to match sglang's explicit dtype="float16" above
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).cuda()
    model.eval()

    yield model

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntropyVsHF:
    """Cross-validate sglang per-token entropy against HuggingFace forward pass."""

    def test_entropy_matches_hf(self, sglang_result, hf_model, prompt_ids):
        """Token-level entropy from sglang must match HF forward-pass entropy.

        Steps:
        1. Extract generated token IDs and per-token entropy from sglang output.
        2. Concatenate prompt token IDs with generated tokens.
        3. Run HF forward pass on the full sequence.
        4. For each generated token position, compute entropy from HF logits.
        5. Compare sglang vs HF entropy with tolerance.
        """
        # ---- 1. Extract sglang results ------------------------------------
        meta = sglang_result["meta_info"]
        assert "output_token_entropy" in meta, f"Entropy missing from sglang output. Keys: {list(meta.keys())}"
        sglang_entropy_list = meta["output_token_entropy"]
        assert len(sglang_entropy_list) > 0, "sglang produced no entropy values"

        # Get the generated token IDs from logprob metadata
        output_token_logprobs = meta["output_token_logprobs"]
        # Each entry in output_token_logprobs is (logprob, token_id)
        generated_token_ids = [int(entry[1]) for entry in output_token_logprobs]

        num_generated = len(generated_token_ids)
        assert len(sglang_entropy_list) == num_generated, (
            f"Entropy count {len(sglang_entropy_list)} != generated token count {num_generated}"
        )

        # ---- 2. Build full sequence from shared prompt_ids ----------------
        prompt_len = len(prompt_ids)
        full_ids = torch.tensor([prompt_ids + generated_token_ids], dtype=torch.long, device="cuda")

        # ---- 3. HF forward pass ------------------------------------------
        with torch.no_grad():
            hf_logits = hf_model(full_ids).logits  # [1, seq_len, vocab_size]

        # ---- 4. Compute HF entropy at generated token positions -----------
        # For generated token i (0-indexed), the logits that predicted it
        # are at position (prompt_len + i - 1), because logits[t] predicts
        # token[t+1].
        hf_entropy_list = []
        for i in range(num_generated):
            logit_pos = prompt_len + i - 1
            logits_at_pos = hf_logits[0, logit_pos, :]  # [vocab_size]
            ent = _compute_entropy_from_logits(logits_at_pos.unsqueeze(0))
            hf_entropy_list.append(ent.item())

        # ---- 5. Compare --------------------------------------------------
        sglang_ent = torch.tensor(sglang_entropy_list, dtype=torch.float32)
        hf_ent = torch.tensor(hf_entropy_list, dtype=torch.float32)

        # Both sides use float16 + SDPA attention (torch_native / sdpa).
        # Remaining differences come from fused RMSNorm kernels and minor
        # implementation details.  Tight tolerance to catch real bugs.
        if num_generated > 1:
            torch.testing.assert_close(
                sglang_ent[1:],
                hf_ent[1:],
                rtol=1e-2,
                atol=1e-2,
                msg=lambda m: (
                    f"Decode-phase entropy mismatch (tokens 1..{num_generated - 1}):\n"
                    f"sglang: {sglang_ent[1:].tolist()}\n"
                    f"hf:     {hf_ent[1:].tolist()}\n{m}"
                ),
            )

        # First token (prefill/extend path) may still differ slightly more
        torch.testing.assert_close(
            sglang_ent[:1],
            hf_ent[:1],
            rtol=5e-2,
            atol=5e-2,
            msg=lambda m: (
                f"Prefill token entropy mismatch:\n"
                f"sglang: {sglang_ent[0].item():.6f}\n"
                f"hf:     {hf_ent[0].item():.6f}\n{m}"
            ),
        )

    def test_entropy_direction(self, sglang_result, hf_model, prompt_ids):
        """Sanity check: sglang and HF entropies should be correlated.

        Even if absolute values differ slightly due to numerical kernels,
        relative ordering (which tokens are high/low entropy) should agree.
        """
        meta = sglang_result["meta_info"]

        sglang_entropy_list = meta["output_token_entropy"]
        output_token_logprobs = meta["output_token_logprobs"]
        generated_token_ids = [int(entry[1]) for entry in output_token_logprobs]
        num_generated = len(generated_token_ids)

        if num_generated < 4:
            pytest.skip("Too few generated tokens for correlation check")

        prompt_len = len(prompt_ids)
        full_ids = torch.tensor([prompt_ids + generated_token_ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            hf_logits = hf_model(full_ids).logits

        hf_entropy_list = []
        for i in range(num_generated):
            logit_pos = prompt_len + i - 1
            logits_at_pos = hf_logits[0, logit_pos, :].unsqueeze(0)
            hf_entropy_list.append(_compute_entropy_from_logits(logits_at_pos).item())

        # Pearson correlation should be high (> 0.95)
        sglang_ent = torch.tensor(sglang_entropy_list, dtype=torch.float64)
        hf_ent = torch.tensor(hf_entropy_list, dtype=torch.float64)

        # Center both
        sg_centered = sglang_ent - sglang_ent.mean()
        hf_centered = hf_ent - hf_ent.mean()

        numer = (sg_centered * hf_centered).sum()
        denom = sg_centered.norm() * hf_centered.norm()
        if denom < 1e-12:
            # All values basically identical -> correlation is perfect
            corr = 1.0
        else:
            corr = (numer / denom).item()

        assert corr > 0.95, (
            f"Pearson correlation between sglang and HF entropy is only {corr:.4f}; "
            f"expected > 0.95.\n"
            f"sglang: {sglang_ent.tolist()}\n"
            f"hf:     {hf_ent.tolist()}"
        )

    def test_max_abs_diff(self, sglang_result, hf_model, prompt_ids):
        """Max absolute difference between sglang and HF entropy should be small."""
        meta = sglang_result["meta_info"]

        sglang_entropy_list = meta["output_token_entropy"]
        output_token_logprobs = meta["output_token_logprobs"]
        generated_token_ids = [int(entry[1]) for entry in output_token_logprobs]
        num_generated = len(generated_token_ids)

        prompt_len = len(prompt_ids)
        full_ids = torch.tensor([prompt_ids + generated_token_ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            hf_logits = hf_model(full_ids).logits

        hf_entropy_list = []
        for i in range(num_generated):
            logit_pos = prompt_len + i - 1
            logits_at_pos = hf_logits[0, logit_pos, :].unsqueeze(0)
            hf_entropy_list.append(_compute_entropy_from_logits(logits_at_pos).item())

        sglang_ent = torch.tensor(sglang_entropy_list, dtype=torch.float32)
        hf_ent = torch.tensor(hf_entropy_list, dtype=torch.float32)
        abs_diff = (sglang_ent - hf_ent).abs()

        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()

        # Log for debugging
        print(f"\n[entropy vs HF] num_tokens={num_generated}")
        print(f"[entropy vs HF] max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}")
        print("[entropy vs HF] dtype=float16, attention: sglang=torch_native, hf=sdpa")

        # Hard ceiling: with matching dtype + attention, max diff should be
        # small.  0.1 nats is generous for residual RMSNorm kernel differences.
        assert max_diff < 0.1, (
            f"Max absolute entropy difference {max_diff:.6f} exceeds 0.1 nats.\nabs_diff per token: {abs_diff.tolist()}"
        )

    def test_logprob_matches_hf(self, sglang_result, hf_model, prompt_ids):
        """Per-token logprob: sglang vs HF.

        Logprobs pick a single point from the distribution, so they're more
        sensitive to kernel drift than entropy (which sums over the full vocab).
        We expect larger diffs here than in test_entropy_matches_hf — the key
        insight is that distributions match (entropy close) even when individual
        token probabilities shift due to fused-kernel rounding.
        """
        meta = sglang_result["meta_info"]
        output_token_logprobs = meta["output_token_logprobs"]
        assert len(output_token_logprobs) > 0, "sglang produced no logprob entries"

        sglang_logprobs = [float(entry[0]) for entry in output_token_logprobs]
        generated_token_ids = [int(entry[1]) for entry in output_token_logprobs]
        num_generated = len(generated_token_ids)

        prompt_len = len(prompt_ids)
        full_ids = torch.tensor([prompt_ids + generated_token_ids], dtype=torch.long, device="cuda")

        with torch.no_grad():
            hf_logits = hf_model(full_ids).logits

        hf_logprobs = []
        for i in range(num_generated):
            logit_pos = prompt_len + i - 1
            log_probs = torch.nn.functional.log_softmax(hf_logits[0, logit_pos, :].float(), dim=-1)
            hf_logprobs.append(log_probs[generated_token_ids[i]].item())

        sg_lp = torch.tensor(sglang_logprobs, dtype=torch.float32)
        hf_lp = torch.tensor(hf_logprobs, dtype=torch.float32)
        abs_diff = (sg_lp - hf_lp).abs()

        print(f"\n[logprob vs HF] num_tokens={num_generated}")
        print(f"[logprob vs HF] max_abs_diff={abs_diff.max():.6f}, mean={abs_diff.mean():.6f}")

        # Loose sanity bound — logprobs are sensitive to kernel-level rounding
        # across 24 layers, unlike entropy which averages over the full vocab.
        # Observed ~0.5 max diff from fused RMSNorm vs PyTorch RMSNorm.
        assert abs_diff.max().item() < 1.0, f"Logprob diff {abs_diff.max():.4f} is catastrophically large"

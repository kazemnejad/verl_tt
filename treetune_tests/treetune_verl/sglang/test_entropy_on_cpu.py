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

import math

import torch

from treetune_verl.sglang.entropy import compute_entropy


class TestComputeEntropy:
    def test_full_vocab_uniform(self):
        """Uniform logits over 4 tokens -> entropy = log(4)."""
        logits = torch.zeros(1, 4)
        result = compute_entropy(logits)
        torch.testing.assert_close(result, torch.tensor([math.log(4)]), atol=1e-5, rtol=1e-5)

    def test_full_vocab_deterministic(self):
        """One-hot-ish logits -> entropy ~ 0."""
        logits = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
        result = compute_entropy(logits)
        assert result.item() < 1e-5

    def test_top_k_uniform(self):
        """top_k=2 from uniform-4 logits -> renormalized over 2 -> entropy = log(2)."""
        logits = torch.zeros(1, 4)
        result = compute_entropy(logits, top_k=2)
        torch.testing.assert_close(result, torch.tensor([math.log(2)]), atol=1e-5, rtol=1e-5)

    def test_top_k_picks_largest(self):
        """top_k=2 picks the two largest logits and computes entropy over them."""
        logits = torch.tensor([[10.0, 1.0, 5.0, 3.0]])
        result = compute_entropy(logits, top_k=2)

        # Should pick 10.0 and 5.0
        top_vals = torch.tensor([[10.0, 5.0]])
        log_probs = torch.nn.functional.log_softmax(top_vals, dim=-1)
        expected = -(torch.exp(log_probs) * log_probs).sum(dim=-1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_batch(self):
        """Batch of 3 sequences with vocab 100 -> output shape (3,), all non-negative."""
        logits = torch.randn(3, 100)
        result = compute_entropy(logits)
        assert result.shape == (3,)
        assert (result >= 0).all()

    def test_bounded_by_log_vocab(self):
        """Entropy should be <= log(vocab_size) + epsilon."""
        logits = torch.randn(5, 50)
        result = compute_entropy(logits)
        assert (result <= math.log(50) + 1e-5).all()

    def test_top_k_bounded_by_log_k(self):
        """Top-k entropy should be <= log(k) + epsilon."""
        logits = torch.randn(5, 50)
        result = compute_entropy(logits, top_k=10)
        assert (result <= math.log(10) + 1e-5).all()

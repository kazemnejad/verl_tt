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

"""Entropy-aware SGLang server and replica subclasses.

Spec: treetune_specs/2026-02-02-sglang-entropy-extraction.md
"""

import os
from typing import Any, Optional

import ray
import torch
from sglang.srt.managers.io_struct import GenerateReqInput

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.sglang_rollout.async_sglang_server import SGLangHttpServer, SGLangReplica


class EntropyTokenOutput(TokenOutput):
    """TokenOutput with per-token entropy of the policy distribution."""

    entropy: Optional[list[float]] = None


class EntropySGLangHttpServer(SGLangHttpServer):
    """SGLangHttpServer that injects entropy patches and returns per-token entropy."""

    async def launch_server(self, master_address=None, master_port=None):
        import sglang.srt.entrypoints.engine

        from treetune_verl.sglang.entropy import apply_parent_patches, custom_run_scheduler_process

        entropy_top_k = getattr(self.config, "entropy_top_k", 0)
        os.environ["TREETUNE_ENTROPY_TOP_K"] = str(entropy_top_k if entropy_top_k else 0)
        sglang.srt.entrypoints.engine.run_scheduler_process = custom_run_scheduler_process
        apply_parent_patches()

        await super().launch_server(master_address, master_port)

    # SYNC WARNING: async_sglang_server.py:SGLangHttpServer.generate() — see agent-docs/sync-warnings.md
    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> EntropyTokenOutput:
        """Generate sequence with entropy extraction."""
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)

        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            max_new_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        max_new_tokens = max(0, min(max_new_tokens, max_possible_tokens))

        assert max_new_tokens <= max_possible_tokens, (
            f"max_new_tokens {max_new_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
        }

        if self.config.enable_rollout_routing_replay:
            request.update({"return_routed_experts": True})

        generate_request = GenerateReqInput(**request)

        output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
        if return_logprob:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None

        # Extract entropy from meta_info (populated by our patches)
        entropy = None
        if return_logprob:
            entropy = output["meta_info"].get("output_token_entropy", None)

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = output.get("meta_info", {}).get("routed_experts", None)
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

        return EntropyTokenOutput(
            token_ids=token_ids, log_probs=log_probs, entropy=entropy, routed_experts=routed_experts
        )


class EntropySGLangReplica(SGLangReplica):
    """SGLang replica that uses EntropySGLangHttpServer."""

    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, is_reward_model=False):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(EntropySGLangHttpServer)

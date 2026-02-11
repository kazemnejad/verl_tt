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

"""Entropy-aware streaming generation: worker + manager composition."""

from __future__ import annotations

import ray
from ray.util.queue import Queue

from treetune_verl.agent_loop.entropy_manager import _load_entropy_replica
from treetune_verl.agent_loop.entropy_worker import EntropyAgentLoopWorker
from treetune_verl.generation.runner import GenerationLoopManager
from treetune_verl.generation.worker import StreamingAgentLoopWorkerMixin


class StreamingEntropyWorker(StreamingAgentLoopWorkerMixin, EntropyAgentLoopWorker):
    """Streaming generation with per-token entropy extraction."""

    pass


class EntropyGenerationLoopManager(GenerationLoopManager):
    """GenerationLoopManager wired with entropy-aware replica + worker."""

    def __init__(self, config, queue: Queue):
        self.rollout_replica_class = _load_entropy_replica()
        self.agent_loop_workers_class = ray.remote(StreamingEntropyWorker)
        super().__init__(config, queue)

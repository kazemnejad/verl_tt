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

"""Unit tests for CollectorAgentLoopWorker.

Tests the worker that pushes completed trajectories to ResultsQueue
as they complete, enabling incremental batch saving.

TDD: Tests written before implementation.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


class TestCollectorAgentLoopWorker:
    """Tests for CollectorAgentLoopWorker class."""

    def test_worker_inherits_from_agent_loop_worker(self):
        """CollectorAgentLoopWorker must inherit from AgentLoopWorker."""
        from treetune_verl.generation.worker import CollectorAgentLoopWorker
        from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

        assert issubclass(CollectorAgentLoopWorker, AgentLoopWorker)

    def test_worker_stores_queue_handle(self):
        """Constructor stores results_queue handle for later use."""
        from treetune_verl.generation.worker import CollectorAgentLoopWorker

        # Mock parent __init__ to avoid needing full config
        mock_queue = MagicMock()
        mock_config = MagicMock()
        mock_servers = []

        with patch(
            "treetune_verl.generation.worker.AgentLoopWorker.__init__",
            return_value=None,
        ):
            worker = CollectorAgentLoopWorker(
                config=mock_config,
                server_handles=mock_servers,
                results_queue=mock_queue,
            )

        assert worker._results_queue is mock_queue

    def test_push_to_queue_calls_remote_put(self):
        """_push_to_queue(idx, result) calls queue.put.remote(idx, result)."""
        from treetune_verl.generation.worker import CollectorAgentLoopWorker

        # Create worker with mocked parent
        mock_queue = MagicMock()
        mock_config = MagicMock()

        with patch(
            "treetune_verl.generation.worker.AgentLoopWorker.__init__",
            return_value=None,
        ):
            worker = CollectorAgentLoopWorker(
                config=mock_config,
                server_handles=[],
                results_queue=mock_queue,
            )

        # Call _push_to_queue
        test_idx = 42
        test_result = {"data": "trajectory"}
        worker._push_to_queue(test_idx, test_result)

        # Verify queue.put.remote was called correctly
        mock_queue.put.remote.assert_called_once_with(test_idx, test_result)

    def test_worker_calls_push_on_trajectory_completion(self):
        """Worker calls _push_to_queue after _run_agent_loop completes."""
        from treetune_verl.generation.worker import CollectorAgentLoopWorker

        async def run_test():
            mock_queue = MagicMock()
            mock_config = MagicMock()

            # Create mock parent output
            mock_output = MagicMock()
            mock_output.model_dump.return_value = {"mocked": "output"}

            with (
                patch(
                    "treetune_verl.generation.worker.AgentLoopWorker.__init__",
                    return_value=None,
                ),
                patch(
                    "treetune_verl.generation.worker.AgentLoopWorker._run_agent_loop",
                    new_callable=AsyncMock,
                    return_value=mock_output,
                ),
            ):
                worker = CollectorAgentLoopWorker(
                    config=mock_config,
                    server_handles=[],
                    results_queue=mock_queue,
                )

                # Mock _push_to_queue to verify it gets called
                worker._push_to_queue = MagicMock()

                # Call _run_agent_loop with sample params
                sampling_params = {"temperature": 0.7}
                trajectory = {
                    "step": 0,
                    "sample_index": 5,
                    "rollout_n": 0,
                    "validate": False,
                }

                await worker._run_agent_loop(
                    sampling_params,
                    trajectory,
                    agent_name="default",
                    index=5,  # The batch index for push
                )

                # Verify _push_to_queue was called with correct args
                worker._push_to_queue.assert_called_once()
                call_args = worker._push_to_queue.call_args
                assert call_args[0][0] == 5  # index
                assert call_args[0][1] is mock_output  # the result

        asyncio.run(run_test())

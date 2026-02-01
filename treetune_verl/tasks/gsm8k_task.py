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

from treetune_verl.tasks.task import Task
from verl.utils.reward_score.gsm8k import extract_solution


class GSM8KTask(Task):
    """Task subclass for GSM8K with answer extraction and reward metadata."""

    def _make_map_fn(self):
        base_fn = super()._make_map_fn()

        def _transform(row: dict, index: int) -> dict:
            result = base_fn(row, index)

            # Extract numeric answer from "#### <number>" pattern
            raw_answer = row["answer"]
            extracted = extract_solution(raw_answer)

            result["reward_model"] = {
                "style": "rule",
                "ground_truth": extracted,
            }

            # Add raw fields to extra_info
            result["extra_info"]["answer"] = raw_answer

            return result

        return _transform

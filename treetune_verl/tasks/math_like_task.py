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

from treetune_verl.tasks.task import Task, _resolve


def _normalize_answer(answer: str) -> str:
    """Wrap answer in LaTeX math mode if not already enclosed."""
    in_brackets = "\\[" in answer and "\\]" in answer
    in_parens = "\\(" in answer and "\\)" in answer
    in_dollars = answer.startswith("$") and answer.endswith("$")
    if not (in_brackets or in_parens or in_dollars):
        answer = f"\\[{answer}\\]"
    return answer


class MathLikeTask(Task):
    """Generic math-problem task with answer normalization and optional filtering."""

    def _make_map_fn(self):
        base_fn = super()._make_map_fn()
        problem_key = _resolve(self.config.get("problem_key", "problem"))
        answer_key = _resolve(self.config.get("answer_key", "answer"))

        def _transform(row: dict, index: int) -> dict:
            result = base_fn(row, index)

            raw_answer = row[answer_key]
            if not isinstance(raw_answer, str):
                raw_answer = str(raw_answer)

            normalized = _normalize_answer(raw_answer)

            result["reward_model"] = {
                "style": "rule",
                "ground_truth": normalized,
            }

            result["extra_info"]["answer"] = str(row[answer_key])
            result["extra_info"]["question"] = row[problem_key]

            return result

        return _transform

    def build_dataset(self):
        ds = super().build_dataset()

        filter_empty = _resolve(self.config.get("filter_empty_answers", True))
        pass_rate_min = _resolve(self.config.get("pass_rate_min", 0.0))
        pass_rate_max = _resolve(self.config.get("pass_rate_max", 1.0))
        pass_rate_key = _resolve(self.config.get("pass_rate_key", "pass_rate"))

        filter_fns = []
        if filter_empty:
            filter_fns.append(
                lambda x: x["reward_model"]["ground_truth"] is not None and len(x["reward_model"]["ground_truth"]) > 0
            )
        if pass_rate_min > 0 or pass_rate_max < 1:
            filter_fns.append(lambda x: x[pass_rate_key] >= pass_rate_min and x[pass_rate_key] <= pass_rate_max)

        if filter_fns:
            ds = ds.filter(lambda x: all(fn(x) for fn in filter_fns))

        return ds

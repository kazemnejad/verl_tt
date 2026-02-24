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

SYSTEM_PROMPT = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)

_STDIN_USER_TEMPLATE = """\
### Question:
{question_content}

### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)"""

_FUNCTIONAL_USER_TEMPLATE = """\
### Question:
{question_content}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```

### Answer: (use the provided format with backticks)"""


class LiveCodeBenchTask(Task):
    """LiveCodeBench coding task with conditional prompt formatting."""

    def _make_map_fn(self):
        base_fn = super()._make_map_fn()

        def _transform(row: dict, index: int) -> dict:
            result = base_fn(row, index)

            # Build conditional prompt
            starter_code = row.get("starter_code", "")
            if starter_code:
                user_content = _FUNCTIONAL_USER_TEMPLATE.format(
                    question_content=row["question_content"],
                    starter_code=starter_code,
                )
            else:
                user_content = _STDIN_USER_TEMPLATE.format(
                    question_content=row["question_content"],
                )

            result["prompt"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # Extra info
            result["extra_info"]["question_id"] = row["question_id"]
            result["extra_info"]["platform"] = row["platform"]
            result["extra_info"]["difficulty"] = row["difficulty"]
            result["extra_info"]["starter_code"] = starter_code

            # Reward model
            result["reward_model"] = {
                "style": "code_exec",
                "test_cases": row["public_test_cases"],
            }

            return result

        return _transform

    def build_dataset(self):
        raw_ds = self._load_from_hf()

        # Difficulty filtering
        difficulty_filter = _resolve(self.config.get("difficulty_filter", None))
        if difficulty_filter is not None:
            allowed = set(difficulty_filter)
            raw_ds = raw_ds.filter(lambda x: x["difficulty"] in allowed)

        map_fn = self._make_map_fn()
        remove_cols = raw_ds.column_names
        ds = raw_ds.map(map_fn, with_indices=True, remove_columns=remove_cols)
        return ds

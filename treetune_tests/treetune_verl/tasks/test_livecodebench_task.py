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

from omegaconf import OmegaConf


def _make_task(overrides: dict | None = None):
    """Create a LiveCodeBenchTask with minimal config."""
    base = {
        "loading_params": {"args": ["dummy"], "kwargs": {"split": "test"}},
        "prompt_template": "{question_content}",
        "data_source": "livecodebench",
    }
    if overrides:
        base.update(overrides)
    cfg = OmegaConf.create(base)

    from treetune_verl.tasks.livecodebench_task import LiveCodeBenchTask

    return LiveCodeBenchTask(cfg, cache_dir="/tmp/test_cache")


# ---------------------------------------------------------------------------
# Shared mock rows
# ---------------------------------------------------------------------------

STDIN_ROW = {
    "question_content": "Given N, print N+1",
    "starter_code": "",
    "question_id": "abc301_a",
    "platform": "atcoder",
    "difficulty": "easy",
    "public_test_cases": '[{"input": "5", "output": "6", "testtype": "stdin"}]',
    "question_title": "Test",
    "contest_id": "abc301",
    "contest_date": "2023-05-13",
}

FUNCTIONAL_ROW = {
    "question_content": "Return the sum of two numbers.",
    "starter_code": "class Solution:\n    def twoSum(self, a: int, b: int) -> int:",
    "question_id": "lc_123",
    "platform": "leetcode",
    "difficulty": "easy",
    "public_test_cases": '[{"input": "1 2", "output": "3", "testtype": "functional"}]',
    "question_title": "Two Sum",
    "contest_id": "weekly_400",
    "contest_date": "2024-01-01",
}


class TestLiveCodeBenchMapFn:
    """Tests for LiveCodeBenchTask._make_map_fn() transform."""

    def test_basic_output_shape(self):
        """Map fn produces prompt, data_source, reward_model, extra_info."""
        task = _make_task()
        fn = task._make_map_fn()
        result = fn(STDIN_ROW, 0)

        assert "prompt" in result
        assert "data_source" in result
        assert "reward_model" in result
        assert "extra_info" in result
        assert result["data_source"] == "livecodebench"

    def test_stdin_prompt_format(self):
        """Stdin problem: user message contains stdin instruction and placeholder."""
        task = _make_task()
        fn = task._make_map_fn()
        result = fn(STDIN_ROW, 0)

        user_msg = result["prompt"][1]["content"]
        assert "Read the inputs from stdin" in user_msg
        assert "# YOUR CODE HERE" in user_msg

    def test_functional_prompt_format(self):
        """Functional problem: user message contains starter code and instruction."""
        task = _make_task()
        fn = task._make_map_fn()
        result = fn(FUNCTIONAL_ROW, 0)

        user_msg = result["prompt"][1]["content"]
        assert FUNCTIONAL_ROW["starter_code"] in user_msg
        assert "use the following starter code" in user_msg.lower()

    def test_system_prompt(self):
        """First message is system role with expert programmer text."""
        task = _make_task()
        fn = task._make_map_fn()
        result = fn(STDIN_ROW, 0)

        assert result["prompt"][0]["role"] == "system"
        assert "expert Python programmer" in result["prompt"][0]["content"]

    def test_extra_info_fields(self):
        """extra_info contains question_id, platform, difficulty, starter_code, index."""
        task = _make_task()
        fn = task._make_map_fn()
        result = fn(STDIN_ROW, 5)

        info = result["extra_info"]
        assert info["question_id"] == "abc301_a"
        assert info["platform"] == "atcoder"
        assert info["difficulty"] == "easy"
        assert info["starter_code"] == ""
        assert info["index"] == 5

    def test_reward_model_structure(self):
        """reward_model has style="code_exec" and test_cases key."""
        task = _make_task()
        fn = task._make_map_fn()
        result = fn(STDIN_ROW, 0)

        assert result["reward_model"]["style"] == "code_exec"
        assert "test_cases" in result["reward_model"]

    def test_question_content_in_prompt(self):
        """The actual question_content text appears in the user message."""
        task = _make_task()
        fn = task._make_map_fn()

        result_stdin = fn(STDIN_ROW, 0)
        assert STDIN_ROW["question_content"] in result_stdin["prompt"][1]["content"]

        result_func = fn(FUNCTIONAL_ROW, 0)
        assert FUNCTIONAL_ROW["question_content"] in result_func["prompt"][1]["content"]

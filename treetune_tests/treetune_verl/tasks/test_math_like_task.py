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
    """Create a MathLikeTask with minimal config."""
    base = {
        "loading_params": {"args": ["dummy"], "kwargs": {"split": "train"}},
        "prompt_template": "{problem}",
        "data_source": "test_math",
        "problem_key": "problem",
        "answer_key": "answer",
    }
    if overrides:
        base.update(overrides)
    cfg = OmegaConf.create(base)

    from treetune_verl.tasks.math_like_task import MathLikeTask

    return MathLikeTask(cfg, cache_dir="/tmp/test_cache")


class TestMathLikeMapFn:
    """Tests for MathLikeTask._make_map_fn() transform."""

    def test_basic_output_shape(self):
        """Map fn produces prompt, data_source, reward_model, extra_info."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "What is 2+2?", "answer": "4"}
        result = fn(row, 0)

        assert "prompt" in result
        assert "data_source" in result
        assert "reward_model" in result
        assert "extra_info" in result
        assert result["data_source"] == "test_math"

    def test_reward_model_structure(self):
        """reward_model has style=rule and ground_truth."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "Solve x", "answer": "42"}
        result = fn(row, 0)

        assert result["reward_model"]["style"] == "rule"
        assert result["reward_model"]["ground_truth"] is not None

    def test_answer_wrapped_in_math_mode(self):
        """Plain answer gets wrapped in \\[...\\]."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "Q", "answer": "42"}
        result = fn(row, 0)

        assert result["reward_model"]["ground_truth"] == "\\[42\\]"

    def test_answer_already_in_brackets_not_double_wrapped(self):
        """Answer already in \\[...\\] is left alone."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "Q", "answer": "\\[42\\]"}
        result = fn(row, 0)

        assert result["reward_model"]["ground_truth"] == "\\[42\\]"

    def test_answer_in_parentheses_not_wrapped(self):
        """Answer in \\(...\\) is left alone."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "Q", "answer": "\\(42\\)"}
        result = fn(row, 0)

        assert result["reward_model"]["ground_truth"] == "\\(42\\)"

    def test_answer_in_dollars_not_wrapped(self):
        """Answer in $...$ is left alone."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "Q", "answer": "$42$"}
        result = fn(row, 0)

        assert result["reward_model"]["ground_truth"] == "$42$"

    def test_non_string_answer_converted(self):
        """Numeric answer is converted to string then wrapped."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "Q", "answer": 42}
        result = fn(row, 0)

        assert result["reward_model"]["ground_truth"] == "\\[42\\]"

    def test_extra_info_has_answer_and_question(self):
        """extra_info carries original answer, question, and index."""
        task = _make_task()
        fn = task._make_map_fn()
        row = {"problem": "What is 2+2?", "answer": "4"}
        result = fn(row, 7)

        assert result["extra_info"]["index"] == 7
        assert result["extra_info"]["answer"] == "4"
        assert result["extra_info"]["question"] == "What is 2+2?"

    def test_custom_problem_answer_keys(self):
        """Respects custom problem_key and answer_key."""
        task = _make_task(
            {
                "problem_key": "question_text",
                "answer_key": "solution",
                "prompt_template": "{question_text}",
            }
        )
        fn = task._make_map_fn()
        row = {"question_text": "Solve", "solution": "7"}
        result = fn(row, 0)

        assert result["extra_info"]["question"] == "Solve"
        assert result["extra_info"]["answer"] == "7"
        assert result["reward_model"]["ground_truth"] == "\\[7\\]"

    def test_prompt_uses_base_template(self):
        """Prompt is constructed by the base Task class template logic."""
        task = _make_task({"system_prompt": "Be helpful."})
        fn = task._make_map_fn()
        row = {"problem": "What is 1+1?", "answer": "2"}
        result = fn(row, 0)

        assert result["prompt"][0]["role"] == "system"
        assert result["prompt"][0]["content"] == "Be helpful."
        assert result["prompt"][1]["role"] == "user"
        assert result["prompt"][1]["content"] == "What is 1+1?"

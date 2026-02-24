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

from collections import Counter

import pytest
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


class TestLiveCodeBenchFiltering:
    """Tests for LiveCodeBenchTask.build_dataset() difficulty filtering."""

    _ROWS = [
        {
            "question_content": "Easy problem",
            "starter_code": "",
            "question_id": "q1",
            "platform": "atcoder",
            "difficulty": "easy",
            "public_test_cases": '[{"input": "1", "output": "2", "testtype": "stdin"}]',
            "question_title": "Easy",
            "contest_id": "c1",
            "contest_date": "2024-01-01",
        },
        {
            "question_content": "Medium problem",
            "starter_code": "",
            "question_id": "q2",
            "platform": "atcoder",
            "difficulty": "medium",
            "public_test_cases": '[{"input": "1", "output": "2", "testtype": "stdin"}]',
            "question_title": "Medium",
            "contest_id": "c2",
            "contest_date": "2024-02-01",
        },
        {
            "question_content": "Hard problem",
            "starter_code": "",
            "question_id": "q3",
            "platform": "atcoder",
            "difficulty": "hard",
            "public_test_cases": '[{"input": "1", "output": "2", "testtype": "stdin"}]',
            "question_title": "Hard",
            "contest_id": "c3",
            "contest_date": "2024-03-01",
        },
    ]

    def _build_from_rows(self, rows: list[dict], overrides: dict | None = None):
        """Build a dataset from in-memory rows through LiveCodeBenchTask."""
        from unittest.mock import patch

        from datasets import Dataset as HFDataset

        task = _make_task(overrides)
        mock_ds = HFDataset.from_list(rows)
        with patch.object(task, "_load_from_hf", return_value=mock_ds):
            return task.build_dataset()

    def test_no_filtering_by_default(self):
        """All difficulties pass through when no filter is set."""
        ds = self._build_from_rows(self._ROWS)
        assert len(ds) == 3

    def test_difficulty_filter(self):
        """difficulty_filter: [easy, medium] drops hard problems."""
        ds = self._build_from_rows(
            self._ROWS, {"difficulty_filter": ["easy", "medium"]}
        )
        assert len(ds) == 2
        difficulties = {row["extra_info"]["difficulty"] for row in ds}
        assert difficulties == {"easy", "medium"}

    def test_difficulty_filter_single(self):
        """difficulty_filter: [hard] keeps only hard problems."""
        ds = self._build_from_rows(self._ROWS, {"difficulty_filter": ["hard"]})
        assert len(ds) == 1
        assert ds[0]["extra_info"]["difficulty"] == "hard"


# ---------------------------------------------------------------------------
# Integration test — hits HuggingFace
# ---------------------------------------------------------------------------


def _load_lcb_from_hf():
    """Download real LCB JSONL files from HuggingFace and load as HF Dataset.

    Workaround: ``datasets>=4.0`` dropped loading-script support, but the
    ``livecodebench/code_generation_lite`` repo still ships one.  We fetch the
    raw JSONL shards via ``hf_hub_download`` instead.
    """
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset as _ld

    repo = "livecodebench/code_generation_lite"
    shards = [f"test{s}.jsonl" for s in ["", "2", "3", "4", "5", "6"]]
    paths = [hf_hub_download(repo_id=repo, filename=f, repo_type="dataset") for f in shards]
    return _ld("json", data_files=paths, split="train")


@pytest.mark.slow
def test_livecodebench_builds_from_hf():
    """Integration: load real LCB data, build dataset, check schema."""
    from unittest.mock import patch

    cfg = OmegaConf.create(
        {
            "loading_params": {
                "args": ["livecodebench/code_generation_lite"],
                "kwargs": {"split": "test", "trust_remote_code": True},
            },
            "prompt_template": "{question_content}",
            "data_source": "livecodebench",
        }
    )

    from treetune_verl.tasks.livecodebench_task import LiveCodeBenchTask

    task = LiveCodeBenchTask(cfg, cache_dir="/tmp/test_lcb_cache")

    # Patch _load_from_hf to work around datasets>=4.0 dropping loading scripts.
    # Data still comes from HuggingFace Hub — only the load mechanism differs.
    with patch.object(task, "_load_from_hf", side_effect=lambda: _load_lcb_from_hf()):
        ds = task.build_dataset()

    # --- basic length ---
    assert len(ds) > 0, "Dataset should not be empty"

    # --- schema of first row ---
    row = ds[0]
    for key in ("prompt", "data_source", "extra_info", "reward_model"):
        assert key in row, f"Missing key: {key}"

    # --- prompt structure: [system, user] ---
    prompt = row["prompt"]
    assert isinstance(prompt, list) and len(prompt) == 2, (
        f"Expected 2-message prompt, got {len(prompt)}"
    )
    assert prompt[0]["role"] == "system"

    # --- extra_info fields ---
    info = row["extra_info"]
    for field in ("question_id", "platform", "difficulty", "starter_code", "index"):
        assert field in info, f"extra_info missing: {field}"

    # --- reward_model fields ---
    rm = row["reward_model"]
    assert "style" in rm, "reward_model missing 'style'"
    assert "test_cases" in rm, "reward_model missing 'test_cases'"

    # --- stats (for visibility) ---
    platforms = Counter(r["extra_info"]["platform"] for r in ds)
    difficulties = Counter(r["extra_info"]["difficulty"] for r in ds)
    print(f"\n--- LiveCodeBench dataset stats ---")
    print(f"Total rows: {len(ds)}")
    print(f"Platforms:    {dict(platforms)}")
    print(f"Difficulties: {dict(difficulties)}")

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

from unittest.mock import patch

import datasets
from omegaconf import OmegaConf

from treetune_verl.tasks.gsm8k_task import GSM8KTask


def _make_gsm8k_task(overrides=None):
    base = {
        "loading_params": {"args": ["openai/gsm8k"], "kwargs": {"name": "main", "split": "train"}},
        "prompt_template": "{question}",
        "data_source": "openai/gsm8k",
    }
    if overrides:
        base.update(overrides)
    return GSM8KTask(OmegaConf.create(base))


def _fake_gsm8k_dataset():
    return datasets.Dataset.from_dict(
        {
            "question": [
                "What is 2+2?",
                "If you have 3 apples and get 5 more, how many do you have?",
            ],
            "answer": [
                "The answer is 2+2=4. #### 4",
                "You have 3+5=8 apples. #### 8",
            ],
        }
    )


def test_gsm8k_map_fn_extracts_answer():
    """Row with answer='some text #### 42' -> reward_model.ground_truth == '42'."""
    task = _make_gsm8k_task()
    map_fn = task._make_map_fn()
    row = {"question": "Q?", "answer": "some text #### 42"}
    result = map_fn(row, 0)
    assert result["reward_model"]["ground_truth"] == "42"


def test_gsm8k_map_fn_sets_data_source():
    """Output has data_source == 'openai/gsm8k'."""
    task = _make_gsm8k_task()
    map_fn = task._make_map_fn()
    row = {"question": "Q?", "answer": "#### 10"}
    result = map_fn(row, 0)
    assert result["data_source"] == "openai/gsm8k"


def test_gsm8k_map_fn_builds_prompt():
    """question field used in prompt template."""
    task = _make_gsm8k_task()
    map_fn = task._make_map_fn()
    row = {"question": "What is 2+2?", "answer": "#### 4"}
    result = map_fn(row, 0)
    assert result["prompt"] == [{"role": "user", "content": "What is 2+2?"}]


def test_gsm8k_map_fn_preserves_extra_info():
    """extra_info contains index and raw answer."""
    task = _make_gsm8k_task()
    map_fn = task._make_map_fn()
    row = {"question": "Q?", "answer": "some text #### 42"}
    result = map_fn(row, 7)
    assert result["extra_info"]["index"] == 7
    assert result["extra_info"]["answer"] == "some text #### 42"


def test_gsm8k_map_fn_reward_model_style():
    """reward_model.style == 'rule'."""
    task = _make_gsm8k_task()
    map_fn = task._make_map_fn()
    row = {"question": "Q?", "answer": "#### 5"}
    result = map_fn(row, 0)
    assert result["reward_model"]["style"] == "rule"


def test_gsm8k_build_dataset():
    """Full pipeline: monkey-patch _load_from_hf, verify output."""
    task = _make_gsm8k_task()
    fake_ds = _fake_gsm8k_dataset()
    with patch.object(task, "_load_from_hf", return_value=fake_ds):
        result = task.build_dataset()
    assert len(result) == 2
    assert "prompt" in result.column_names
    assert "data_source" in result.column_names
    assert "reward_model" in result.column_names
    assert "extra_info" in result.column_names
    # Verify extracted answers
    assert result[0]["reward_model"]["ground_truth"] == "4"
    assert result[1]["reward_model"]["ground_truth"] == "8"

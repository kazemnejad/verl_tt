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

import os
from unittest.mock import patch

import datasets
import pytest
from omegaconf import OmegaConf

from treetune_verl.tasks.task import Task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(config_dict, cache_dir=None):
    """Create a Task with an OmegaConf config."""
    return Task(OmegaConf.create(config_dict), cache_dir=cache_dir)


def _fake_hf_dataset():
    """Small in-memory HF dataset for testing."""
    return datasets.Dataset.from_dict(
        {
            "question": ["What is 2+2?", "What is 3+3?"],
            "answer": ["4", "6"],
            "difficulty": ["easy", "medium"],
        }
    )


# ---------------------------------------------------------------------------
# _make_map_fn — template mode
# ---------------------------------------------------------------------------


def test_template_mode_basic():
    """prompt_template='{question}' -> prompt=[{role:user, content:...}], data_source set."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    row = {"question": "What is 2+2?", "answer": "4"}
    result = map_fn(row, 0)
    assert result["prompt"] == [{"role": "user", "content": "What is 2+2?"}]
    assert result["data_source"] == "test_ds"


def test_template_mode_with_system_prompt():
    """System prompt prepended as first message."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "system_prompt": "You are a math tutor.",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    row = {"question": "What is 2+2?"}
    result = map_fn(row, 0)
    assert len(result["prompt"]) == 2
    assert result["prompt"][0] == {"role": "system", "content": "You are a math tutor."}
    assert result["prompt"][1] == {"role": "user", "content": "What is 2+2?"}


def test_template_mode_multi_field():
    """prompt_template='Subject: {subject}\\n{question}' uses multiple columns."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "Subject: {subject}\n{question}",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    row = {"subject": "Math", "question": "What is 2+2?"}
    result = map_fn(row, 0)
    assert result["prompt"] == [{"role": "user", "content": "Subject: Math\nWhat is 2+2?"}]


# ---------------------------------------------------------------------------
# _make_map_fn — chat_messages mode
# ---------------------------------------------------------------------------


def test_chat_messages_mode():
    """chat_messages mode passes through the message list."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_format": "chat_messages",
            "chat_messages_field": "conversations",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    messages = [{"role": "user", "content": "hi"}]
    row = {"conversations": messages}
    result = map_fn(row, 0)
    assert result["prompt"] == messages


def test_chat_messages_mode_prepends_system():
    """Adds system prompt if first message isn't system."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_format": "chat_messages",
            "chat_messages_field": "conversations",
            "system_prompt": "Be helpful.",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    messages = [{"role": "user", "content": "hi"}]
    row = {"conversations": messages}
    result = map_fn(row, 0)
    assert len(result["prompt"]) == 2
    assert result["prompt"][0] == {"role": "system", "content": "Be helpful."}
    assert result["prompt"][1] == {"role": "user", "content": "hi"}


def test_chat_messages_mode_no_duplicate_system():
    """Doesn't duplicate if already has system message."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_format": "chat_messages",
            "chat_messages_field": "conversations",
            "system_prompt": "Be helpful.",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    messages = [
        {"role": "system", "content": "Existing system."},
        {"role": "user", "content": "hi"},
    ]
    row = {"conversations": messages}
    result = map_fn(row, 0)
    assert len(result["prompt"]) == 2
    assert result["prompt"][0]["role"] == "system"
    assert result["prompt"][0]["content"] == "Existing system."


# ---------------------------------------------------------------------------
# extra_fields
# ---------------------------------------------------------------------------


def test_extra_fields_in_extra_info():
    """extra_fields=['answer','difficulty'] -> those appear in extra_info."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
            "extra_fields": ["answer", "difficulty"],
        }
    )
    map_fn = task._make_map_fn()
    row = {"question": "Q?", "answer": "A", "difficulty": "easy"}
    result = map_fn(row, 0)
    assert result["extra_info"]["answer"] == "A"
    assert result["extra_info"]["difficulty"] == "easy"


def test_extra_info_always_has_index():
    """extra_info.index always present."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    row = {"question": "Q?"}
    result = map_fn(row, 5)
    assert result["extra_info"]["index"] == 5


# ---------------------------------------------------------------------------
# build_dataset
# ---------------------------------------------------------------------------


def test_build_dataset_applies_map():
    """build_dataset loads HF data and applies the map function."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
            "extra_fields": ["answer"],
        }
    )
    fake_ds = _fake_hf_dataset()
    with patch.object(task, "_load_from_hf", return_value=fake_ds):
        result = task.build_dataset()

    assert len(result) == 2
    assert "prompt" in result.column_names
    assert "data_source" in result.column_names
    assert "extra_info" in result.column_names
    # Verify prompt content
    assert result[0]["prompt"] == [{"role": "user", "content": "What is 2+2?"}]


# ---------------------------------------------------------------------------
# get_parquet_path
# ---------------------------------------------------------------------------


def test_get_parquet_path_creates_file(tmp_path):
    """Uses tmp dir as cache, verify .parquet file created."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
        },
        cache_dir=str(tmp_path),
    )
    fake_ds = _fake_hf_dataset()
    with patch.object(task, "_load_from_hf", return_value=fake_ds):
        path = task.get_parquet_path()

    assert path.endswith(".parquet")
    assert os.path.exists(path)


def test_get_parquet_path_cache_hit(tmp_path):
    """Second call returns same path without rebuilding."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
        },
        cache_dir=str(tmp_path),
    )
    fake_ds = _fake_hf_dataset()
    with patch.object(task, "_load_from_hf", return_value=fake_ds) as mock_load:
        path1 = task.get_parquet_path()
        path2 = task.get_parquet_path()

    assert path1 == path2
    # _load_from_hf should only be called once (first call builds, second is cache hit)
    assert mock_load.call_count == 1


def test_get_parquet_path_respects_env_var(tmp_path, monkeypatch):
    """Set TREETUNE_TASK_CACHE_DIR env var, verify it's used."""
    env_cache = str(tmp_path / "env_cache")
    monkeypatch.setenv("TREETUNE_TASK_CACHE_DIR", env_cache)
    # No cache_dir arg — should use env var
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
        }
    )
    fake_ds = _fake_hf_dataset()
    with patch.object(task, "_load_from_hf", return_value=fake_ds):
        path = task.get_parquet_path()

    assert env_cache in path
    assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_invalid_prompt_format_raises():
    """Unknown prompt_format raises ValueError."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_format": "invalid_mode",
            "data_source": "test_ds",
        }
    )
    map_fn = task._make_map_fn()
    with pytest.raises(ValueError, match="Unknown prompt_format"):
        map_fn({"question": "Q?"}, 0)


def test_extra_fields_missing_column_skipped():
    """Missing extra_field column is silently omitted from extra_info."""
    task = _make_task(
        {
            "loading_params": {"args": ["dummy"]},
            "prompt_template": "{question}",
            "data_source": "test_ds",
            "extra_fields": ["answer", "nonexistent"],
        }
    )
    map_fn = task._make_map_fn()
    row = {"question": "Q?", "answer": "A"}
    result = map_fn(row, 0)
    assert result["extra_info"]["answer"] == "A"
    assert "nonexistent" not in result["extra_info"]

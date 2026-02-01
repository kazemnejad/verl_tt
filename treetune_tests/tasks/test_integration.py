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
import textwrap
from unittest.mock import patch

import datasets
from omegaconf import OmegaConf

from treetune_verl.tasks.task import (
    Task,
    _resolve_task_cls,
    get_dataset_paths,
    resolve_tasks_into_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_hf_dataset():
    return datasets.Dataset.from_dict({"question": ["Q1", "Q2"], "answer": ["A1", "A2"]})


def _task_config(**overrides):
    base = {
        "loading_params": {"args": ["dummy"]},
        "prompt_template": "{question}",
        "data_source": "test_ds",
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# _resolve_task_cls
# ---------------------------------------------------------------------------


def test_resolve_default_task():
    """Config without custom_cls -> returns base Task."""
    cfg = _task_config()
    cls = _resolve_task_cls(cfg)
    assert cls is Task


def test_resolve_custom_cls(tmp_path):
    """Config with custom_cls.path + custom_cls.name -> loads that class."""
    src = textwrap.dedent("""\
        from treetune_verl.tasks.task import Task
        class DummyTask(Task):
            pass
    """)
    filepath = tmp_path / "dummy_task.py"
    filepath.write_text(src)
    cfg = _task_config(custom_cls={"path": str(filepath), "name": "DummyTask"})
    cls = _resolve_task_cls(cfg)
    assert cls.__name__ == "DummyTask"


def test_resolve_custom_cls_default_name(tmp_path):
    """Config with custom_cls.path but no name -> loads class named 'Task'."""
    src = textwrap.dedent("""\
        from treetune_verl.tasks.task import Task as _Base
        class Task(_Base):
            custom_marker = True
    """)
    filepath = tmp_path / "custom_task.py"
    filepath.write_text(src)
    cfg = _task_config(custom_cls={"path": str(filepath)})
    cls = _resolve_task_cls(cfg)
    assert cls.__name__ == "Task"
    assert getattr(cls, "custom_marker", False) is True


# ---------------------------------------------------------------------------
# get_dataset_paths
# ---------------------------------------------------------------------------


def test_get_dataset_paths_returns_parquet_list(tmp_path):
    """Pass list of 2 task configs, verify returns list of 2 parquet paths."""
    configs = [_task_config(), _task_config(data_source="ds2")]
    with patch.object(Task, "_load_from_hf", return_value=_fake_hf_dataset()):
        paths = get_dataset_paths(configs, cache_dir=str(tmp_path))
    assert len(paths) == 2
    for p in paths:
        assert p.endswith(".parquet")
        assert os.path.exists(p)


def test_get_dataset_paths_with_custom_cls(tmp_path):
    """One task uses custom_cls, verify it uses the right class."""
    src = textwrap.dedent("""\
        from treetune_verl.tasks.task import Task
        class MarkerTask(Task):
            built = False
            def build_dataset(self):
                MarkerTask.built = True
                return super().build_dataset()
    """)
    filepath = tmp_path / "marker_task.py"
    filepath.write_text(src)
    configs = [
        _task_config(custom_cls={"path": str(filepath), "name": "MarkerTask"}),
    ]
    with patch.object(Task, "_load_from_hf", return_value=_fake_hf_dataset()):
        paths = get_dataset_paths(configs, cache_dir=str(tmp_path))
    assert len(paths) == 1
    assert os.path.exists(paths[0])


# ---------------------------------------------------------------------------
# resolve_tasks_into_config
# ---------------------------------------------------------------------------


def test_resolve_patches_train_files(tmp_path):
    """Config with train_tasks -> config.data.train_files gets set."""
    config = OmegaConf.create(
        {
            "train_tasks": [
                {
                    "loading_params": {"args": ["dummy"]},
                    "prompt_template": "{question}",
                    "data_source": "test_ds",
                }
            ],
            "data": {},
        }
    )
    with patch.object(Task, "_load_from_hf", return_value=_fake_hf_dataset()):
        resolve_tasks_into_config(config, cache_dir=str(tmp_path))
    assert "train_files" in config.data
    assert len(config.data.train_files) == 1
    assert config.data.train_files[0].endswith(".parquet")


def test_resolve_patches_val_files(tmp_path):
    """Config with val_tasks -> config.data.val_files gets set."""
    config = OmegaConf.create(
        {
            "val_tasks": [
                {
                    "loading_params": {"args": ["dummy"]},
                    "prompt_template": "{question}",
                    "data_source": "test_ds",
                }
            ],
            "data": {},
        }
    )
    with patch.object(Task, "_load_from_hf", return_value=_fake_hf_dataset()):
        resolve_tasks_into_config(config, cache_dir=str(tmp_path))
    assert "val_files" in config.data
    assert len(config.data.val_files) == 1


def test_resolve_noop_without_tasks():
    """Config without train_tasks/val_tasks -> config unchanged."""
    config = OmegaConf.create({"data": {"existing_key": "value"}})
    original = OmegaConf.to_container(config, resolve=True)
    resolve_tasks_into_config(config)
    after = OmegaConf.to_container(config, resolve=True)
    assert original == after


def test_resolve_only_touches_file_paths(tmp_path):
    """Verify no other config keys are modified."""
    config = OmegaConf.create(
        {
            "train_tasks": [
                {
                    "loading_params": {"args": ["dummy"]},
                    "prompt_template": "{question}",
                    "data_source": "test_ds",
                }
            ],
            "data": {"train_batch_size": 64},
            "actor_rollout_ref": {"model": {"path": "test_model"}},
        }
    )
    with patch.object(Task, "_load_from_hf", return_value=_fake_hf_dataset()):
        resolve_tasks_into_config(config, cache_dir=str(tmp_path))
    # Other keys untouched
    assert config.data.train_batch_size == 64
    assert config.actor_rollout_ref.model.path == "test_model"

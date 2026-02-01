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

import copy
import logging
import os
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf

from treetune_verl.tasks.cache import compute_cache_key
from verl.utils.import_utils import load_module

logger = logging.getLogger(__name__)


def _resolve(val):
    """Convert OmegaConf nodes to plain Python; pass-through primitives."""
    if isinstance(val, DictConfig | ListConfig):
        return OmegaConf.to_container(val, resolve=True)
    return val


DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/treetune/tasks")


class Task:
    """Config-driven data loading layer that produces cached parquet files.

    Loads a HuggingFace dataset, constructs prompts from config, caches the
    processed result as parquet, and returns the cached path for consumption
    by verl's ``RLHFDataset``.
    """

    def __init__(self, config: DictConfig | dict, cache_dir: str | None = None):
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        # Cache dir resolution: arg -> env -> default
        if cache_dir is not None:
            self.cache_dir = cache_dir
        elif os.environ.get("TREETUNE_TASK_CACHE_DIR"):
            self.cache_dir = os.environ["TREETUNE_TASK_CACHE_DIR"]
        else:
            self.cache_dir = DEFAULT_CACHE_DIR

    def _load_from_hf(self):
        """Load the dataset from HuggingFace."""
        args = list(_resolve(self.config.loading_params.args))
        kwargs = _resolve(self.config.loading_params.get("kwargs", {}))
        return load_dataset(*args, **kwargs)

    def _make_map_fn(self):
        """Return a per-row transform function based on config.

        The returned function has signature ``fn(row: dict, index: int) -> dict``.
        """
        prompt_format = _resolve(self.config.get("prompt_format", "template"))
        prompt_template = _resolve(self.config.get("prompt_template", "{}"))
        system_prompt = _resolve(self.config.get("system_prompt", None))
        chat_messages_field = _resolve(self.config.get("chat_messages_field", "messages"))
        data_source = _resolve(self.config.get("data_source", "unknown"))
        extra_fields = _resolve(self.config.get("extra_fields", []))

        def _transform(row: dict, index: int) -> dict:
            result = {}

            # -- Prompt construction --
            if prompt_format == "template":
                content = prompt_template.format(**row)
                messages = [{"role": "user", "content": content}]
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})
            elif prompt_format == "chat_messages":
                messages = copy.deepcopy(row[chat_messages_field])
                if system_prompt and (not messages or messages[0].get("role") != "system"):
                    messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                raise ValueError(f"Unknown prompt_format: {prompt_format}")

            result["prompt"] = messages
            result["data_source"] = data_source

            # -- Extra info --
            extra_info = {"index": index}
            for field in extra_fields:
                if field in row:
                    extra_info[field] = row[field]
            result["extra_info"] = extra_info

            return result

        return _transform

    def build_dataset(self):
        """Load HF dataset and apply the map function. Returns an HF Dataset."""
        ds = self._load_from_hf()
        map_fn = self._make_map_fn()
        # Remove all original columns so only mapped columns remain
        remove_cols = ds.column_names
        ds = ds.map(
            map_fn,
            with_indices=True,
            remove_columns=remove_cols,
        )
        return ds

    def get_parquet_path(self) -> str:
        """Return path to the cached parquet file, building if needed."""
        cache_key = compute_cache_key(self.__class__, self.config)
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = cache_dir / f"{cache_key}.parquet"

        if parquet_path.exists():
            logger.info("Cache hit: %s", parquet_path)
            return str(parquet_path)

        logger.info("Cache miss — building dataset for %s", cache_key)
        ds = self.build_dataset()
        ds.to_parquet(str(parquet_path))
        logger.info("Saved parquet: %s", parquet_path)
        return str(parquet_path)


# ---------------------------------------------------------------------------
# Module-level integration utilities
# ---------------------------------------------------------------------------


def _resolve_task_cls(task_config: DictConfig) -> type:
    """Resolve a Task (sub)class from config.

    If ``task_config.custom_cls.path`` is set, load the class using verl's
    ``load_module``. Falls back to the base :class:`Task`.

    The module is registered in ``sys.modules`` so that ``inspect.getfile``
    works for cache-key hashing.
    """
    custom_cls = task_config.get("custom_cls", None)
    if custom_cls is not None and custom_cls.get("path", None) is not None:
        cls_name = custom_cls.get("name", "Task")
        # Use a stable module name so the module is registered in sys.modules
        # (needed by inspect.getfile / cache key). Reuse if already loaded.
        import hashlib

        path_hash = hashlib.sha256(os.path.abspath(custom_cls.path).encode()).hexdigest()[:16]
        module_name = f"treetune_task_{path_hash}"
        if module_name in sys.modules:
            mod = sys.modules[module_name]
        else:
            mod = load_module(custom_cls.path, module_name=module_name)
        return getattr(mod, cls_name)
    return Task


def get_dataset_paths(
    task_configs: list[DictConfig],
    cache_dir: str | None = None,
) -> list[str]:
    """Build/cache parquet files for each task config.

    Returns a list of absolute paths to ``.parquet`` files.
    """
    paths = []
    for cfg in task_configs:
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        cls = _resolve_task_cls(cfg)
        task = cls(cfg, cache_dir=cache_dir)
        paths.append(task.get_parquet_path())
    return paths


def resolve_tasks_into_config(config: DictConfig, cache_dir: str | None = None):
    """Resolve ``train_tasks`` / ``val_tasks`` into parquet file lists.

    Patches ``config.data.train_files`` and ``config.data.val_files``
    in-place. No other keys are touched.
    """
    train_tasks = config.get("train_tasks", None)
    if train_tasks is not None:
        paths = get_dataset_paths(list(train_tasks), cache_dir=cache_dir)
        OmegaConf.update(config, "data.train_files", paths)

    val_tasks = config.get("val_tasks", None)
    if val_tasks is not None:
        paths = get_dataset_paths(list(val_tasks), cache_dir=cache_dir)
        OmegaConf.update(config, "data.val_files", paths)


def run_with_tasks(config: DictConfig):
    """Resolve tasks and delegate to verl's ``run_ppo``."""
    resolve_tasks_into_config(config)
    from verl.trainer.main_ppo import run_ppo

    run_ppo(config)

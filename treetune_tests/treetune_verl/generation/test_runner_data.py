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

"""Unit tests for GenerationRunner data loading logic.

Tests the data loading methods added to GenerationRunner:
- _load_data(): Load parquet files, apply max_samples, store DataFrame
- Task system integration: if tasks configured, resolves to data.files

TDD approach: tests written first, then implementation.
"""

from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from treetune_verl.generation.runner import GenerationRunner


@pytest.fixture
def make_parquet_file(tmp_path):
    """Factory fixture to create parquet files with given data."""

    def _make(filename: str, data: dict) -> str:
        """Create a parquet file from dict of columns and return path."""
        table = pa.table(data)
        filepath = tmp_path / filename
        pq.write_table(table, filepath)
        return str(filepath)

    return _make


@pytest.fixture
def runner_with_config(tmp_path):
    """Create a GenerationRunner with minimal config for data loading tests."""

    def _make(config_dict: dict):
        """Create runner with given config values."""
        runner = object.__new__(GenerationRunner)
        runner.config = OmegaConf.create(config_dict)
        runner.output_dir = tmp_path
        # Initialize state variables that _load_data will set
        runner.dataframe = None
        runner.prompt_key = None
        runner.total_samples = 0
        return runner

    return _make


class TestLoadSingleParquet:
    """Tests for loading a single parquet file."""

    def test_load_single_parquet(self, runner_with_config, make_parquet_file):
        """Loads single parquet file and stores as DataFrame.

        Verifies:
        - dataframe is populated
        - total_samples matches row count
        - prompt_key is extracted from config
        """
        # Create test parquet file
        parquet_path = make_parquet_file(
            "test.parquet",
            {
                "prompt": ["Hello", "World", "Test"],
                "metadata": ["a", "b", "c"],
            },
        )

        # Create runner with config pointing to the file
        runner = runner_with_config(
            {
                "data": {
                    "files": [parquet_path],
                    "prompt_key": "prompt",
                    "max_samples": None,
                },
                "tasks": None,
            }
        )

        # Load data
        runner._load_data()

        # Verify
        assert runner.dataframe is not None
        assert isinstance(runner.dataframe, pd.DataFrame)
        assert len(runner.dataframe) == 3
        assert runner.total_samples == 3
        assert runner.prompt_key == "prompt"
        assert list(runner.dataframe["prompt"]) == ["Hello", "World", "Test"]


class TestLoadMultipleParquet:
    """Tests for loading and concatenating multiple parquet files."""

    def test_load_multiple_parquet(self, runner_with_config, make_parquet_file):
        """Concatenates multiple parquet files into single DataFrame.

        Verifies:
        - Multiple files are loaded
        - Rows are concatenated in order
        - total_samples is sum of all files
        """
        # Create multiple test parquet files
        file1 = make_parquet_file(
            "file1.parquet",
            {
                "prompt": ["A", "B"],
                "value": [1, 2],
            },
        )
        file2 = make_parquet_file(
            "file2.parquet",
            {
                "prompt": ["C", "D", "E"],
                "value": [3, 4, 5],
            },
        )

        # Create runner with config pointing to both files
        runner = runner_with_config(
            {
                "data": {
                    "files": [file1, file2],
                    "prompt_key": "prompt",
                    "max_samples": None,
                },
                "tasks": None,
            }
        )

        # Load data
        runner._load_data()

        # Verify concatenation
        assert runner.dataframe is not None
        assert len(runner.dataframe) == 5
        assert runner.total_samples == 5
        # Check order: file1 rows first, then file2
        assert list(runner.dataframe["prompt"]) == ["A", "B", "C", "D", "E"]
        assert list(runner.dataframe["value"]) == [1, 2, 3, 4, 5]


class TestLoadAppliesMaxSamples:
    """Tests for max_samples limiting."""

    def test_load_applies_max_samples(self, runner_with_config, make_parquet_file):
        """Limits DataFrame to max_samples if set.

        Verifies:
        - Only first max_samples rows are kept
        - total_samples reflects the limited count
        """
        # Create parquet with more rows than max_samples
        parquet_path = make_parquet_file(
            "large.parquet",
            {
                "prompt": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
                "idx": list(range(10)),
            },
        )

        # Create runner with max_samples = 3
        runner = runner_with_config(
            {
                "data": {
                    "files": [parquet_path],
                    "prompt_key": "prompt",
                    "max_samples": 3,
                },
                "tasks": None,
            }
        )

        # Load data
        runner._load_data()

        # Verify limited to 3 samples
        assert len(runner.dataframe) == 3
        assert runner.total_samples == 3
        assert list(runner.dataframe["prompt"]) == ["P1", "P2", "P3"]

    def test_load_max_samples_null_keeps_all(self, runner_with_config, make_parquet_file):
        """When max_samples is None, keeps all samples."""
        parquet_path = make_parquet_file(
            "data.parquet",
            {
                "prompt": ["A", "B", "C", "D", "E"],
            },
        )

        runner = runner_with_config(
            {
                "data": {
                    "files": [parquet_path],
                    "prompt_key": "prompt",
                    "max_samples": None,
                },
                "tasks": None,
            }
        )

        runner._load_data()

        assert len(runner.dataframe) == 5
        assert runner.total_samples == 5


class TestLoadExtractsPromptKey:
    """Tests for prompt_key extraction from config."""

    def test_load_extracts_prompt_key(self, runner_with_config, make_parquet_file):
        """Uses configured prompt_key column name.

        Verifies:
        - runner.prompt_key matches config value
        - Works with custom column names
        """
        # Create parquet with custom prompt column name
        parquet_path = make_parquet_file(
            "custom.parquet",
            {
                "my_custom_prompt": ["Question 1", "Question 2"],
                "other_col": ["x", "y"],
            },
        )

        runner = runner_with_config(
            {
                "data": {
                    "files": [parquet_path],
                    "prompt_key": "my_custom_prompt",
                    "max_samples": None,
                },
                "tasks": None,
            }
        )

        runner._load_data()

        # Verify prompt_key is set correctly
        assert runner.prompt_key == "my_custom_prompt"
        # Verify the column exists and has expected values
        assert "my_custom_prompt" in runner.dataframe.columns
        assert list(runner.dataframe["my_custom_prompt"]) == ["Question 1", "Question 2"]


class TestTaskSystemResolvesToFiles:
    """Tests for task system integration."""

    def test_task_system_resolves_to_files(self, runner_with_config, make_parquet_file):
        """If tasks configured, resolves to data.files via get_dataset_paths().

        Verifies:
        - get_dataset_paths is called with task configs
        - Resolved paths are used for loading
        - data.files is updated with resolved paths
        """
        # Create the actual parquet file that the mock will return
        parquet_path = make_parquet_file(
            "task_data.parquet",
            {
                "prompt": ["Task prompt 1", "Task prompt 2", "Task prompt 3"],
                "label": [0, 1, 0],
            },
        )

        # Task config that would be passed to get_dataset_paths
        task_config = {
            "loading_params": {
                "args": ["test/dataset"],
                "kwargs": {"split": "train"},
            },
            "prompt_template": "{question}",
        }

        runner = runner_with_config(
            {
                "data": {
                    "files": None,  # Not set when using tasks
                    "prompt_key": "prompt",
                    "max_samples": None,
                },
                "tasks": [task_config],
            }
        )

        # Mock get_dataset_paths to return our test parquet path
        with patch("treetune_verl.tasks.get_dataset_paths") as mock_get_paths:
            mock_get_paths.return_value = [parquet_path]

            runner._load_data()

            # Verify get_dataset_paths was called with task configs
            mock_get_paths.assert_called_once()
            call_args = mock_get_paths.call_args
            # First arg should be list of task configs
            assert len(call_args[0][0]) == 1

        # Verify data was loaded from resolved path
        assert runner.dataframe is not None
        assert len(runner.dataframe) == 3
        assert runner.total_samples == 3
        assert list(runner.dataframe["prompt"]) == [
            "Task prompt 1",
            "Task prompt 2",
            "Task prompt 3",
        ]

    def test_task_system_not_used_when_files_provided(self, runner_with_config, make_parquet_file):
        """When tasks is None and files are provided, uses files directly.

        Verifies:
        - get_dataset_paths is NOT called
        - data.files is used directly
        """
        parquet_path = make_parquet_file(
            "direct.parquet",
            {
                "prompt": ["Direct 1", "Direct 2"],
            },
        )

        runner = runner_with_config(
            {
                "data": {
                    "files": [parquet_path],
                    "prompt_key": "prompt",
                    "max_samples": None,
                },
                "tasks": None,  # No tasks
            }
        )

        with patch("treetune_verl.tasks.get_dataset_paths") as mock_get_paths:
            runner._load_data()

            # get_dataset_paths should NOT be called
            mock_get_paths.assert_not_called()

        # Data should be loaded from files directly
        assert runner.dataframe is not None
        assert len(runner.dataframe) == 2

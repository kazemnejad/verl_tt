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

"""Tests for task_definitions.yaml — verify each entry has expected structure."""

from pathlib import Path

from omegaconf import OmegaConf

TASK_DEFS_PATH = Path(__file__).resolve().parents[2] / "treetune_verl" / "tasks" / "config" / "task_definitions.yaml"


def _load_task_defs():
    return OmegaConf.load(TASK_DEFS_PATH)


def test_hmmt_feb_2025_defined():
    cfg = _load_task_defs()
    assert "hmmt_feb_2025" in cfg, "hmmt_feb_2025 missing from task_definitions.yaml"

    t = cfg.hmmt_feb_2025
    # custom_cls should resolve to the math_like_cls anchor
    assert t.custom_cls.path == "treetune_verl/tasks/math_like_task.py"
    assert t.custom_cls.name == "MathLikeTask"

    # loading params
    assert t.loading_params.args[0] == "MathArena/hmmt_feb_2025"
    assert t.loading_params.kwargs.split == "train"

    # column keys
    assert t.problem_key == "problem"
    assert t.answer_key == "answer"

    # data_source
    assert t.data_source == "hmmt_feb_2025"

    # prompt_template should be the shared math prompt
    assert "\\boxed{" in t.prompt_template
    assert "{problem}" in t.prompt_template

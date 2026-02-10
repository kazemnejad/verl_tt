# MathLikeTask Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-ng:executing-plans to implement this plan task-by-task.

**Goal:** Port the old `MathLikeTask` to the new task system as a `Task` subclass in `treetune_verl/tasks/`.

**Architecture:** Override `_make_map_fn()` for answer normalization + reward_model injection (following the GSM8KTask pattern). Override `build_dataset()` to add a post-map filter step for empty answers and pass-rate thresholds. Config keys accessed via `self.config.get(...)` — no dataclass needed.

**Tech Stack:** Python, HuggingFace datasets, OmegaConf, pytest

---

## Context for implementer

### What MathLikeTask does (from old codebase)

A generic math-problem task that:
1. Reads `problem_key` and `answer_key` columns from dataset
2. Auto-wraps answers in `\[...\]` if not already in LaTeX math mode
3. Produces `reward_model: {"style": "rule", "ground_truth": normalized_answer}`
4. Optionally filters out rows with empty answers
5. Optionally filters by pass-rate thresholds (`pass_rate_min`/`pass_rate_max`)
6. Carries original answer + question in `extra_info`

### New system patterns (follow these)

- **Base class:** `treetune_verl/tasks/task.py:Task` — `_make_map_fn()` returns `fn(row, index) -> dict`
- **Subclass pattern:** see `gsm8k_task.py` — calls `super()._make_map_fn()`, wraps it
- **Config access:** `_resolve(self.config.get("key", default))` where `_resolve` converts OmegaConf → plain Python
- **No registry** — use `custom_cls` in YAML to reference the file
- **File placement:** `treetune_verl/tasks/math_like_task.py`
- **Tests:** `treetune_tests/treetune_verl/tasks/test_math_like_task.py`

### Config keys for MathLikeTask

| Key | Default | Description |
|-----|---------|-------------|
| `problem_key` | `"problem"` | Column name for the problem text |
| `answer_key` | `"answer"` | Column name for the ground-truth answer |
| `filter_empty_answers` | `true` | Drop rows where normalized answer is empty |
| `pass_rate_min` | `0.0` | Min pass rate threshold (inclusive) |
| `pass_rate_max` | `1.0` | Max pass rate threshold (inclusive) |
| `pass_rate_key` | `"pass_rate"` | Column name for pass rate values |

Plus all base Task keys (`loading_params`, `prompt_template`, `system_prompt`, `data_source`, `extra_fields`, etc.)

### Real-world config patterns (from old codebase)

Old configs use MathLikeTask heavily — ~13 task instances across 10 YAML files.

**Datasets used:** DeepScaleR, DeepMath (`question`/`final_answer`), OpenMath, AIME 2024/2025, Konkur. All use the same prompt suffix: `'\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.`

**Key pattern:** `prompt_template` in old system used **positional** format (`'{}'.format(problem)`). New system uses **keyword** format (`'{problem}'.format(**row)`). Config migration: `'{}'` → `'{problem}'`.

### Features NOT ported (base Task concerns, not MathLikeTask)

These existed in the old base Task but are absent from the new base Task. Flagged for awareness — **not in scope** for this plan:

| Feature | Old usage | Impact |
|---------|-----------|--------|
| `val_sampling_params` | `n: 32` on all val configs — injected as flat `val_sampling_params.n` column | Needed for multi-sample eval (pass@k). Downstream consumption TBD. |
| `max_samples` | Debug configs (`max_samples: 8`, `30`) | Dataset truncation for dev/debug runs |
| `data_proportion` | Not used in any config found | Subset by fraction |
| `shuffle_before_sampling` | Not used in any config found | Shuffle before truncation |

These should be added to the base `Task` class in a separate task if needed.

---

## Tasks

### Task 1: Write failing tests for MathLikeTask map function

**Files:**
- Create: `treetune_tests/treetune_verl/tasks/test_math_like_task.py`

**Step 1: Write the failing tests**

```python
# Copyright 2025 Individual Contributor: Amirhossein Kazemnejad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...standard Apache 2.0 header...

import pytest
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
        task = _make_task({
            "problem_key": "question_text",
            "answer_key": "solution",
            "prompt_template": "{question_text}",
        })
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
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest treetune_tests/treetune_verl/tasks/test_math_like_task.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (math_like_task doesn't exist yet)

---

### Task 2: Implement MathLikeTask

**Files:**
- Create: `treetune_verl/tasks/math_like_task.py`

**Step 1: Write the implementation**

```python
# Copyright 2025 Individual Contributor: Amirhossein Kazemnejad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...standard Apache 2.0 header...

from treetune_verl.tasks.task import Task, _resolve


def _normalize_answer(answer: str) -> str:
    """Wrap answer in LaTeX math mode if not already enclosed."""
    in_brackets = "\\[" in answer and "\\]" in answer
    in_parens = "\\(" in answer and "\\)" in answer
    in_dollars = answer.startswith("$") and answer.endswith("$")
    if not (in_brackets or in_parens or in_dollars):
        answer = f"\\[{answer}\\]"
    return answer


class MathLikeTask(Task):
    """Generic math-problem task with answer normalization and optional filtering."""

    def _make_map_fn(self):
        base_fn = super()._make_map_fn()
        problem_key = _resolve(self.config.get("problem_key", "problem"))
        answer_key = _resolve(self.config.get("answer_key", "answer"))

        def _transform(row: dict, index: int) -> dict:
            result = base_fn(row, index)

            raw_answer = row[answer_key]
            if not isinstance(raw_answer, str):
                raw_answer = str(raw_answer)

            normalized = _normalize_answer(raw_answer)

            result["reward_model"] = {
                "style": "rule",
                "ground_truth": normalized,
            }

            result["extra_info"]["answer"] = str(row[answer_key])
            result["extra_info"]["question"] = row[problem_key]

            return result

        return _transform

    def build_dataset(self):
        ds = super().build_dataset()

        filter_empty = _resolve(self.config.get("filter_empty_answers", True))
        pass_rate_min = _resolve(self.config.get("pass_rate_min", 0.0))
        pass_rate_max = _resolve(self.config.get("pass_rate_max", 1.0))
        pass_rate_key = _resolve(self.config.get("pass_rate_key", "pass_rate"))

        filter_fns = []
        if filter_empty:
            filter_fns.append(
                lambda x: x["reward_model"]["ground_truth"] is not None
                and len(x["reward_model"]["ground_truth"]) > 0
            )
        if pass_rate_min > 0 or pass_rate_max < 1:
            filter_fns.append(
                lambda x: x[pass_rate_key] >= pass_rate_min
                and x[pass_rate_key] <= pass_rate_max
            )

        if filter_fns:
            ds = ds.filter(lambda x: all(fn(x) for fn in filter_fns))

        return ds
```

**Step 2: Run tests to verify they pass**

Run: `python -m pytest treetune_tests/treetune_verl/tasks/test_math_like_task.py -v`
Expected: All 10 tests PASS

**Step 3: Commit**

```bash
git add treetune_verl/tasks/math_like_task.py treetune_tests/treetune_verl/tasks/test_math_like_task.py
committer "feat(tasks): add MathLikeTask with answer normalization and filtering"
```

---

### Task 3: Write and pass filter tests

**Files:**
- Modify: `treetune_tests/treetune_verl/tasks/test_math_like_task.py`

**Step 1: Add filter tests to the test file**

Append a new test class:

```python
class TestMathLikeFiltering:
    """Tests for MathLikeTask.build_dataset() filtering."""

    def _build_from_rows(self, rows: list[dict], overrides: dict | None = None):
        """Build a dataset from in-memory rows through MathLikeTask."""
        from unittest.mock import patch

        from datasets import Dataset as HFDataset

        task = _make_task(overrides)
        mock_ds = HFDataset.from_list(rows)
        with patch.object(task, "_load_from_hf", return_value=mock_ds):
            return task.build_dataset()

    def test_empty_answer_filtered_by_default(self):
        rows = [
            {"problem": "Q1", "answer": "42"},
            {"problem": "Q2", "answer": ""},
        ]
        ds = self._build_from_rows(rows)
        assert len(ds) == 1
        assert ds[0]["extra_info"]["question"] == "Q1"

    def test_empty_answer_kept_when_disabled(self):
        rows = [
            {"problem": "Q1", "answer": "42"},
            {"problem": "Q2", "answer": ""},
        ]
        ds = self._build_from_rows(rows, {"filter_empty_answers": False})
        assert len(ds) == 2

    def test_pass_rate_filtering(self):
        rows = [
            {"problem": "Q1", "answer": "1", "pass_rate": 0.0},
            {"problem": "Q2", "answer": "2", "pass_rate": 0.5},
            {"problem": "Q3", "answer": "3", "pass_rate": 1.0},
        ]
        ds = self._build_from_rows(rows, {
            "pass_rate_min": 0.3,
            "pass_rate_max": 0.7,
        })
        assert len(ds) == 1
        assert ds[0]["extra_info"]["question"] == "Q2"

    def test_no_filtering_when_all_defaults(self):
        rows = [
            {"problem": "Q1", "answer": "1"},
            {"problem": "Q2", "answer": "2"},
        ]
        ds = self._build_from_rows(rows)
        assert len(ds) == 2
```

**Step 2: Run tests**

Run: `python -m pytest treetune_tests/treetune_verl/tasks/test_math_like_task.py -v`
Expected: All tests PASS (including new filter tests)

**Step 3: Commit**

```bash
git add treetune_tests/treetune_verl/tasks/test_math_like_task.py
committer "test(tasks): add filter tests for MathLikeTask"
```

---

### Task 4: Export MathLikeTask from package + add YAML example

**Files:**
- Modify: `treetune_verl/tasks/__init__.py` — add `MathLikeTask` to exports

**Step 1: Update `__init__.py`**

Add to imports:
```python
from treetune_verl.tasks.math_like_task import MathLikeTask
```
Add `"MathLikeTask"` to `__all__`.

**Step 2: Verify import works**

Run: `python -c "from treetune_verl.tasks import MathLikeTask; print(MathLikeTask)"`
Expected: `<class 'treetune_verl.tasks.math_like_task.MathLikeTask'>`

**Step 3: Commit**

```bash
git add treetune_verl/tasks/__init__.py
committer "feat(tasks): export MathLikeTask from tasks package"
```

---

## YAML usage example (for reference, not a task)

```yaml
train_tasks:
  - custom_cls:
      path: treetune_verl/tasks/math_like_task.py
      name: MathLikeTask
    loading_params:
      args: ["openai/gsm8k"]
      kwargs: { name: main, split: train }
    prompt_template: "{question}"
    problem_key: "question"
    answer_key: "answer"
    data_source: "gsm8k"
    system_prompt: "Solve step by step."
```

Or with `custom_cls` pointing to the package path:
```yaml
train_tasks:
  - custom_cls:
      path: treetune_verl/tasks/math_like_task.py
    loading_params:
      args: ["hendrycks/competition_math"]
      kwargs: { split: train }
    prompt_template: "{problem}"
    problem_key: "problem"
    answer_key: "solution"
    data_source: "math"
    filter_empty_answers: true
    pass_rate_min: 0.0
    pass_rate_max: 0.8
    pass_rate_key: "pass_rate"
```

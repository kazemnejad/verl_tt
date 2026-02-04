# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Generation runner module for trajectory collection without training.

Public exports:
- GenerationRunner: Core orchestrator for generation-only workflows
- ResultsQueue: Ray actor for accumulating generation results
- CollectorAgentLoopWorker: Worker that pushes results to queue on completion

Config is handled via Hydra YAML (generation.yaml), not a Python dataclass.
Checkpoint logic is built into GenerationRunner.
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies and heavy imports at module load."""
    if name == "ResultsQueue":
        from treetune_verl.generation.queue import ResultsQueue

        return ResultsQueue
    elif name == "GenerationRunner":
        from treetune_verl.generation.runner import GenerationRunner

        return GenerationRunner
    elif name == "CollectorAgentLoopWorker":
        from treetune_verl.generation.worker import CollectorAgentLoopWorker

        return CollectorAgentLoopWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GenerationRunner",
    "ResultsQueue",
    "CollectorAgentLoopWorker",
]

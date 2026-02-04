# Copyright 2025 Treetune Authors
# Licensed under the Apache License, Version 2.0
"""Generation runner module for trajectory collection without training."""


def __getattr__(name: str):
    """Lazy import to allow incremental development."""
    if name == "CheckpointManager":
        from treetune_verl.generation.checkpoint import CheckpointManager

        return CheckpointManager
    elif name == "GenerationConfig":
        from treetune_verl.generation.config import GenerationConfig

        return GenerationConfig
    elif name == "ResultsQueue":
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
    "GenerationConfig",
    "ResultsQueue",
    "CollectorAgentLoopWorker",
    "CheckpointManager",
]

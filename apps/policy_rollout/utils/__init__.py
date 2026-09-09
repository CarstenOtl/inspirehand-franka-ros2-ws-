"""Utilities owned by the policy rollout app.

Heavy dependencies such as matplotlib remain behind their individual modules;
importing this package is safe in camera-preflight and recording processes.
"""

from .data_collection import (
    LoadedRollout,
    RolloutArtifact,
    RolloutDataCollector,
    create_rollout_dir,
    load_rollout,
)
from .evaluation import evaluate_rollout
from .plotting import plot_rollout

__all__ = [
    "LoadedRollout",
    "RolloutArtifact",
    "RolloutDataCollector",
    "create_rollout_dir",
    "evaluate_rollout",
    "load_rollout",
    "plot_rollout",
]

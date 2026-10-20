"""Recording schedules (DESIGN.md §5, "Schedules").

Step t means "the state after t optimizer updates". The three schedules nest:
keyframe_steps ⊆ summary_steps ⊆ log_steps, and all three contain 0 and the horizon, so
telemetry, predictions, fingerprints and oracle rows exist at every summary step and
every keyframe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RecordConfig


def _every(steps: int, k: int) -> np.ndarray:
    return np.union1d(np.arange(0, steps + 1, k), [steps]).astype(np.int64)


@dataclass(frozen=True)
class Schedule:
    log_steps: np.ndarray
    summary_steps: np.ndarray
    keyframe_steps: np.ndarray

    @classmethod
    def build(cls, steps: int, rec: RecordConfig) -> "Schedule":
        return cls(
            _every(steps, rec.log_every),
            _every(steps, rec.summary_every),
            _every(steps, rec.keyframe_every),
        )

    def sets(self) -> tuple[set, set, set]:
        return set(self.log_steps.tolist()), set(self.summary_steps.tolist()), set(self.keyframe_steps.tolist())

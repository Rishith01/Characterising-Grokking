"""Learner interface (project.md Section 6): a generator, not a training script."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Protocol

import numpy as np


@dataclass
class Snapshot:
    t: int
    M: np.ndarray  # d x d feature matrix
    metrics: dict = field(default_factory=dict)
    agop: Optional[np.ndarray] = None  # d x d AGOP, when the learner has one distinct from M


class Learner(Protocol):
    def steps(self) -> Iterator[Snapshot]: ...

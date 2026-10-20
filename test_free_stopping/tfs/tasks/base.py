"""Task interface (DESIGN.md §6): what is learned, independent of how it is trained."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import numpy as np
import torch

from ..registry import Registry
from ..seeding import SeedStreams


@dataclass
class TaskData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    d_in: int
    d_out: int
    kind: Literal["classification", "regression"]
    n_classes: Optional[int] = None

    def to(self, device) -> "TaskData":
        return TaskData(
            self.x_train.to(device), self.y_train.to(device),
            self.x_test.to(device), self.y_test.to(device),
            self.d_in, self.d_out, self.kind, self.n_classes,
        )


class Task(Protocol):
    name: str

    def build(self, params: dict, streams: SeedStreams) -> TaskData: ...

    def accuracy(self, pred: torch.Tensor, y: torch.Tensor) -> float: ...

    def train_pred_record(self, pred: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Per-sample record stored every log step: argmax class (int16) for
        classification, per-sample error (float16) for regression."""
        ...


TASKS: Registry = Registry("task")


class ClassificationTask:
    """Shared accuracy / prediction record for argmax-classification tasks."""

    def accuracy(self, pred: torch.Tensor, y: torch.Tensor) -> float:
        return float((pred.argmax(-1) == y).float().mean())

    def train_pred_record(self, pred: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        return pred.argmax(-1).to(torch.int16).cpu().numpy()


def get_task(name: str) -> Task:
    from . import dummy  # noqa: F401  (registration side effect)

    return TASKS.get(name)()

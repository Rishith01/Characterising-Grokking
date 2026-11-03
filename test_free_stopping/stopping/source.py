"""The data-source interface every signal reads through (DESIGN.md §13.1).

A RunSource lists runs and opens one as a RunHandle. Signals never see the handle: the
runner (`stopping.compute`) turns it into a RunContext (time-independent facts) and one
Observation per step, in increasing step order, so a signal cannot look ahead.

Held-out (test) data is reachable only through `RunHandle.heldout()`, which the runner
calls only for signals that set `uses_heldout` (baseline B6).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch

# What a signal can ask for. "models" means replay to the eval steps.
REQUIREMENTS = frozenset({"telemetry", "preds", "summaries", "train_data", "models", "heldout"})


@dataclass(frozen=True)
class TrainData:
    """Training set, on the signal's device. `target` is y in the form the loss takes
    (one-hot for MSE classification), so `loss_fn(model(x), target)` is the training
    loss, a mean over the examples given."""
    x: torch.Tensor
    y: torch.Tensor
    target: torch.Tensor
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    batch: Callable[[int], Optional[torch.Tensor]]  # indices the update from step t uses; None = full batch


@dataclass
class RunContext:
    run_id: str
    app: str
    config: dict  # the run's full config, as stored
    optimizer: str
    lr: float
    lr_eff: float  # lr / (1 - momentum) for SGD with momentum, else lr
    momentum: float
    gamma: float
    decay: str  # "decoupled" | "coupled" | "none"
    loss: str
    kind: str  # "classification" | "regression"
    d_in: int
    d_out: int
    n_train: int
    batch_size: Optional[int]  # None = full batch
    steps: int
    log_steps: np.ndarray
    summary_steps: np.ndarray
    task_params: dict
    device: str = "cpu"
    _train_data: Optional[Callable[[], TrainData]] = field(default=None, repr=False)

    def train_data(self) -> TrainData:
        if self._train_data is None:
            raise RuntimeError("train data not loaded: the signal must declare 'train_data'")
        return self._train_data()


@dataclass
class Observation:
    """Everything a signal may see at one step. Fields a signal did not ask for, or that
    do not exist at this step, are None."""
    step: int
    telemetry: Optional[dict] = None  # this log step's telemetry row
    train_preds: Optional[np.ndarray] = None  # this log step's train predictions (or per-sample error)
    summaries: Optional[dict] = None  # dense summaries, at summary steps
    model: Optional[torch.nn.Module] = None  # the replayed model, at eval steps; read only
    heldout: Optional[dict] = None  # test loss/acc row: only for uses_heldout signals


class RunHandle(ABC):
    """One run, opened with the parts a set of signals needs."""

    @property
    @abstractmethod
    def context(self) -> RunContext: ...

    @abstractmethod
    def telemetry(self) -> pd.DataFrame: ...

    @abstractmethod
    def train_preds(self) -> np.ndarray:
        """[n_log, n_train], row i = log step i."""

    @abstractmethod
    def summaries(self, step: int) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def models(self, steps: Iterable[int]) -> Iterator[tuple[int, torch.nn.Module]]:
        """(step, model) for each step, in increasing order."""

    @abstractmethod
    def heldout(self) -> pd.DataFrame:
        """Test loss/accuracy per log step. Scoring side and B6 only."""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RunSource(ABC):
    @abstractmethod
    def run_ids(self) -> list[str]: ...

    @abstractmethod
    def open(self, run_id: str, needs: Iterable[str]) -> RunHandle: ...

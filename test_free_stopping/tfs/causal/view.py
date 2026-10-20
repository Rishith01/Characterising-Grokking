"""TrainContext: everything time-independent that measurement code may see (DESIGN.md
§7.1). Training data and config only. The test tensors are dropped before this object
exists (DESIGN.md §10).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import RunConfig
from ..optim import RegSpec, reg_spec
from ..seeding import SeedStreams
from ..tasks import Task, get_task


@dataclass(frozen=True)
class TrainContext:
    config: RunConfig
    reg: RegSpec
    task: Task
    x_train: torch.Tensor
    y_train: torch.Tensor
    kind: str
    d_in: int
    d_out: int

    @classmethod
    def from_config(cls, cfg: RunConfig) -> "TrainContext":
        task = get_task(cfg.task.name)
        data = task.build(cfg.task.params, SeedStreams(cfg.seed))
        ctx = cls(cfg, reg_spec(cfg.optim), task, data.x_train, data.y_train,
                  data.kind, data.d_in, data.d_out)
        del data  # the test split never leaves this function
        return ctx

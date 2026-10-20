"""Teacher–student regression (Omnigrok §3; PLAN.md §2).

The teacher is a fixed random MLP (default 5-100-100-5, tanh, standard init) built from
the seed's "teacher" stream. Inputs are x ~ N(0, I) from the "data" stream. A sample
counts as correct when its mean squared error is below θ (Omnigrok uses θ = 0.01).
"""
from __future__ import annotations

import numpy as np
import torch

from ..models import MLP
from ..seeding import SeedStreams
from .base import TASKS, TaskData


@TASKS.register("teacher_student")
class TeacherStudent:
    name = "teacher_student"

    def __init__(self):
        self.theta = 0.01

    def build(self, params: dict, streams: SeedStreams) -> TaskData:
        d_in = int(params.get("d_in", 5))
        d_out = int(params.get("d_out", 5))
        self.theta = float(params.get("theta", 0.01))
        teacher = MLP(d_in, d_out, int(params.get("teacher_depth", 3)), int(params.get("teacher_width", 100)),
                      params.get("teacher_activation", "tanh"), 1.0, generator=streams.torch("teacher"))
        rng = streams.numpy("data")
        n_train, n_test = int(params["n_train"]), int(params.get("n_test", 100))
        x = torch.tensor(rng.normal(size=(n_train + n_test, d_in)), dtype=torch.float32)
        with torch.no_grad():
            y = teacher(x)
        return TaskData(x[:n_train], y[:n_train], x[n_train:], y[n_train:], d_in, d_out, "regression")

    def per_sample_error(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return ((pred - y) ** 2).mean(-1)

    def accuracy(self, pred: torch.Tensor, y: torch.Tensor) -> float:
        return float((self.per_sample_error(pred, y) < self.theta).float().mean())

    def train_pred_record(self, pred: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        return self.per_sample_error(pred, y).to(torch.float16).cpu().numpy()

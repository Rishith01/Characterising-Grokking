"""A tiny synthetic classification task for infrastructure tests (Phase 0, gate G0).

Gaussian class blobs in d_in dimensions. Small enough that a full run takes seconds on
CPU; not used for science.
"""
from __future__ import annotations

import torch

from ..seeding import SeedStreams
from .base import TASKS, ClassificationTask, TaskData


@TASKS.register("dummy")
class DummyBlobs(ClassificationTask):
    name = "dummy"

    def build(self, params: dict, streams: SeedStreams) -> TaskData:
        d_in = params.get("d_in", 8)
        n_classes = params.get("n_classes", 4)
        n_train = params.get("n_train", 256)
        n_test = params.get("n_test", 256)
        noise = params.get("noise", 1.0)

        rng = streams.numpy("data")
        centers = rng.normal(size=(n_classes, d_in))

        def sample(n):
            y = rng.integers(0, n_classes, size=n)
            x = centers[y] + noise * rng.normal(size=(n, d_in))
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

        x_tr, y_tr = sample(n_train)
        x_te, y_te = sample(n_test)
        return TaskData(x_tr, y_tr, x_te, y_te, d_in, n_classes, "classification", n_classes)

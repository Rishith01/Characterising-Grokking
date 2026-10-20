"""Modular arithmetic (PLAN.md §2): f(a, b) = op(a, b) mod p.

Input: one-hot(a) ⊕ one-hot(b) in R^{2p}. Output: p classes. Every pair (a, b) is built,
then a seeded fraction `train_frac` is the training set and the rest is the test set.
For division, b ranges over 1..p-1 and the label is a · b^{-1} mod p.
"""
from __future__ import annotations

import numpy as np
import torch

from ..seeding import SeedStreams
from .base import TASKS, ClassificationTask, TaskData

OPS = {
    "x+y": lambda a, b, p: (a + b) % p,
    "x-y": lambda a, b, p: (a - b) % p,
    "x*y": lambda a, b, p: (a * b) % p,
    "x/y": lambda a, b, p: (a * np.array([pow(int(v), p - 2, p) for v in b])) % p,
    "x^2+xy+y^2": lambda a, b, p: (a * a + a * b + b * b) % p,
}


def modular_pairs(op: str, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; have {list(OPS)}")
    b_range = np.arange(1, p) if op == "x/y" else np.arange(p)
    a, b = np.meshgrid(np.arange(p), b_range, indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    return a, b, OPS[op](a, b, p).astype(np.int64)


@TASKS.register("modular")
class Modular(ClassificationTask):
    name = "modular"

    def build(self, params: dict, streams: SeedStreams) -> TaskData:
        op, p, frac = params["op"], int(params["p"]), float(params["train_frac"])
        a, b, y = modular_pairs(op, p)
        n = len(y)
        perm = streams.numpy("data").permutation(n)
        n_train = int(round(frac * n))
        x = np.zeros((n, 2 * p), dtype=np.float32)
        x[np.arange(n), a] = 1.0
        x[np.arange(n), p + b] = 1.0
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        tr, te = torch.from_numpy(perm[:n_train]), torch.from_numpy(perm[n_train:])
        return TaskData(x[tr], y[tr], x[te], y[te], 2 * p, p, "classification", p)

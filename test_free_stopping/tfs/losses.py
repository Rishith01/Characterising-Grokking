"""Training losses, fixed per application (PLAN.md §2)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_loss(name: str, pred: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    if name == "ce":
        if kind != "classification":
            raise ValueError("cross-entropy needs a classification task")
        return F.cross_entropy(pred, y)
    if name == "mse":
        target = F.one_hot(y, pred.shape[-1]).to(pred.dtype) if kind == "classification" else y
        return torch.mean((pred - target) ** 2)
    raise ValueError(f"unknown loss {name!r}; have ['ce', 'mse']")

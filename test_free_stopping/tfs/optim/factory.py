"""The four optimizers (PLAN.md §1) and how weight decay enters each (RegSpec).

| name         | torch optimizer      | weight decay                          |
|--------------|----------------------|---------------------------------------|
| adamw        | AdamW                | decoupled                              |
| adam_l2      | Adam(weight_decay=γ) | coupled: γw added to the gradient (L2) |
| sgd_momentum | SGD(momentum=μ)      | coupled                                |
| sgd          | SGD                  | coupled                                |

For the coupled optimizers, torch adds γw to p.grad inside step(), so p.grad as seen
before step() is the pure data-loss gradient. Telemetry relies on this.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from ..config import OptimConfig

OPTIMIZERS = ("adamw", "adam_l2", "sgd_momentum", "sgd")


@dataclass(frozen=True)
class RegSpec:
    kind: Literal["decoupled", "coupled", "none"]
    gamma: float

    def to_dict(self) -> dict:
        return asdict(self)


def reg_spec(cfg: OptimConfig) -> RegSpec:
    if cfg.weight_decay == 0.0:
        return RegSpec("none", 0.0)
    return RegSpec("decoupled" if cfg.name == "adamw" else "coupled", cfg.weight_decay)


def build_optimizer(cfg: OptimConfig, params) -> torch.optim.Optimizer:
    params = list(params)
    # foreach/fused kernels are disabled: the single-tensor path is the one whose
    # determinism we verify, and it must not change with tensor count or device.
    common = dict(lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False)
    if cfg.name == "adamw":
        return torch.optim.AdamW(params, betas=tuple(cfg.betas), **common)
    if cfg.name == "adam_l2":
        return torch.optim.Adam(params, betas=tuple(cfg.betas), **common)
    if cfg.name == "sgd_momentum":
        if cfg.momentum <= 0:
            raise ValueError("sgd_momentum needs momentum > 0")
        return torch.optim.SGD(params, momentum=cfg.momentum, **common)
    if cfg.name == "sgd":
        if cfg.momentum != 0:
            raise ValueError("plain sgd must have momentum = 0; use sgd_momentum")
        return torch.optim.SGD(params, **common)
    raise ValueError(f"unknown optimizer {cfg.name!r}; have {list(OPTIMIZERS)}")

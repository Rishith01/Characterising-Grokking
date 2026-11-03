"""Telemetry-only signals: B0 (step), B1 (train loss), N1 (weight norm). METRICS.md §3-4."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from ..signal import Signal, SignalConfig, register
from ..source import Observation
from ._common import LogRate

_W_NORM = re.compile(r"^l(\d+)_w_norm$")


@dataclass(frozen=True)
class StepConfig(SignalConfig):
    """B0: the step itself, so a fixed budget is a threshold rule on it."""
    name: ClassVar[str] = "step"


@register(StepConfig)
class StepSignal(Signal):
    requires = frozenset({"telemetry"})
    cadence = "log"

    def update(self, obs: Observation) -> Optional[dict]:
        return {"value": float(obs.step)}


@dataclass(frozen=True)
class TrainLossConfig(SignalConfig):
    """B1: training loss (and accuracy), for a plateau rule."""
    name: ClassVar[str] = "train_loss"


@register(TrainLossConfig)
class TrainLoss(Signal):
    requires = frozenset({"telemetry"})
    cadence = "log"

    def update(self, obs: Observation) -> Optional[dict]:
        t = obs.telemetry
        return {"value": float(t["train_loss"]), "train_acc": float(t["train_acc"])}


@dataclass(frozen=True)
class WeightNormConfig(SignalConfig):
    """N1: total weight norm sqrt(sum_l ||w_l||^2), per-layer norms, and the windowed
    rate of change of log||w|| (signed and absolute)."""
    name: ClassVar[str] = "weight_norm"
    window: int = 20  # log rows (1000 steps at log_every = 50)
    normalize: str = "none"  # "none" | "lr" | "decay" (see _common.LogRate)


@register(WeightNormConfig)
class WeightNorm(Signal):
    requires = frozenset({"telemetry"})
    cadence = "log"

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.rate = LogRate(self.cfg.window, self.cfg.normalize, ctx.lr_eff, ctx.gamma)
        self.cols: Optional[list[tuple[int, str]]] = None

    def update(self, obs: Observation) -> Optional[dict]:
        t = obs.telemetry
        if self.cols is None:
            self.cols = sorted((int(m.group(1)), k) for k in t for m in [_W_NORM.match(k)] if m)
        norms = [float(t[k]) for _, k in self.cols]
        total = float(np.sqrt(np.sum(np.square(norms))))
        rate = self.rate(obs.step, total)
        return {"value": total, "log_rate": rate, "abs_rate": abs(rate),
                **{f"l{i}": n for (i, _), n in zip(self.cols, norms)}}

"""B4: Label Wave (Yuan et al., ICLR 2024). METRICS.md §4.

Prediction changes between consecutive log steps on the training set. Classification:
the fraction of training examples whose predicted label changed. Teacher-student stores
per-sample RMS error instead of labels; there "regression" picks the analogue:
"flip" = the fraction whose correctness (error < theta) changed, "abs_change" = the mean
|change in error|. The paper stops at the first local minimum of the (smoothed) count; a
plateau rule (direction below) is its causal form.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from ..signal import Signal, SignalConfig, register
from ..source import Observation


@dataclass(frozen=True)
class LabelWaveConfig(SignalConfig):
    name: ClassVar[str] = "label_wave"
    regression: str = "flip"  # "flip" | "abs_change"


@register(LabelWaveConfig)
class LabelWave(Signal):
    requires = frozenset({"preds"})
    cadence = "log"

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.prev: Optional[np.ndarray] = None
        self.theta = float(ctx.task_params.get("theta", 0.01))

    def update(self, obs: Observation) -> Optional[dict]:
        cur = np.asarray(obs.train_preds)
        prev, self.prev = self.prev, cur
        if prev is None:
            return None
        if self.ctx.kind == "classification":
            return {"value": float(np.mean(cur != prev))}
        err, perr = cur.astype(np.float64), prev.astype(np.float64)
        flip = float(np.mean((err < self.theta) != (perr < self.theta)))
        change = float(np.mean(np.abs(err - perr)))
        if self.cfg.regression == "flip":
            return {"value": flip, "abs_change": change}
        if self.cfg.regression == "abs_change":
            return {"value": change, "flip": flip}
        raise ValueError(f"unknown regression mode {self.cfg.regression!r}")

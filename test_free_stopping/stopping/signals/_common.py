"""Helpers shared by several signals: layer aggregation and windowed rates of change."""
from __future__ import annotations

from collections import deque
from typing import Literal, Sequence

import numpy as np

Aggregate = Literal["mean", "min", "max", "first", "last"]
Normalize = Literal["none", "lr", "decay"]


def aggregate(values: Sequence[float], how: str) -> float:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan")
    if how == "mean":
        return float(v.mean())
    if how == "min":
        return float(v.min())
    if how == "max":
        return float(v.max())
    if how == "first":
        return float(v[0])
    if how == "last":
        return float(v[-1])
    raise ValueError(f"unknown aggregate {how!r}")


class LogRate:
    """Rate of change of log(value) over the last `window` rows (causal):
    (log v_t - log v_{t-w}) / time(t-w, t), where time is steps / 1000 ("none"),
    lr_eff * steps ("lr", i.e. per unit of learning-rate time), or lr_eff * gamma * steps
    ("decay", per unit of the weight-decay timescale; NaN at gamma = 0). The choice is
    the learning-rate normalization question of METRICS.md §3 (F2 caveats)."""

    def __init__(self, window: int, normalize: str, lr_eff: float, gamma: float):
        self.hist: deque = deque(maxlen=window + 1)
        if normalize == "none":
            self.unit = 1e-3
        elif normalize == "lr":
            self.unit = lr_eff
        elif normalize == "decay":
            self.unit = lr_eff * gamma if gamma > 0 else float("nan")
        else:
            raise ValueError(f"unknown normalize {normalize!r}")

    def __call__(self, step: int, value: float) -> float:
        self.hist.append((step, value))
        if len(self.hist) < self.hist.maxlen:
            return float("nan")
        (s0, v0), (s1, v1) = self.hist[0], self.hist[-1]
        if not (v0 > 0 and v1 > 0) or s1 == s0:
            return float("nan")
        return float((np.log(v1) - np.log(v0)) / ((s1 - s0) * self.unit))

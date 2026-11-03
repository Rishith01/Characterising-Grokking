"""Rules: turn one signal trace into a stopping step (DESIGN.md §13.4).

Every rule is the same causal template: smooth (EMA or trailing median), test a
condition, and stop at the first step where the condition has held for `patience`
consecutive emitted rows (and step >= min_step).

Two conditions:
- "threshold": smoothed value below/above `threshold`.
- "plateau": no improvement. `direction` says which way is better ("below": smaller is
  better). A row improves if it beats the best so far by more than `threshold` relative
  to |best|; the best is updated only on improvement (as in Keras EarlyStopping).
  Covers B1, B3, B4 and B6.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleConfig:
    column: str = "value"
    mode: Literal["threshold", "plateau"] = "threshold"
    direction: Literal["below", "above"] = "below"
    threshold: float = 0.0
    patience: int = 1
    smoother: Literal["none", "ema", "median"] = "ema"
    smooth_param: float = 0.1  # EMA weight of the new value, or the median window (rows)
    min_step: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StopDecision:
    step: Optional[int]  # None = never stopped
    value: float = float("nan")  # smoothed value at the stop
    part: Optional[int] = None  # which part of a Method stopped first
    reason: str = ""


class _Smoother:
    def __init__(self, kind: str, param: float):
        self.kind, self.param = kind, param
        self.state: Optional[float] = None
        self.window: deque = deque(maxlen=max(1, int(param)) if kind == "median" else 1)

    def __call__(self, x: float) -> float:
        if not np.isfinite(x):
            return self.state if self.state is not None else float("nan")
        if self.kind == "none":
            self.state = x
        elif self.kind == "ema":
            self.state = x if self.state is None else self.param * x + (1 - self.param) * self.state
        elif self.kind == "median":
            self.window.append(x)
            self.state = float(np.median(self.window))
        else:
            raise ValueError(f"unknown smoother {self.kind!r}")
        return self.state


class Rule:
    def __init__(self, cfg: RuleConfig):
        if cfg.patience < 1:
            raise ValueError("patience must be >= 1")
        self.cfg = cfg

    def trace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Row by row: step, raw, smoothed value, condition, run length, stop flag."""
        c = self.cfg
        if c.column not in df.columns:
            raise KeyError(f"column {c.column!r} not in trace; have {list(df.columns)}")
        smooth = _Smoother(c.smoother, c.smooth_param)
        best, run, stopped = None, 0, False
        out = []
        for step, raw in zip(df["step"].to_numpy(), df[c.column].to_numpy(dtype=float)):
            s = smooth(raw)
            if not np.isfinite(s):
                cond = False
            elif c.mode == "threshold":
                cond = s < c.threshold if c.direction == "below" else s > c.threshold
            elif c.mode == "plateau":
                if best is None:
                    best, cond = s, False
                else:
                    margin = c.threshold * abs(best)
                    improved = s < best - margin if c.direction == "below" else s > best + margin
                    if improved:
                        best = s
                    cond = not improved
            else:
                raise ValueError(f"unknown mode {c.mode!r}")
            run = run + 1 if cond else 0
            stop = not stopped and run >= c.patience and step >= c.min_step
            stopped = stopped or stop
            out.append((int(step), raw, s, cond, run, stop))
        return pd.DataFrame(out, columns=["step", "raw", "smooth", "cond", "run", "stop"])

    def decide(self, df: pd.DataFrame) -> StopDecision:
        t = self.trace(df)
        hit = t.index[t["stop"]]
        if len(hit) == 0:
            return StopDecision(None, reason="condition never held long enough")
        r = t.loc[hit[0]]
        return StopDecision(int(r["step"]), float(r["smooth"]),
                            reason=f"{self.cfg.column} {self.cfg.mode} {self.cfg.direction} "
                                   f"{self.cfg.threshold:g} for {self.cfg.patience} rows")

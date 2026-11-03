"""Spectral signals: R1 (effective rank of activations), B5 (HTSR alpha, and correlation
traps). METRICS.md §3 (R1) and §4 (B5)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from ..signal import Signal, SignalConfig, register
from ..source import Observation
from ._common import LogRate, aggregate
from ._linalg import correlation_traps, effective_rank, powerlaw_fit


def _layers(summaries: dict, kind: str) -> list[tuple[int, np.ndarray]]:
    pat = re.compile(rf"^{kind}/l(\d+)$")
    return sorted((int(m.group(1)), v) for k, v in summaries.items() for m in [pat.match(k)] if m)


@dataclass(frozen=True)
class ERankConfig(SignalConfig):
    """R1: effective rank of each hidden layer's training-set activation covariance
    (summaries actspec/l{i}), aggregated over layers, plus its windowed log-rate."""
    name: ClassVar[str] = "erank"
    aggregate: str = "mean"  # over hidden layers: mean | min | max | first | last
    window: int = 4  # summary rows (2000 steps at summary_every = 500)
    normalize: str = "none"


@register(ERankConfig)
class ERank(Signal):
    requires = frozenset({"summaries"})
    cadence = "summary"

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.rate = LogRate(self.cfg.window, self.cfg.normalize, ctx.lr_eff, ctx.gamma)

    def update(self, obs: Observation) -> Optional[dict]:
        ranks = [(i, effective_rank(v)) for i, v in _layers(obs.summaries, "actspec")]
        value = aggregate([r for _, r in ranks], self.cfg.aggregate)
        rate = self.rate(obs.step, value)
        return {"value": value, "log_rate": rate, "abs_rate": abs(rate),
                **{f"l{i}": r for i, r in ranks}}


@dataclass(frozen=True)
class HtsrAlphaConfig(SignalConfig):
    """B5: power-law exponent alpha of each layer's weight ESD (eigenvalues of W^T W =
    squared singular values, summaries sv/l{i}), averaged over layers with at least
    min_eigs eigenvalues (small layers give unstable fits)."""
    name: ClassVar[str] = "htsr_alpha"
    min_eigs: int = 50
    min_tail: int = 10
    aggregate: str = "mean"


@register(HtsrAlphaConfig)
class HtsrAlpha(Signal):
    requires = frozenset({"summaries"})
    cadence = "summary"

    def update(self, obs: Observation) -> Optional[dict]:
        row, alphas, ks = {}, [], []
        for i, sv in _layers(obs.summaries, "sv"):
            if len(sv) < self.cfg.min_eigs:
                continue
            a, _, d, _ = powerlaw_fit(np.asarray(sv, dtype=np.float64) ** 2, self.cfg.min_tail)
            row[f"l{i}"], row[f"ks_l{i}"] = a, d
            alphas.append(a)
            ks.append(d)
        return {"value": aggregate(alphas, self.cfg.aggregate), "ks": aggregate(ks, "mean"), **row}


@dataclass(frozen=True)
class HtsrTrapsConfig(SignalConfig):
    """B5 (second part): correlation traps per layer, from the replayed weights at the
    eval steps. Layers with fewer than min_eigs singular values are skipped."""
    name: ClassVar[str] = "htsr_traps"
    min_eigs: int = 50
    margin: float = 0.1
    seed: int = 0


@register(HtsrTrapsConfig)
class HtsrTraps(Signal):
    requires = frozenset({"models"})
    cadence = "eval"

    def update(self, obs: Observation) -> Optional[dict]:
        rng = np.random.default_rng([self.cfg.seed, obs.step])
        row, total = {}, 0
        for name, p in obs.model.named_parameters():
            if p.ndim != 2 or min(p.shape) < self.cfg.min_eigs:
                continue
            n = correlation_traps(p, rng, self.cfg.margin)
            row[name.rsplit(".", 1)[0].replace("layers.", "l")] = float(n)
            total += n
        return {"value": float(total), **row}

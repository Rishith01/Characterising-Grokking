"""Gradient-statistics signals: B2 (Mahsereci EB on the data gradient) and B3 (gradient
disparity). METRICS.md §4. S1b (EB on the regularized gradient) is B2 plus the gamma*w
term and is not built yet."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
import torch

from ..signal import Signal, SignalConfig, register
from ..source import Observation
from ._linalg import batch_grad, evidence, layer_groups, per_example_moments


def _short(group: str) -> str:
    return group.replace("layers.", "l")


@dataclass(frozen=True)
class EvidenceConfig(SignalConfig):
    """B2: EB = 1 - (B/P) sum_k g_k^2 / s_k^2 over the per-example data-loss gradients of
    the batch the update from step t uses (the minibatch for MNIST; all n examples for
    full-batch tasks, where it is a heuristic: METRICS.md §3, S1b caveats). Stop when
    EB > 0. Reported over all parameters ("value") and per layer."""
    name: ClassVar[str] = "eb"
    chunk: int = 128
    var_floor: float = 0.0


@register(EvidenceConfig)
class Evidence(Signal):
    requires = frozenset({"models", "train_data"})
    cadence = "eval"

    def update(self, obs: Observation) -> Optional[dict]:
        data = self.ctx.train_data()
        idx = data.batch(obs.step)
        x, t = (data.x, data.target) if idx is None else (data.x[idx], data.target[idx])
        s1, s2, n = per_example_moments(obs.model, x, t, data.loss_fn, self.cfg.chunk)
        row, num, p_total = {"batch": float(n)}, 0.0, 0
        for group, names in layer_groups(obs.model).items():
            a = torch.cat([s1[k] for k in names])
            b = torch.cat([s2[k] for k in names])
            eb, p = evidence(a, b, n, self.cfg.var_floor)
            row[_short(group)] = eb
            if p:
                num += (1.0 - eb) * p  # = (B) sum g^2/s^2 over this group's kept coordinates
                p_total += p
        row["value"] = 1.0 - num / p_total if p_total else float("nan")
        row["n_coords"] = float(p_total)
        return row


@dataclass(frozen=True)
class DisparityConfig(SignalConfig):
    """B3: gradient disparity, the expected ||g1 - g2|| between two independent
    minibatch gradients (Forouzesh & Thiran 2021).

    form "summary" (cheap, every summary step): sqrt(2 tr Cov(g)) from the stored
    gradient-noise summaries over 8 fixed probe batches of 256 (gnoise/l{i}, entry 1).
    Degenerate (zero) when n_train <= probe batch size, i.e. teacher-student at N = 50,
    100: emitted as NaN there.
    form "replay" (eval steps): ||g1 - g2|| for two disjoint random batches of size
    `batch` (default: the training batch size, or n/2 for full batch), drawn from a
    fixed per-(seed, step) RNG."""
    name: ClassVar[str] = "grad_disparity"
    form: str = "summary"
    batch: Optional[int] = None
    seed: int = 0


@register(DisparityConfig)
class Disparity(Signal):
    cadence = property(lambda self: "summary" if self.cfg.form == "summary" else "eval")

    @property
    def requires(self) -> frozenset[str]:
        if self.cfg.form == "summary":
            return frozenset({"summaries"})
        if self.cfg.form == "replay":
            return frozenset({"models", "train_data"})
        raise ValueError(f"unknown form {self.cfg.form!r}")

    def update(self, obs: Observation) -> Optional[dict]:
        return self._summary(obs) if self.cfg.form == "summary" else self._replay(obs)

    def _summary(self, obs: Observation) -> dict:
        probe = int(self.ctx.config.get("record", {}).get("probe_batch", 256))
        pat = re.compile(r"^gnoise/l(\d+)$")
        layers = sorted((int(m.group(1)), v) for k, v in obs.summaries.items() for m in [pat.match(k)] if m)
        if self.ctx.n_train <= probe:
            return {"value": float("nan"), **{f"l{i}": float("nan") for i, _ in layers}}
        tr = [max(float(v[1]), 0.0) for _, v in layers]
        total_sq = [float(v[2]) for _, v in layers]
        d = float(np.sqrt(2 * sum(tr)))
        rel = d / float(np.sqrt(sum(total_sq))) if sum(total_sq) > 0 else float("nan")
        return {"value": d, "rel": rel, **{f"l{i}": float(np.sqrt(2 * t)) for (i, _), t in zip(layers, tr)}}

    def _replay(self, obs: Observation) -> dict:
        data = self.ctx.train_data()
        n = data.x.shape[0]
        b = self.cfg.batch or self.ctx.batch_size or n // 2
        b = min(b, n // 2)
        rng = np.random.default_rng([self.cfg.seed, obs.step])
        perm = torch.as_tensor(rng.permutation(n)[: 2 * b], device=data.x.device)
        g1 = batch_grad(obs.model, data.x[perm[:b]], data.target[perm[:b]], data.loss_fn)
        g2 = batch_grad(obs.model, data.x[perm[b:]], data.target[perm[b:]], data.loss_fn)
        row, total = {}, 0.0
        for group, names in layer_groups(obs.model).items():
            sq = float(sum(((g1[k] - g2[k]) ** 2).sum() for k in names))
            row[_short(group)] = float(np.sqrt(sq))
            total += sq
        mean_sq = float(sum(((g1[k] + g2[k]) ** 2 / 4).sum() for k in g1))
        d = float(np.sqrt(total))
        return {"value": d, "rel": d / np.sqrt(mean_sq) if mean_sq > 0 else float("nan"), **row}

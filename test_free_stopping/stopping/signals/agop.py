"""F2 study signal: the path of the input AGOP, and of each layer's weight direction, on
the unit sphere (AGOP.md §4, THEORY.md §3-4).

At each eval step this computes the AGOP G on the training set, the feature matrix
M = G^{1/2}, and each layer's flattened weights w_l, normalizes each to unit Frobenius
norm, and records the **squared chord distance to every earlier eval step**:
d2[t, s] = ||x_t - x_s||^2 for unit vectors x. For unit vectors the angle is
2 arcsin(sqrt(d2) / 2), and the angle between any two window means can be rebuilt from
these distances (stopping.agop_study.mean_chord2). So one replay pass is enough to
evaluate every lag, window, smoothing and threshold offline.

Distances are computed directly (not as 1 - cos), so small angles keep their precision
(about 1e-7 rad, from the float32 storage of the unit vectors).

Causal: row t holds distances to steps <= t only. The full history of unit vectors is
kept in memory on the signal's device (MNIST: ~2.5 MB per matrix per eval step, so
~1.5 GB for G and M over 300 eval steps; the weights of MNIST depth 8 add ~1.5 GB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from ..signal import Signal, SignalConfig, register
from ..source import Observation
from ._linalg import effective_rank, input_agop


@dataclass(frozen=True)
class AgopPathConfig(SignalConfig):
    name: ClassVar[str] = "agop_path"
    weights: bool = True  # also record each layer's weight-direction path (THEORY.md §3)
    mask_constant_inputs: bool = True  # drop input coordinates constant on the training set (AGOP.md §1.3)
    chunk: int = 4096  # training examples per forward/backward pass
    block: int = 16  # history rows per distance block (memory vs. speed)


class _Path:
    """History of unit vectors; add() returns the squared distances to all earlier ones."""

    def __init__(self, block: int):
        self.buf: Optional[torch.Tensor] = None
        self.k = 0
        self.block = block

    def add(self, v: torch.Tensor) -> list[float]:
        v = v.reshape(-1).to(torch.float32)
        out = []
        for j in range(0, self.k, self.block):
            diff = (self.buf[j:min(j + self.block, self.k)] - v).to(torch.float64)
            out.append((diff * diff).sum(1))
        d2 = torch.cat(out).cpu().tolist() if out else []
        if self.buf is None:
            self.buf = torch.empty((16, v.numel()), dtype=torch.float32, device=v.device)
        elif self.k == self.buf.shape[0]:
            grown = torch.empty((2 * self.k, v.numel()), dtype=torch.float32, device=v.device)
            grown[: self.k] = self.buf
            self.buf = grown
        self.buf[self.k] = v
        self.k += 1
        return d2


def _unit(a: torch.Tensor) -> torch.Tensor:
    a = a.to(torch.float64)
    n = torch.linalg.norm(a)
    return a / n if n > 0 else a


@register(AgopPathConfig)
class AgopPath(Signal):
    requires = frozenset({"models", "train_data"})
    cadence = "eval"

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.paths: dict[str, _Path] = {}
        self.x: Optional[torch.Tensor] = None
        self.keep: Optional[torch.Tensor] = None

    def _path(self, key: str) -> _Path:
        if key not in self.paths:
            self.paths[key] = _Path(self.cfg.block)
        return self.paths[key]

    def update(self, obs: Observation) -> Optional[dict]:
        if self.x is None:
            x = self.ctx.train_data().x.to(torch.float32)
            self.keep = (x.std(0) > 0) if self.cfg.mask_constant_inputs else torch.ones(
                x.shape[1], dtype=torch.bool, device=x.device)
            self.x = x
        G = input_agop(obs.model, self.x, self.cfg.chunk)[self.keep][:, self.keep]
        G = 0.5 * (G + G.T)
        lam, vec = torch.linalg.eigh(G)
        lam = lam.clamp(min=0)
        M = (vec * lam.sqrt()) @ vec.T
        tr, fro = float(lam.sum()), float(torch.linalg.norm(lam))
        row = {
            "tr_G": tr, "fro_G": fro,
            "purity_G": fro ** 2 / tr ** 2 if tr > 0 else float("nan"),
            "erank_G": effective_rank(lam.cpu().numpy()),
            "erank_M": effective_rank(lam.sqrt().cpu().numpy()),
            "d_kept": float(int(self.keep.sum())),
            "d2_G": self._path("G").add(_unit(G)),
            "d2_M": self._path("M").add(_unit(M)),
        }
        sq = 0.0
        for i, layer in enumerate(obs.model.layers):
            w = torch.cat([layer.weight.detach().reshape(-1), layer.bias.detach().reshape(-1)])
            n = float(torch.linalg.norm(w.to(torch.float64)))
            row[f"w_norm_l{i}"] = n
            sq += n * n
            if self.cfg.weights:
                row[f"d2_w{i}"] = self._path(f"w{i}").add(_unit(w))
        row["w_norm"] = sq ** 0.5
        # value: angle (rad) of M to the previous eval step, for quick plots only
        row["value"] = 2.0 * math.asin(min(math.sqrt(row["d2_M"][-1]) / 2, 1.0)) if row["d2_M"] else float("nan")
        return row

"""Measurements made on the live model during the replay pass (PHASE5_PLAN.md §2.3).

A probe declares the steps it needs (`steps(plan)`), is called at each of them with the
replay engine (`at(step, eng)`: the model is eng.loop.model, the optimizer eng.loop.opt,
the training inputs eng.loop.x_train), and returns its arrays at the end (`finish()`).
Probes only read: no parameter, gradient buffer, optimizer state or RNG stream is
touched, so they cannot change the replayed trajectory (the replay check would catch it).

Probes store raw material, not metrics (trajectory.quantities defines the metrics):
- AgopProbe:          AGOP spectrum per grid point, and the squared chord distance between
                      the unit feature matrices of every pair of grid points (for M = G^1/2
                      and for G). Every angle quantity, lag and variant comes from these.
- EarlySpectraProbe:  singular values and activation spectra at the grid points that are
                      not summary steps, computed by the same Recorder code as the stored
                      summaries, so early and recorded points are the same quantity.
- AdamDecayProbe:     Adam+L2 only: per layer, sum_j w_j^2 / (sqrt(v_hat_j) + eps) at every
                      log step, the weight-decay pull through Adam's scaling (N.3).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from stopping.signals._linalg import input_agop

from ..grid import early_points

_SUMMARY = re.compile(r"^(sv|actspec)/l(\d+)$")


@dataclass(frozen=True)
class Plan:
    grid: np.ndarray
    log_steps: np.ndarray
    summary_every: int
    steps: int
    optimizer: str
    gamma: float


class Probe:
    name = ""
    version = 1

    def steps(self, plan: Plan) -> set[int]:
        raise NotImplementedError

    def start(self, eng, plan: Plan) -> None:
        pass

    def at(self, step: int, eng) -> None:
        raise NotImplementedError

    def finish(self) -> dict[str, np.ndarray]:
        raise NotImplementedError


# ---- AGOP ----------------------------------------------------------------------------------

class _ChordHistory:
    """Unit vectors seen so far (float32, on their device); add() returns the squared
    distances ||x_new - x_s||^2 to every earlier one, computed directly in float64 rather
    than as 2 - 2 cos, so small angles keep about 1e-7 rad of precision."""

    def __init__(self, block: int = 16):
        self.buf: Optional[torch.Tensor] = None
        self.k = 0
        self.block = block

    def add(self, v: torch.Tensor) -> np.ndarray:
        v = v.reshape(-1).to(torch.float32)
        out = []
        for j in range(0, self.k, self.block):
            diff = (self.buf[j:min(j + self.block, self.k)] - v).to(torch.float64)
            out.append((diff * diff).sum(1))
        d2 = torch.cat(out).cpu().numpy() if out else np.zeros(0)
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


def _full_matrix(rows: list[np.ndarray]) -> np.ndarray:
    """Symmetric n x n matrix from rows of distances to earlier points."""
    n = len(rows)
    out = np.zeros((n, n), dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, :i] = r
        out[:i, i] = r
    return out


class AgopProbe(Probe):
    """G = (1/n) sum_i J(x_i)^T J(x_i) over the training set (Mallinar et al. Def. 2.1),
    restricted to input coordinates that vary over the training set (MNIST's constant
    border pixels would otherwise shrink G's rank under weight decay alone)."""
    name = "agop"
    version = 1

    def __init__(self, chunk: int = 4096, mask_constant_inputs: bool = True):
        self.chunk = chunk
        self.mask = mask_constant_inputs

    def steps(self, plan: Plan) -> set[int]:
        return set(int(s) for s in plan.grid)

    def start(self, eng, plan: Plan) -> None:
        x = eng.loop.x_train.to(torch.float32)
        self.keep = (x.std(0) > 0) if self.mask else torch.ones(x.shape[1], dtype=torch.bool, device=x.device)
        self.x = x
        self.hist = {"M": _ChordHistory(), "G": _ChordHistory()}
        self.rows = {"M": [], "G": []}
        self.step_list, self.eigs = [], []

    def at(self, step: int, eng) -> None:
        G = input_agop(eng.loop.model, self.x, self.chunk)[self.keep][:, self.keep]
        G = 0.5 * (G + G.T)
        lam, vec = torch.linalg.eigh(G)
        lam = lam.clamp(min=0)
        M = (vec * lam.sqrt()) @ vec.T
        self.rows["G"].append(self.hist["G"].add(_unit(G)))
        self.rows["M"].append(self.hist["M"].add(_unit(M)))
        self.step_list.append(step)
        self.eigs.append(lam.flip(0).cpu().numpy())  # descending

    def finish(self) -> dict[str, np.ndarray]:
        return {
            "agop_steps": np.asarray(self.step_list, dtype=np.int64),
            "agop_eig": np.stack(self.eigs) if self.eigs else np.zeros((0, 0)),  # [n_grid, d_kept], descending
            "agop_d2_M": _full_matrix(self.rows["M"]),
            "agop_d2_G": _full_matrix(self.rows["G"]),
            "agop_d_kept": np.asarray(int(self.keep.sum())),
        }


# ---- early spectra -------------------------------------------------------------------------

class EarlySpectraProbe(Probe):
    name = "early_spectra"
    version = 1

    def steps(self, plan: Plan) -> set[int]:
        return set(int(s) for s in early_points(plan.grid, plan.summary_every, plan.steps))

    def start(self, eng, plan: Plan) -> None:
        self.step_list, self.rows = [], {}

    def at(self, step: int, eng) -> None:
        self.step_list.append(step)
        for k, v in eng.recorder.summaries().items():  # the stored summaries' own code
            m = _SUMMARY.match(k)
            if m:
                self.rows.setdefault(f"early_{m.group(1)}_l{m.group(2)}", []).append(np.asarray(v, dtype=np.float64))

    def finish(self) -> dict[str, np.ndarray]:
        out = {"early_steps": np.asarray(self.step_list, dtype=np.int64)}
        out.update({k: np.stack(v) for k, v in self.rows.items()})
        return out


# ---- Adam+L2 decay pull --------------------------------------------------------------------

class AdamDecayProbe(Probe):
    """For Adam with coupled L2 (torch Adam, weight_decay = gamma), the decay term gamma*w
    enters the gradient and is divided by Adam's denominator sqrt(v_hat) + eps. Its inward
    pull on ||w||^2 per step is then about lr * gamma * sum_j w_j^2 / (sqrt(v_hat_j) + eps)
    (new_metrics.md §1.2). This probe stores S = sum_j w_j^2 / (sqrt(v_hat_j) + eps) per
    layer (weight and bias), with v_hat = exp_avg_sq / (1 - beta2^k) as torch computes it."""
    name = "adam_decay"
    version = 1

    def steps(self, plan: Plan) -> set[int]:
        if plan.optimizer != "adam_l2" or plan.gamma <= 0:
            return set()
        return set(int(s) for s in plan.log_steps)

    def start(self, eng, plan: Plan) -> None:
        self.step_list, self.rows = [], []

    def at(self, step: int, eng) -> None:
        opt = eng.loop.opt
        group = opt.param_groups[0]
        beta2, eps = group["betas"][1], group["eps"]
        row = []
        for layer in eng.loop.model.layers:
            total = 0.0
            for p in (layer.weight, layer.bias):
                st = opt.state.get(p, {})
                if "exp_avg_sq" not in st:
                    total = math.nan
                    break
                k = float(st["step"])
                bc2 = 1.0 - beta2 ** k
                denom = (st["exp_avg_sq"].to(torch.float64) / bc2).sqrt() + eps
                total += float((p.detach().to(torch.float64) ** 2 / denom).sum())
            row.append(total)
        self.step_list.append(step)
        self.rows.append(row)

    def finish(self) -> dict[str, np.ndarray]:
        if not self.step_list:
            return {}
        return {"adam_steps": np.asarray(self.step_list, dtype=np.int64),
                "adam_S": np.asarray(self.rows, dtype=np.float64)}  # [n_log, n_layers]


def default_probes() -> list[Probe]:
    return [AgopProbe(), EarlySpectraProbe(), AdamDecayProbe()]

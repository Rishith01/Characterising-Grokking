"""Weight norm: N.1, N.2, N.3 (new_metrics.md §1).

N1  log ||w||, all layers (weights and biases) together
N2  log ||w_l|| per layer
N3  rho_bar*(s, t) = (||w_s||^2 - ||w_t||^2) / (2 * sum_{k=s}^{t-1} D_k) over each grid
    interval (s, t], reported at t, for the whole network (layer -1) and per layer.
    D_k is the inward pull of weight decay alone in step k:
        SGD           lr_k * gamma * ||w_k||^2
        SGD+momentum  lr_k * gamma * ||w_k||^2 / (1 - mu)      (windows >> 1/(1-mu) steps)
        AdamW         lr_k * gamma * ||w_k||^2                 (exact)
        Adam+L2       lr_k * gamma * S_k, S_k = sum_j w_j^2 / (sqrt(v_hat_j) + eps)
                      (replay probe AdamDecayProbe; NaN until the replay file exists)
    The sum over steps is the trapezoid integral of D over the 50-step telemetry.
    1 = the norm falls as fast as decay alone would make it (pure decay), 0 = stationary,
    < 0 = growing. Undefined (NaN) at gamma = 0.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._common import at_grid, frame


def decay_pull(rec: dict, meta: dict, rep: Optional[dict] = None) -> Optional[np.ndarray]:
    """D at every log step, [n_log, n_layers]; None if undefined (gamma = 0, or Adam+L2
    without the replay probe)."""
    gamma = float(meta["gamma"])
    if gamma <= 0:
        return None
    lr = np.asarray(rec["lr"], dtype=np.float64)[:, None]
    sq = np.asarray(rec["w_norm"], dtype=np.float64) ** 2
    opt = meta["optimizer"]
    if opt in ("sgd", "adamw"):
        return lr * gamma * sq
    if opt == "sgd_momentum":
        return lr * gamma * sq / (1.0 - float(meta["momentum"]))
    if opt == "adam_l2":
        if rep is None or "adam_S" not in rep:
            return None
        S = at_grid(rep["adam_steps"], rep["adam_S"], np.asarray(rec["log_steps"]))
        return lr * gamma * S
    raise ValueError(f"unknown optimizer {opt!r}")


def rho_bar(steps: np.ndarray, sq_norm: np.ndarray, D: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """rho_bar* over each grid interval, reported at its right end (NaN at the first grid
    step). steps: log steps; sq_norm and D: values at those steps (1-D)."""
    steps = np.asarray(steps, dtype=np.float64)
    # cumulative trapezoid integral of D over steps
    # (a NaN in D, e.g. Adam's state before its first update, voids only the intervals
    # that contain it)
    seg = 0.5 * (D[1:] + D[:-1]) * np.diff(steps)
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(seg))])
    cum = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(seg), seg, 0.0))])
    idx = np.searchsorted(steps, grid)
    ok = (idx < len(steps)) & (steps[np.clip(idx, 0, len(steps) - 1)] == grid)
    out = np.full(len(grid), np.nan)
    for j in range(1, len(grid)):
        if not (ok[j] and ok[j - 1]):
            continue
        a, b = idx[j - 1], idx[j]
        if bad[b] - bad[a]:
            continue
        denom = 2.0 * (cum[b] - cum[a])
        if denom > 0 and np.isfinite(denom):
            out[j] = (sq_norm[a] - sq_norm[b]) / denom
    return out


def curves(rec: dict, meta: dict, grid: np.ndarray, rep: Optional[dict] = None) -> list[pd.DataFrame]:
    steps = np.asarray(rec["log_steps"])
    w = np.asarray(rec["w_norm"], dtype=np.float64)  # [n_log, L]
    total = np.sqrt((w ** 2).sum(1))
    out = [frame("N1", "main", -1, grid, np.log(at_grid(steps, total, grid)))]
    wg = at_grid(steps, w, grid)
    out += [frame("N2", "main", i, grid, np.log(wg[:, i])) for i in range(w.shape[1])]
    D = decay_pull(rec, meta, rep)
    if D is not None:
        out.append(frame("N3", "main", -1, grid, rho_bar(steps, total ** 2, D.sum(1), grid)))
        out += [frame("N3", "main", i, grid, rho_bar(steps, w[:, i] ** 2, D[:, i], grid)) for i in range(w.shape[1])]
    return out

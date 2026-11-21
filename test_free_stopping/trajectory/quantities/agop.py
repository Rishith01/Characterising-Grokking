"""AGOP, as in the paper (AP) and online (AO): new_metrics.md §3-4.

All from the replay probe's matrices d2[i, j] = ||X_i - X_j||^2 between the unit
feature matrices X = M / ||M||_F (M = G^{1/2}) at grid points i and j (and the same for
X = G / ||G||_F, variant "G"). For unit vectors:
    angle  theta = 2 arcsin(sqrt(d2) / 2)        in [0, pi/2] for PSD matrices
    cosine        = 1 - d2 / 2

AP1  cos(M_t, M_T), T = the last grid step (Mallinar et al. eq. 8). Uses the end of the
     run: a hindsight reference, never a stopping signal.
AO1  theta(M_t, M_{t/2}): the angle turned since half the current time. t/2 is taken at
     the nearest grid step at or before t (always <= t, so causal).
AO2  theta(M_t, M_{t - cT}) with c = 0.05 (main) and 0.02 (variant "c0.02"); NaN for
     t < cT. Variant "lag1": the angle to the previous grid step, the jitter level.
The online quantities use only grid steps <= t: causal by construction (tested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..grid import nearest_index
from ._common import at_grid, frame

AO2_FRACS = {"main": 0.05, "c0.02": 0.02}


def angle(d2: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(np.sqrt(np.clip(d2, 0.0, None)) / 2.0, 0.0, 1.0))


def alignment_to_final(d2: np.ndarray) -> np.ndarray:
    """cos(X_i, X_last) for every grid index i."""
    return 1.0 - d2[:, -1] / 2.0


def angle_to_lag(d2: np.ndarray, steps: np.ndarray, back: np.ndarray) -> np.ndarray:
    """theta(X_i, X_j) with j the grid index nearest steps[i] - back[i], restricted to
    j < i (and steps[j] <= steps[i]); NaN where steps[i] - back[i] < 0 or no j < i."""
    steps = np.asarray(steps, dtype=np.float64)
    target = steps - np.asarray(back, dtype=np.float64)
    j = nearest_index(steps, target)
    out = np.full(len(steps), np.nan)
    for i in range(len(steps)):
        if target[i] < 0 or i == 0:
            continue
        jj = min(int(j[i]), i - 1)
        out[i] = angle(d2[i, jj])
    return out


def curves(rep: dict, meta: dict, grid: np.ndarray) -> list[pd.DataFrame]:
    steps = np.asarray(rep["agop_steps"], dtype=np.int64)
    T = float(meta["steps"])
    out = []
    for variant, key in (("main", "agop_d2_M"), ("G", "agop_d2_G")):
        d2 = np.asarray(rep[key], dtype=np.float64)
        out.append(frame("AP1", variant, -1, grid, at_grid(steps, alignment_to_final(d2), grid)))
    d2 = np.asarray(rep["agop_d2_M"], dtype=np.float64)
    half = np.where(steps > 0, steps / 2.0, -1.0)
    out.append(frame("AO1", "main", -1, grid, at_grid(steps, angle_to_lag(d2, steps, half), grid)))
    for variant, c in AO2_FRACS.items():
        back = np.full(len(steps), c * T)
        out.append(frame("AO2", variant, -1, grid, at_grid(steps, angle_to_lag(d2, steps, back), grid)))
    prev = np.full(len(steps), np.nan)
    prev[1:] = angle(d2[np.arange(1, len(steps)), np.arange(len(steps) - 1)])
    out.append(frame("AO2", "lag1", -1, grid, at_grid(steps, prev, grid)))
    return out

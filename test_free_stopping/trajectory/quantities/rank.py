"""Rank: R.1, R.2, R.3 (new_metrics.md §2).

R1  effective rank of each hidden layer's training-set activation covariance
    (eigenvalues = actspec/l{i}); variant "pr": participation ratio 1 / sum p^2
R2  effective rank of each weight matrix from its singular values (sv/l{i}), all layers
R3  effective rank of the AGOP feature matrix M = G^{1/2} (eigenvalues sqrt(eig G)),
    layer -1; variant "G": of G itself

Effective rank (Roy & Vetterli 2007) = exp(entropy of p), p = spectrum / its sum.
Grid points that are summary steps use the recorded summaries; the early grid points
use the replay probe EarlySpectraProbe (the same Recorder code), when available.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from stopping.signals._linalg import effective_rank

from ._common import at_grid, frame

_KEY = re.compile(r"^(sv|actspec)_l(\d+)$")


def participation_ratio(eigs: np.ndarray) -> float:
    lam = np.clip(np.asarray(eigs, dtype=np.float64), 0.0, None)
    s = lam.sum()
    if not s > 0:
        return float("nan")
    p = lam / s
    return float(1.0 / (p * p).sum())


def _series(rec: dict, rep: Optional[dict], kind: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """layer -> (steps, spectra rows) merging recorded summary steps and early replay steps."""
    out = {}
    for k in rec:
        m = _KEY.match(k)
        if not m or m.group(1) != kind:
            continue
        layer = int(m.group(2))
        steps, rows = list(rec["summary_steps"]), list(rec[k])
        ek = f"early_{kind}_l{layer}"
        if rep is not None and ek in rep:
            steps += list(rep["early_steps"])
            rows += list(rep[ek])
        order = np.argsort(steps, kind="stable")
        out[layer] = (np.asarray(steps, dtype=np.int64)[order], [rows[i] for i in order])
    return out


def _apply(series: dict, fn, grid: np.ndarray) -> dict[int, np.ndarray]:
    return {layer: at_grid(steps, np.array([fn(r) for r in rows]), grid) for layer, (steps, rows) in series.items()}


def curves(rec: dict, grid: np.ndarray, rep: Optional[dict] = None) -> list[pd.DataFrame]:
    out = []
    act = _series(rec, rep, "actspec")
    for layer, v in sorted(_apply(act, effective_rank, grid).items()):
        out.append(frame("R1", "main", layer, grid, v))
    for layer, v in sorted(_apply(act, participation_ratio, grid).items()):
        out.append(frame("R1", "pr", layer, grid, v))
    sv = _series(rec, rep, "sv")
    for layer, v in sorted(_apply(sv, effective_rank, grid).items()):
        out.append(frame("R2", "main", layer, grid, v))
    if rep is not None and "agop_eig" in rep and len(rep["agop_eig"]):
        eig = np.clip(np.asarray(rep["agop_eig"], dtype=np.float64), 0.0, None)
        steps = np.asarray(rep["agop_steps"])
        out.append(frame("R3", "main", -1, grid, at_grid(steps, np.array([effective_rank(np.sqrt(e)) for e in eig]), grid)))
        out.append(frame("R3", "G", -1, grid, at_grid(steps, np.array([effective_rank(e) for e in eig]), grid)))
    return out

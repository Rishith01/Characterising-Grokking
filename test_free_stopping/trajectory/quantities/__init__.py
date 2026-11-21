"""Stage Q: every metric of new_metrics.md as a pure function of the raw arrays,
evaluated on the common grid. No I/O, no randomness: changing a definition means
re-running this stage (minutes), never the replay.

compute_curves() returns one long table for a run with columns
    metric   metric ID as in new_metrics.md (N1, N2, N3, R1, R2, R3, AP1, AO1, AO2)
    variant  "main" or the name of a robustness variant
    layer    layer index, or -1 for a whole-network quantity
    step     grid step
    value
Quantities that need the replay pass are simply absent until its raw file exists.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..grid import common_grid
from . import agop, norm, rank

CURVE_COLUMNS = ("metric", "variant", "layer", "step", "value")
VERSION = 1


def grid_of(meta: dict) -> np.ndarray:
    return common_grid(int(meta["steps"]), int(meta["log_every"]), int(meta["summary_every"]))


def compute_curves(recorded: tuple[dict, dict], replay: Optional[tuple[dict, dict]] = None) -> pd.DataFrame:
    """All curves for one run. recorded / replay: (arrays, meta) from trajectory.rawfile.load_raw."""
    rec, meta = recorded
    rep = replay[0] if replay is not None else None
    grid = grid_of(meta)
    if rep is not None and "grid" in rep and not np.array_equal(rep["grid"], grid):
        raise ValueError(f"{meta.get('run_id')}: replay grid differs from the recorded grid")
    parts = norm.curves(rec, meta, grid, rep) + rank.curves(rec, grid, rep)
    if rep is not None and "agop_d2_M" in rep:
        parts += agop.curves(rep, meta, grid)
    rows = [df for df in parts if len(df)]
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(CURVE_COLUMNS))
    return out[list(CURVE_COLUMNS)]

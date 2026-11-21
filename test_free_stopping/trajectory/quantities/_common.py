"""Helpers shared by the quantity modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def at_grid(src_steps: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """values (indexed like src_steps, any trailing shape) at the grid steps; NaN where a
    grid step is not among src_steps."""
    src_steps = np.asarray(src_steps)
    values = np.asarray(values, dtype=np.float64)
    out = np.full((len(grid),) + values.shape[1:], np.nan)
    if len(src_steps) == 0:
        return out
    idx = np.searchsorted(src_steps, grid)
    idx_c = np.clip(idx, 0, len(src_steps) - 1)
    hit = (idx < len(src_steps)) & (src_steps[idx_c] == grid)
    out[hit] = values[idx_c[hit]]
    return out


def frame(metric: str, variant: str, layer: int, grid: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"metric": metric, "variant": variant, "layer": int(layer),
                         "step": np.asarray(grid, dtype=np.int64), "value": np.asarray(values, dtype=np.float64)})


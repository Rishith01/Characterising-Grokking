"""The common grid (new_metrics.md §0): every metric is evaluated at the same steps.

Grid = every summary step (multiples of summary_every, 500 in the corpus) plus n_early
log-spaced steps before early_max (default 10 x summary_every = 5,000), each snapped to
the nearest log step (multiples of log_every, 50). Every grid step is therefore a log
step, so telemetry exists at all of them; summary-based quantities exist at the summary
steps and need the replay pass for the early points in between.

For the corpus (T = 100,000, log 50, summary 500) this gives 201 summary steps plus
about 25 distinct early points: about 226 points per run.
"""
from __future__ import annotations

import numpy as np


def common_grid(steps: int, log_every: int, summary_every: int, n_early: int = 30,
                early_max: int | None = None) -> np.ndarray:
    """Sorted unique grid steps in [0, steps]. Contains 0, every summary step and the horizon."""
    if steps <= 0 or log_every <= 0 or summary_every % log_every:
        raise ValueError("need steps > 0 and summary_every a multiple of log_every")
    summary = np.union1d(np.arange(0, steps + 1, summary_every), [steps])
    hi = min(early_max if early_max is not None else 10 * summary_every, steps)
    early = np.array([], dtype=np.int64)
    if hi > log_every and n_early > 0:
        raw = np.geomspace(log_every, hi, n_early)
        early = (np.round(raw / log_every) * log_every).astype(np.int64)
        early = early[(early > 0) & (early <= hi)]
    return np.union1d(summary, early).astype(np.int64)


def early_points(grid: np.ndarray, summary_every: int, steps: int) -> np.ndarray:
    """Grid steps that are not summary steps: summary-based quantities need replay there."""
    grid = np.asarray(grid, dtype=np.int64)
    return grid[(grid % summary_every != 0) & (grid != steps)]


def nearest_index(grid: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index of the grid step nearest each target, ties to the earlier step."""
    grid = np.asarray(grid)
    target = np.asarray(target, dtype=np.float64)
    hi = np.clip(np.searchsorted(grid, target), 0, len(grid) - 1)
    lo = np.clip(hi - 1, 0, len(grid) - 1)
    return np.where(np.abs(grid[lo] - target) <= np.abs(grid[hi] - target), lo, hi)

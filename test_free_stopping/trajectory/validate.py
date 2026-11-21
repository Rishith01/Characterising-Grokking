"""Checks of the Phase 5 curves against the F2 study, for the replay pilot (PHASE5_PLAN.md M2).

F2 (phase4/agop_final/series.parquet) recorded, per run and per F2 eval step:
    align_M, align_G   cos(X_t, X_T) for X = M = G^{1/2} and X = G      -> AP1 main / G
    erank_G            effective rank of the AGOP G                       -> R3 variant G
    w_norm             total parameter norm                               -> exp(N1)
Both pipelines compute these the same way from the same replayed weights, so at the steps
the two grids share they must agree to float32 precision (about 1e-6). F2's theta_const
is a window-smoothed angle, so it is not compared (AO2 has no smoothing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIRS = (  # (metric, variant, F2 column, transform of our value)
    ("AP1", "main", "align_M", None),
    ("AP1", "G", "align_G", None),
    ("R3", "G", "erank_G", None),
    ("N1", "main", "w_norm", np.exp),
)


def compare_with_f2(curves: pd.DataFrame, f2_series: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, quantity): shared steps and the largest relative difference."""
    out = []
    f2 = f2_series.set_index(["run_id", "step"])
    for rid, c in curves.groupby("run_id"):
        if rid not in f2.index.get_level_values(0):
            continue
        theirs = f2.loc[rid]
        for metric, variant, col, fn in PAIRS:
            ours = c[(c["metric"] == metric) & (c["variant"] == variant) & (c["layer"] == -1)].set_index("step")["value"]
            shared = ours.index.intersection(theirs.index)
            a = ours.loc[shared].to_numpy(dtype=float)
            if fn is not None:
                a = fn(a)
            b = theirs.loc[shared, col].to_numpy(dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            rel = np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12)
            out.append({"run_id": rid, "quantity": f"{metric} {variant} vs {col}", "shared_steps": int(ok.sum()),
                        "max_rel_diff": float(rel.max()) if ok.any() else float("nan")})
    return pd.DataFrame(out)

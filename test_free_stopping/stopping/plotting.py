"""Plots for looking at signals: each signal under the run's train/test accuracy, with
every method's stop marked (DESIGN.md §13.6). Takes the curves as data, so this module
never reads test data itself."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .rule import StopDecision  # noqa: E402


def plot_run(curves: Mapping[str, list], traces: Mapping[str, pd.DataFrame],
             decisions: Mapping[str, StopDecision], out_path: Path, title: str = "",
             columns: Optional[Mapping[str, str]] = None) -> Path:
    """curves: step, train_acc, test_acc (lists). traces: label -> trace (step + columns).
    decisions: method name -> stop. columns: label -> column to draw (default "value")."""
    columns = columns or {}
    n = 1 + len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.9 * n + 0.6), sharex=True, squeeze=False)
    axes = axes[:, 0]
    steps = np.maximum(np.asarray(curves["step"], dtype=float), 1)
    ax = axes[0]
    ax.plot(steps, curves["train_acc"], label="train acc", color="tab:blue")
    ax.plot(steps, curves["test_acc"], label="test acc", color="tab:orange")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower right", fontsize=7)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (name, d) in enumerate(sorted(decisions.items())):
        if d.step is not None:
            for a in axes:
                a.axvline(max(d.step, 1), color=colors[k % len(colors)], lw=0.9, ls="--",
                          label=name if a is ax else None)
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    for a, (label, df) in zip(axes[1:], traces.items()):
        col = columns.get(label, "value")
        if len(df) and col in df:
            y = df[col].to_numpy(dtype=float)
            a.plot(np.maximum(df["step"].to_numpy(dtype=float), 1), y, lw=1)
            fin = y[np.isfinite(y)]
            if fin.size and fin.min() > 0 and fin.max() / fin.min() > 100:
                a.set_yscale("log")  # losses, disparity: spans decades
        a.set_ylabel(f"{label}\n[{col}]", fontsize=7)
    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("step")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path

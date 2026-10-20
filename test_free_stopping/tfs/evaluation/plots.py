"""Diagnostic plots of single runs (evaluation side: may read the oracle)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..storage.reader import RunReader  # noqa: E402
from .oracle import Oracle  # noqa: E402

C_TRAIN, C_TEST, C_NORM = "#eb6834", "#2a78d6", "#52514e"


def run_label(reader: RunReader) -> str:
    c = reader.config
    t = c.task.params
    task = {"modular": lambda: f"{t['op']} p={t['p']} frac={t['train_frac']}",
            "mnist": lambda: f"n={t['n_train']}",
            "teacher_student": lambda: f"N={t['n_train']}"}.get(c.app, lambda: "")()
    return (f"{c.app} {task} | depth {c.model.depth}×{c.model.width} {c.model.activation} α={c.model.init_scale:g} | "
            f"{c.optim.name} lr={c.optim.lr:g} γ={c.optim.weight_decay:g} | seed {c.seed}")


def plot_run(run_dir: Path, ax_acc, ax_norm: Optional[object] = None, title: bool = True) -> None:
    r = RunReader(run_dir)
    tel = r.telemetry()
    orc = Oracle(run_dir).table()
    x = np.maximum(tel["step"].to_numpy(), 1)
    ax_acc.plot(x, tel["train_acc"], color=C_TRAIN, ls=(0, (3, 2)), lw=1.6, label="train")
    ax_acc.plot(np.maximum(orc["step"].to_numpy(), 1), orc["test_acc"], color=C_TEST, lw=1.8, label="test")
    ax_acc.set_xscale("log")
    ax_acc.set_ylim(-0.03, 1.03)
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_xlabel("step")
    ax_acc.grid(alpha=0.3)
    if title:
        ax_acc.set_title(run_label(r), fontsize=8)
    if ax_norm is not None:
        cols = [c for c in tel.columns if c.endswith("_w_norm")]
        total = np.sqrt((tel[cols] ** 2).sum(axis=1))
        ax_norm.plot(x, total, color=C_NORM, lw=1.2, label="‖w‖")
        ax_norm.set_ylabel("‖w‖", color=C_NORM)
        ax_norm.tick_params(axis="y", colors=C_NORM)


def plot_runs(run_dirs: list[Path], out_path: Path, ncols: int = 2) -> Path:
    n = len(run_dirs)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.6 * nrows), squeeze=False)
    for ax, d in zip(axes.flat, run_dirs):
        plot_run(d, ax, ax.twinx())
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    return out_path

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
    # Molab's dark theme leaked into Phase 1's figures (black axes, invisible labels):
    # always render in matplotlib's default light style on a white background.
    with plt.style.context("default"):
        return _plot_runs(run_dirs, out_path, ncols)


def _plot_runs(run_dirs: list[Path], out_path: Path, ncols: int) -> Path:
    n = len(run_dirs)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 3.6 * nrows), squeeze=False,
                             facecolor="white")
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


def plot_lr_scan(rows: list[dict], curves: dict[str, dict], out_dir: Path) -> list[Path]:
    """One figure per application variant: train loss vs step for every scanned lr, one
    panel per (optimizer, alpha). Unstable runs (lr_select.scan_row) are dashed."""
    with plt.style.context("default"):
        return [_plot_lr_variant(v, [r for r in rows if r["variant"] == v], curves, Path(out_dir))
                for v in sorted({r["variant"] for r in rows})]


def _plot_lr_variant(variant: str, rows: list[dict], curves: dict, out_dir: Path) -> Path:
    from matplotlib.colors import LogNorm

    optims = [o for o in ("adamw", "adam_l2", "sgd_momentum", "sgd") if any(r["optim"] == o for r in rows)]
    alphas = {o: sorted({r["alpha"] for r in rows if r["optim"] == o}) for o in optims}
    ncols = max(len(a) for a in alphas.values())
    lrs = [r["lr"] for r in rows]
    norm = LogNorm(min(lrs), max(lrs)) if min(lrs) < max(lrs) else None
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(len(optims), ncols, figsize=(4.2 * ncols, 3.0 * len(optims)), squeeze=False,
                             facecolor="white")
    for i, o in enumerate(optims):
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(alphas[o]):
                ax.axis("off")
                continue
            a = alphas[o][j]
            for r in sorted((r for r in rows if r["optim"] == o and r["alpha"] == a), key=lambda r: r["lr"]):
                c = curves.get(r["run_id"])
                if not c:
                    continue
                color = cmap(norm(r["lr"])) if norm else cmap(0.5)
                ax.plot(np.maximum(c["step"], 1), c["train_loss"], color=color, lw=1.2,
                        ls="-" if r["stable"] else (0, (3, 2)), label=f"{r['lr']:g}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{o}, α={a:g}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=2, title="lr (dashed: unstable)", title_fontsize=6)
    fig.suptitle(f"lr scan: {variant} (train loss)", fontsize=11)
    fig.tight_layout()
    out = out_dir / f"lr_scan_{variant.replace(' ', '_').replace('^', '').replace('*', 'mul').replace('+', 'plus')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    return out

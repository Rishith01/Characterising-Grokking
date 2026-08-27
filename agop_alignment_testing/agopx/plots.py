"""Plotting (project.md Section 6). Phase 0 only needs a sanity-check plot of the
raw metrics.jsonl (accuracy/loss/trace(M) vs t) to eyeball that a run behaved. The
Figure 2B / 5B reproduction plots (alignment + circulant deviation vs t) are Phase 1
work, gated on agopx/probes/offline.py being implemented.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_json(Path(run_dir) / "metrics.jsonl", lines=True)


def plot_run_diagnostics(run_dir: Path, out_path: Path | None = None):
    df = load_metrics(run_dir)
    run_dir = Path(run_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["t"], df["train/accuracy"], label="train")
    axes[0].plot(df["t"], df["test/accuracy"], label="test")
    axes[0].set_title("accuracy")
    axes[0].set_xlabel("t")
    axes[0].legend()

    axes[1].plot(df["t"], df["train/loss"], label="train")
    axes[1].plot(df["t"], df["test/loss"], label="test")
    axes[1].set_yscale("log")
    axes[1].set_title("loss")
    axes[1].set_xlabel("t")
    axes[1].legend()

    axes[2].plot(df["t"], df["trace_M"])
    axes[2].set_title("trace(M)")
    axes[2].set_xlabel("t")

    fig.suptitle(run_dir.name)
    fig.tight_layout()

    out_path = out_path or (run_dir / "diagnostics.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

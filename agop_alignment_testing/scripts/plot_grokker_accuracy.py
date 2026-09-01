"""Train/test accuracy for the Phase 2 grokker runs, one panel per
(learner-kernel, operation) cell.

Rows are the three learner configurations in the grokker corpus -- RFM with the
quadratic kernel, RFM with the Gaussian kernel, and the one-hidden-layer quadratic
network -- and columns are the four modular operations. Seeds are collapsed: one
representative seed is drawn per cell (--seed, default 0), since seed variance is a
Phase 3 scoring axis rather than something to read off an accuracy plot.

Reads run directories directly rather than corpus.jsonl, so it works before the
corpus is built. Output goes to runs/plots/ rather than runs/<corpus>/plots/, since
runs/plots/ is the one path under runs/ that is not gitignored -- figures are small
and worth versioning, unlike the feature matrices.

The x axes are deliberately not shared across rows: RFM panels are indexed by RFM
iteration and NN panels by epoch. The NN runs go to 300 epochs so that the circulant
deviation's post-minimum behaviour is observable, but the accuracy transition is over
within ~15, so --nn-xlim truncates the NN row for readability. The truncation is
always stated in the row label; it is never silent.

Colours are slots 1 and 2 of the reference categorical palette. The pair validates on
all six checks against the light surface (worst-pair CVD deltaE 24.7, normal-vision
33.6, both well clear of the >=8 / >=15 floors), and solid-vs-dashed carries the same
distinction independently of hue, so identity never rests on colour alone.

Usage: python -m scripts.plot_grokker_accuracy [--seed 0] [--nn-xlim 50]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OPS = ["x+y", "x-y", "x*y", "x/y"]
OP_SLUGS = {"x+y": "xplusy", "x-y": "xminusy", "x*y": "xmulty", "x/y": "xdivy"}

# (row label, run-id prefix, x-axis label, truncatable by --nn-xlim)
ROWS = [
    ("RFM · quadratic", "grok_rfm_quadratic", "RFM iteration", False),
    ("RFM · gaussian", "grok_rfm_gaussian", "RFM iteration", False),
    ("Neural net · quadratic act.", "grok_nn", "epoch", True),
]

# Reference categorical palette, slots 1 and 2. Validated as a pair on the light
# surface; see the module docstring.
C_TEST = "#2a78d6"
C_TRAIN = "#eb6834"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d8d4"

TRAIN_STYLE = dict(color=C_TRAIN, linestyle=(0, (3, 2)), linewidth=1.7, label="train")
TEST_STYLE = dict(color=C_TEST, linestyle="-", linewidth=2.0, label="test")


def load_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_json(run_dir / "metrics.jsonl", lines=True)


def _style_axis(ax, show_yticklabels: bool) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.set_ylim(-0.06, 1.06)
    ax.set_yticks([0.0, 0.5, 1.0])
    if not show_yticklabels:
        ax.set_yticklabels([])


def plot_grid(runs_dir: Path, seed: int, nn_xlim: int, out_path: Path) -> Path:
    fig, axes = plt.subplots(len(ROWS), len(OPS), figsize=(14, 8.2), squeeze=False)

    missing = []
    for r, (row_label, prefix, xlabel, truncatable) in enumerate(ROWS):
        for c, op in enumerate(OPS):
            ax = axes[r][c]
            _style_axis(ax, show_yticklabels=(c == 0))

            run_dir = runs_dir / f"{prefix}_{OP_SLUGS[op]}_seed{seed}"
            if not (run_dir / "metrics.jsonl").exists():
                missing.append(run_dir.name)
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color=C_TRAIN, fontsize=9)
                continue

            df = load_metrics(run_dir)
            ax.plot(df["t"], df["train/accuracy"], **TRAIN_STYLE)
            ax.plot(df["t"], df["test/accuracy"], **TEST_STYLE)

            hi = df["t"].max()
            if truncatable and nn_xlim > 0:
                hi = min(hi, nn_xlim)
            ax.set_xlim(df["t"].min(), hi)

            if r == 0:
                ax.set_title(op, fontsize=12, color=INK, pad=10)
            if r in (1, 2):
                ax.set_xlabel(xlabel, fontsize=9.5, color=INK_SECONDARY)
            if c == 0:
                ax.set_ylabel("accuracy", fontsize=9.5, color=INK_SECONDARY)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005),
               handlelength=2.6, columnspacing=2.4, labelcolor=INK_SECONDARY)

    fig.suptitle("Phase 2 grokkers — train and test accuracy", fontsize=15,
                 color=INK, x=0.5, y=0.985)
    fig.text(0.5, 0.945, f"p = 61 · training fraction 0.5 · seed {seed}",
             ha="center", fontsize=10, color=INK_SECONDARY)

    fig.tight_layout(rect=(0.055, 0.035, 1, 0.925))

    # Row labels go left of the y-axis labels, positioned from the laid-out axes so
    # they stay put whatever the figure size.
    for r, (row_label, _, _, truncatable) in enumerate(ROWS):
        box = axes[r][0].get_position()
        label = row_label
        if truncatable and nn_xlim > 0:
            label += f"\n(first {nn_xlim} of 300 epochs)"
        fig.text(0.012, box.y0 + box.height / 2, label, rotation=90,
                 va="center", ha="left", fontsize=11, color=INK, fontweight="bold",
                 linespacing=1.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)

    if missing:
        print(f"warning: {len(missing)} run(s) missing: {', '.join(missing)}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", type=str, default="runs/phase2")
    parser.add_argument("--out-dir", type=str, default="runs/plots")
    parser.add_argument(
        "--nn-xlim", type=int, default=50,
        help="truncate the NN row to this many epochs (0 = full 300-epoch trajectory)",
    )
    args = parser.parse_args()

    suffix = "full" if args.nn_xlim <= 0 else f"nn{args.nn_xlim}"
    out = Path(args.out_dir) / f"grokker_accuracy_seed{args.seed}_{suffix}.png"
    print(f"wrote {plot_grid(Path(args.runs_dir), args.seed, args.nn_xlim, out)}")


if __name__ == "__main__":
    main()

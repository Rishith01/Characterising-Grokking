"""Summary plots for the Phase 2 frozen corpus (project.md Phase 2). Reads
corpus.jsonl + the individual runs' metrics.jsonl and produces one overview plot per
category, plus a compact grok-step summary across the whole corpus.

Usage: python -m scripts.plot_phase2
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from agopx.runner import load_run

OUT_DIR = Path("runs/phase2/plots")
OP_COLORS = {
    "x+y": "tab:blue",
    "x-y": "tab:orange",
    "x*y": "tab:green",
    "x/y": "tab:red",
}


def _traj_df(run_dir: str) -> pd.DataFrame:
    traj = load_run(run_dir)
    return pd.DataFrame([{"t": s.t, **s.metrics} for s in traj])


def plot_grokkers(corpus: pd.DataFrame):
    grokkers = corpus[corpus["category"] == "grokker"]
    kernels = ["quadratic", "gaussian", None]  # None -> NN
    titles = ["RFM: quadratic", "RFM: gaussian", "NN"]
    ops = ["x+y", "x-y", "x*y", "x/y"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, kernel, title in zip(axes, kernels, titles):
        if kernel is None:
            subset = grokkers[grokkers["learner"] == "nn"]
        else:
            subset = grokkers[grokkers["kernel_type"] == kernel]
        for op in ops:
            op_rows = subset[subset["operation"] == op]
            for _, row in op_rows.iterrows():
                df = _traj_df(row["run_dir"])
                ax.plot(df["t"], df["test/accuracy"], color=OP_COLORS[op], alpha=0.7, linewidth=1.3)
        ax.set_title(title)
        ax.set_xlabel("t (iteration / epoch)")
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("test accuracy")
    handles = [plt.Line2D([], [], color=c, label=op) for op, c in OP_COLORS.items()]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("Phase 2 grokkers: 4 ops x {quadratic, gaussian, NN} x 3 seeds (36 runs)")
    fig.tight_layout()
    out = OUT_DIR / "grokkers.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_non_grokkers(corpus: pd.DataFrame):
    non_grok = corpus[corpus["category"] == "non_grokker"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in non_grok.iterrows():
        df = _traj_df(row["run_dir"])
        label = row["run_id"].replace("nongrok_", "")
        style = "-" if "no_decay" in row["run_id"] else ("--" if "random_labels" in row["run_id"] else ":")
        ax.plot(df["t"], df["test/accuracy"], label=label, linewidth=1.5, linestyle=style)
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=1, label="grok threshold")
    ax.set_xlabel("t (iteration / epoch)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Phase 2 non-grokkers: low training fraction, NN no-decay, random labels (9 runs)")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    out = OUT_DIR / "non_grokkers.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_fast_learners(corpus: pd.DataFrame):
    fast = corpus[corpus["category"] == "fast_learner"]
    grokkers = corpus[(corpus["category"] == "grokker") & (corpus["kernel_type"] == "quadratic") & (corpus["seed"] == 0)]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    ops = ["x+y", "x-y", "x*y", "x/y"]
    for ax, op in zip(axes, ops):
        fast_row = fast[fast["operation"] == op].iloc[0]
        df_fast = _traj_df(fast_row["run_dir"])
        ax.plot(df_fast["t"], df_fast["test/accuracy"], color="tab:purple", label="fast-learner M_0", linewidth=2)

        grok_row = grokkers[grokkers["operation"] == op]
        if len(grok_row) > 0:
            df_grok = _traj_df(grok_row.iloc[0]["run_dir"])
            ax.plot(df_grok["t"], df_grok["test/accuracy"], color="gray", label="ordinary M_0=I", linewidth=1.3, linestyle="--")

        ax.axhline(0.9, color="gray", linestyle=":", linewidth=1)
        ax.set_title(op)
        ax.set_xlabel("t")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Phase 2 fast learners: oracle-seeded M_0 vs ordinary M_0=I (same op, quadratic, seed 0)")
    fig.tight_layout()
    out = OUT_DIR / "fast_learners.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_grok_step_summary(corpus: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["grokker", "fast_learner", "non_grokker"]
    cat_x = {c: i for i, c in enumerate(categories)}
    markers = {"rfm": "o", "nn": "^"}

    never_y = -3
    for _, row in corpus.iterrows():
        x = cat_x[row["category"]] + (hash(row["run_id"]) % 1000) / 1000 * 0.6 - 0.3
        y = row["grok_step"] if pd.notna(row["grok_step"]) else never_y
        op = row["operation"]
        color = OP_COLORS.get(op, "black")
        marker = markers.get(row["learner"], "x")
        ax.scatter(x, y, color=color, marker=marker, s=45, alpha=0.8, edgecolors="none")

    ax.axhline(never_y + 1, color="gray", linestyle=":", linewidth=1)
    ax.text(len(categories) - 0.5, never_y, "never grokked", va="center", fontsize=9, color="gray")
    ax.set_xticks(list(cat_x.values()))
    ax.set_xticklabels(list(cat_x.keys()))
    ax.set_ylabel("grok_step")
    ax.set_title("Phase 2 corpus: grok_step by category (color=op, o=RFM, ^=NN)")

    op_handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", label=op) for op, c in OP_COLORS.items()]
    learner_handles = [
        plt.Line2D([], [], color="black", marker="o", linestyle="", label="RFM"),
        plt.Line2D([], [], color="black", marker="^", linestyle="", label="NN"),
    ]
    ax.legend(handles=op_handles + learner_handles, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    out = OUT_DIR / "grok_step_summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open("corpus.jsonl")]
    corpus = pd.DataFrame(rows)

    outputs = [
        plot_grokkers(corpus),
        plot_non_grokkers(corpus),
        plot_fast_learners(corpus),
        plot_grok_step_summary(corpus),
    ]
    for out in outputs:
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

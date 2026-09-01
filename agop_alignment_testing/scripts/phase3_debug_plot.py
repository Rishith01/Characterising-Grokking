"""Phase 3 debug-set comparison plots (project.md Phase 3, "Online probes"):
train/test accuracy, AGOP alignment (the causal=False baseline), and a causal
online-probe candidate, side by side, for a small hand-picked debug set.

Debug set (all grokkers, no non-grokkers -- per instruction): x+y and x-y, each on
RFM-quadratic, RFM-gaussian, and NN. Chosen to span kernel/learner variety on the
two operations we understand best, before testing on the full corpus (project.md:
"A probe tuned on x+y at p=61 will overfit" -- this debug set is a first look, not
the evaluation).

Generates one plot per online-probe candidate explored so far, ALL in one pass, from
the current run data every time -- deliberately not "just regenerate the one metric
I'm currently looking at": with three separate output files sharing the same AGOP
alignment baseline and the same 6 runs, regenerating only the most recent one leaves
the others holding stale data the moment any underlying run changes (this happened
for real: the NN betas fix left two of the three files showing pre-fix NN curves
until this was caught by inspection and fixed).

Usage: python -m scripts.phase3_debug_plot
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from agopx import probes as probe_registry
from agopx.runner import load_run

DEBUG_SET = [
    "grok_rfm_quadratic_xplusy_seed0",
    "grok_rfm_quadratic_xminusy_seed0",
    "grok_rfm_gaussian_xplusy_seed0",
    "grok_rfm_gaussian_xminusy_seed0",
    "grok_nn_xplusy_seed0",
    "grok_nn_xminusy_seed0",
]

# (registry name, metric key, legend label, plot title suffix, output filename)
CANDIDATES = [
    ("increment_norm", "increment_norm", "||Delta_t||_F", "increment norm", "phase3_debug_increment_norm.png"),
    (
        "consecutive_agop_alignment",
        "consecutive_agop_alignment",
        "AGOP alignment (vs M_{t-1})",
        "consecutive AGOP alignment",
        "phase3_debug_consecutive_alignment.png",
    ),
    (
        "increment_coherence",
        "increment_coherence",
        "increment coherence cos(Delta_t, Delta_{t-1})",
        "increment coherence",
        "phase3_debug_increment_coherence.png",
    ),
]


def plot_one(run_id: str, ax_acc, ax_probe, probe_name: str, metric_key: str, metric_label: str):
    run_dir = Path("runs/phase2") / run_id
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    p = config["p"]
    source = "sqrt_agop" if config.get("learner", "rfm") == "nn" else "M"

    traj = load_run(run_dir)
    df_metrics = pd.DataFrame([{"t": s.t, **s.metrics} for s in traj])

    align_cls = probe_registry.get("agop_alignment")
    probe_cls = probe_registry.get(probe_name)
    align = pd.DataFrame(align_cls(p=p, source=source).finalize(traj)["trajectory"])
    metric = pd.DataFrame(probe_cls(p=p, source=source).finalize(traj)["trajectory"])

    ax_acc.plot(df_metrics["t"], df_metrics["train/accuracy"], color="tab:blue", label="train acc")
    ax_acc.plot(df_metrics["t"], df_metrics["test/accuracy"], color="tab:orange", label="test acc")
    ax_acc.set_ylim(-0.05, 1.05)

    ax_probe.plot(align["t"], align["agop_alignment"], color="tab:green", label="AGOP alignment (vs M*)")
    ax_probe.plot(metric["t"], metric[metric_key], color="tab:purple", label=metric_label)

    ax_acc.set_title(run_id, fontsize=9)
    ax_acc.set_xlabel("t")


def make_plot(probe_name: str, metric_key: str, metric_label: str, title_suffix: str, out_name: str):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    probe_axes = []
    for ax, run_id in zip(axes.flat, DEBUG_SET):
        ax_probe = ax.twinx()
        plot_one(run_id, ax, ax_probe, probe_name, metric_key, metric_label)
        probe_axes.append(ax_probe)

    lo = min(a.get_ylim()[0] for a in probe_axes)
    hi = max(a.get_ylim()[1] for a in probe_axes)
    for a in probe_axes:
        a.set_ylim(lo, hi)

    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("accuracy")
    probe_axes[2].set_ylabel("cosine similarity" if "coherence" in metric_key or "alignment" in metric_key else metric_label)
    probe_axes[5].set_ylabel("cosine similarity" if "coherence" in metric_key or "alignment" in metric_key else metric_label)

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = probe_axes[0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f"Phase 3 debug set: accuracy vs AGOP alignment (vs M*) vs {title_suffix}", y=1.06)
    fig.tight_layout()

    out = Path("runs/phase2/plots") / out_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    for probe_name, metric_key, metric_label, title_suffix, out_name in CANDIDATES:
        make_plot(probe_name, metric_key, metric_label, title_suffix, out_name)


if __name__ == "__main__":
    main()

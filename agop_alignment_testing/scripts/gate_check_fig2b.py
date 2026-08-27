"""Phase 1 gate check (project.md Section 5, Phase 1 / Section 7 next action #3):
reproduce the qualitative shape of Figure 2B for one run -- AGOP alignment rising
and circulant deviation falling while test accuracy/loss stay flat.

Usage: python -m scripts.gate_check_fig2b <run_dir>

p and operation are read from the run's own config.yaml (written by runner.run()),
so the right circulant-deviation transform (features.CIRCULANT_TRANSFORMS) is picked
automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from agopx.probes.offline import AGOPAlignment, CirculantDeviation
from agopx.runner import load_run


def gate_check(run_dir: Path, out_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    p, operation = config["p"], config["operation"]

    traj = load_run(run_dir)

    align = AGOPAlignment(p=p).finalize(traj)["trajectory"]
    dev = CirculantDeviation(p=p, operation=operation).finalize(traj)["trajectory"]

    df_metrics = pd.DataFrame([{"t": s.t, **s.metrics} for s in traj])
    df = df_metrics.merge(pd.DataFrame(align), on="t").merge(pd.DataFrame(dev), on="t")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    ax = axes[0]
    ax.plot(df["t"], df["train/accuracy"], label="train acc", color="tab:blue")
    ax.plot(df["t"], df["test/accuracy"], label="test acc", color="tab:orange")
    ax2 = ax.twinx()
    ax2.plot(df["t"], df["agop_alignment"], label="AGOP alignment", color="tab:green")
    ax.set_xlabel("t")
    ax.set_ylabel("accuracy")
    ax2.set_ylabel("rho(M_t, M*)")
    ax.set_title("accuracy vs AGOP alignment")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax = axes[1]
    ax.plot(df["t"], df["train/accuracy"], label="train acc", color="tab:blue")
    ax.plot(df["t"], df["test/accuracy"], label="test acc", color="tab:orange")
    ax2 = ax.twinx()
    ax2.plot(df["t"], df["circulant_deviation"], label="circulant deviation", color="tab:red")
    ax.set_xlabel("t")
    ax.set_ylabel("accuracy")
    ax2.set_ylabel("D(A)")
    ax.set_title("accuracy vs circulant deviation")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle(f"Phase 1 gate check: {run_dir.name} ({operation}, p={p})")
    fig.tight_layout()

    out_path = out_path or (run_dir / "gate_check.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return df, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str)
    args = parser.parse_args()

    df, out_path = gate_check(Path(args.run_dir))
    print(df.to_string())
    print(f"wrote {out_path}")

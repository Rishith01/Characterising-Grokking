"""Figures and numbers for the F2 report (report_AGOP/REPORT.md).

Usage: python report_AGOP/make_figures.py <dir with rows.parquet and series.parquet>
Those two files are on HF at advay333/tfs:phase4/agop_final/. The tuned settings and
their confidence intervals come from detailed/tables/calibration.csv, the run list from
detailed/tables/selection.csv. Writes figures/*.png and tables/*.csv, and prints the
numbers quoted in the report.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1])
FIG, TAB = HERE / "figures", HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
for old in FIG.glob("*.png"):  # numbering changes between versions
    old.unlink()

INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
BLUE, BLUE_D, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, GREY = ("#2a78d6", "#184f95", "#eb6834", "#1baf7a",
                                                            "#eda100", "#e87ba4", "#008300", "#898781")
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 10, "axes.titlecolor": INK})

# candidates in report order: (family, short label, colour)
CANDS = [("horizon", "Never stop", GREY),
         ("fixed_frac", "Fixed fraction", ORANGE),
         ("bound", "F2-A  extrapolated rotation", BLUE),
         ("rel_delta", "F2-B  rotation, growing window", BLUE),
         ("const_delta", "F2-C  rotation, fixed window", BLUE),
         ("msd", "F2-D  drift test (MSD)", BLUE),
         ("geweke", "F2-E  early-vs-late test", BLUE),
         ("F2 best", "F2, tuned pick of A-E", BLUE_D),
         ("oracle_cos", "Oracle: alignment with final", MAGENTA),
         ("oracle_settle", "Oracle: stays near final", MAGENTA),
         ("oracle_teps", "Perfect stop (test set)", INK)]
LABEL = {f: l for f, l, _ in CANDS}
COLOR = {f: c for f, _, c in CANDS}
ONLINE = ["bound", "rel_delta", "const_delta", "msd", "geweke"]
TESTS = [("modular", "seeds", "Modular, Test A (new seeds)"),
         ("modular", "operation", "Modular, Test B (new operations)"),
         ("mnist", "seeds", "MNIST, Test A (new seeds)"),
         ("mnist", "depth", "MNIST, Test B (8-layer network)")]
CATS = [("fast", "learns fast", BLUE), ("grokker", "groks (learns late)", ORANGE),
        ("joint_plateau", "train and test both stall, then rise", AQUA),
        ("late_collapse", "learns, then collapses", YELLOW), ("never", "never learns", MAGENTA),
        ("no_fit", "never fits its training set", GREEN)]
VAR_NAME = {"modular x+y": "x + y", "modular x*y": "x × y", "modular x/y": "x ÷ y",
            "mnist d3": "3-layer, 1k images", "mnist d8": "8-layer, 4k images"}

calib = pd.read_csv(HERE / "detailed" / "tables" / "calibration.csv")
sel = pd.read_csv(HERE / "detailed" / "tables" / "selection.csv").set_index("run_id")
sel["tf"] = sel["t_eps"] / sel["steps"]
rows = pd.read_parquet(DATA / "rows.parquet").join(sel[["app", "seed", "variant", "category", "steps"]], on="run_id")
series = pd.read_parquet(DATA / "series.parquet").join(sel[["app", "variant", "t_eps", "max_test", "category"]],
                                                       on="run_id")
out_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out_lines.append(s)
    print(s)


def test_ids(app, split):
    m = sel["app"] == app
    if split == "seeds":
        return set(sel.index[m & (sel["seed"] >= 2)])
    return set(sel.index[m & sel["variant"].isin(["modular x*y", "modular x/y", "mnist d8"])])


def verdict(lo, hi):
    return "better" if hi < 0 else ("worse" if lo > 0 else "tie")


def C(lam, app, split, fam):
    return calib[(calib["lambda"] == lam) & (calib["app"] == app) & (calib["split"] == split)
                 & (calib["family"] == fam)].iloc[0]


def picked_rows(lam, app, split, fam):
    r = C(lam, app, split, fam)
    return rows[rows["run_id"].isin(test_ids(app, split)) & (rows["setting"] == r["setting"])]


def style(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ==== tables ===========================================================================
# corpus: one row per (app, variant)
corp = []
for (app, var), g in sel.groupby(["app", "variant"]):
    d = {"app": app, "variant": var, "runs": len(g), "cells": g["cell"].nunique(),
         "optimizers": ", ".join(sorted(g["optim"].unique())),
         "lr": ", ".join(f"{x:g}" for x in sorted(g["lr"].unique())),
         "weight_decay": ", ".join(f"{x:g}" for x in sorted(g["gamma"].unique())),
         "init_scale": ", ".join(f"{x:g}" for x in sorted(g["alpha"].unique())),
         "train_size": ", ".join(f"{x:g}" for x in sorted(g["size"].unique()))}
    for c, _, _ in CATS:
        d[c] = int((g["category"] == c).sum())
    corp.append(d)
corp = pd.DataFrame(corp)
corp.to_csv(TAB / "corpus.csv", index=False)
say("corpus\n", corp.to_string(index=False))
say("t_eps / T by category\n", sel.groupby(["app", "category"])["tf"].describe()[["count", "25%", "50%", "75%"]].round(3))
say("best test acc by category\n", sel.groupby(["app", "category"])["max_test"].median().round(3))

# every candidate x test x lambda
summ = []
for lam in (0.03, 0.1, 0.3):
    for app, split, _ in TESTS:
        for fam, _, _ in CANDS:
            r = C(lam, app, split, fam)
            t = picked_rows(lam, app, split, fam)
            gk = t[t["category"] == "grokker"]
            summ.append({"lambda": lam, "app": app, "split": split, "family": fam, "label": LABEL[fam],
                         "setting": r["setting"], "cal_runs": r["cal_runs"], "test_runs": t["run_id"].nunique(),
                         "test_cells": r["test_cells"], "cal_score": r["cal_cost"], "score": r["test_cost"],
                         "score_lo": r["ci_lo"], "score_hi": r["ci_hi"],
                         "acc_lost_pts": 100 * t["regret"].mean(), "budget_wasted_pct": 100 * t["late"].mean(),
                         "too_early_pct": 100 * t["premature"].mean(), "never_stops_pct": 100 * t["never"].mean(),
                         "grokkers": len(gk), "too_early_grokkers_pct": 100 * gk["premature"].mean() if len(gk) else np.nan,
                         "vs_never": r["d_horizon"], "vs_never_lo": r["d_horizon_lo"], "vs_never_hi": r["d_horizon_hi"],
                         "verdict_vs_never": verdict(r["d_horizon_lo"], r["d_horizon_hi"]) if fam != "horizon" else "",
                         "vs_fixed": r["d_fixed"], "vs_fixed_lo": r["d_fixed_lo"], "vs_fixed_hi": r["d_fixed_hi"],
                         "verdict_vs_fixed": verdict(r["d_fixed_lo"], r["d_fixed_hi"]) if fam != "fixed_frac" else "",
                         "cells_better_than_fixed": r["cells_better_f"], "cells_worse_than_fixed": r["cells_worse_f"]})
summ = pd.DataFrame(summ)
summ.round(4).to_csv(TAB / "summary.csv", index=False)
show = ["label", "setting", "score", "score_lo", "score_hi", "acc_lost_pts", "budget_wasted_pct", "too_early_pct",
        "never_stops_pct", "too_early_grokkers_pct", "vs_never", "verdict_vs_never", "vs_fixed", "vs_fixed_lo",
        "vs_fixed_hi", "verdict_vs_fixed"]
for lam in (0.1, 0.03, 0.3):
    for app, split, name in TESTS:
        s = summ[(summ["lambda"] == lam) & (summ["app"] == app) & (summ["split"] == split)]
        say(f"\n--- lambda {lam}: {name}  (cal {s['cal_runs'].iloc[0]}, test {s['test_runs'].iloc[0]} runs, "
            f"{s['test_cells'].iloc[0]} cells)")
        say(s[show].round(3).to_string(index=False))

# outcome-category breakdown at lambda 0.1
cat_rows = []
for app, split, _ in TESTS:
    for fam in ("horizon", "fixed_frac", "const_delta", "F2 best", "oracle_cos"):
        t = picked_rows(0.1, app, split, fam)
        for cat, g in t.groupby("category"):
            cat_rows.append({"app": app, "split": split, "family": fam, "category": cat, "runs": len(g),
                             "score": g["cost_0.1"].mean(), "acc_lost_pts": 100 * g["regret"].mean(),
                             "budget_wasted_pct": 100 * g["late"].mean(), "too_early_pct": 100 * g["premature"].mean()})
cat_tab = pd.DataFrame(cat_rows)
cat_tab.round(4).to_csv(TAB / "by_outcome.csv", index=False)

# ==== Figure 1: the corpus =============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
for i, app in enumerate(("modular", "mnist")):
    ax = axes[i, 0]
    vs = [v for v in VAR_NAME if v.startswith(app)]
    for k, v in enumerate(vs):
        g = sel[sel["variant"] == v]
        left = 0
        for c, lab, col in CATS:
            n = int((g["category"] == c).sum())
            if n:
                ax.barh(k, n, left=left, color=col, height=0.62, edgecolor=BG, lw=1.5)
                if n >= 3:
                    ax.text(left + n / 2, k, str(n), ha="center", va="center", fontsize=8.5,
                            color=INK if col == YELLOW else "#ffffff")
                left += n
        ax.text(left + 1, k, f"{left} runs", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(vs)), [VAR_NAME[v] for v in vs])
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("number of runs")
    ax.set_title(f"{'Modular arithmetic' if app == 'modular' else 'MNIST'}: what happened in each run", loc="left")
    style(ax)
    ax = axes[i, 1]
    g = sel[(sel["app"] == app) & (sel["category"] != "never")]
    present = [c for c in CATS if (g["category"] == c[0]).any()]
    rng = np.random.default_rng(0)
    for k, (c, lab, col) in enumerate(present):
        x = np.clip(g.loc[g["category"] == c, "tf"].to_numpy(), 1e-3, 1)
        ax.scatter(x, k + rng.uniform(-0.22, 0.22, len(x)), s=16, color=col, alpha=0.8, lw=0)
        ax.plot([np.median(x)] * 2, [k - 0.32, k + 0.32], color=INK, lw=1.4)
    ax.set_yticks(range(len(present)), [f"{lab} ({int((g['category'] == c).sum())})" for c, lab, _ in present])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(8e-4, 1.3)
    ax.set_xlabel("when test accuracy stopped improving (share of the budget)")
    ax.set_title("When learning finished (dot = one run, tick = median)", loc="left")
    style(ax)
handles = [Patch(color=col, label=lab) for _, lab, col in CATS]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.03))
fig.text(0.01, 0.005, "Right: runs that never learn are left out, since they have no moment when learning finished.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "01_corpus.png", dpi=150)
plt.close(fig)

# ==== Figure 2: the score, explained on one run ========================================
gk = sel[(sel["app"] == "modular") & (sel["category"] == "grokker") & sel["tf"].between(0.3, 0.45)
         & (sel["seed"] >= 2)].sort_values("tf")
rid = gk.index[len(gk) // 2]
d = series[series["run_id"] == rid]
T, te = float(sel.loc[rid, "steps"]), float(sel.loc[rid, "t_eps"])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, (stop, title) in zip(axes, ((0.15 * T, "Stopping too early: accuracy is lost"),
                                    (0.85 * T, "Stopping too late: budget is wasted"))):
    ax.plot(d["step"] / T, d["test_acc"], color=AQUA, lw=2, label="test accuracy")
    ax.axvline(te / T, color=INK, lw=0.8, ls=":")
    ax.text(te / T + 0.01, 0.5, "test accuracy\nstops improving", fontsize=8, color=INK2, va="center")
    acc = float(np.interp(stop, d["step"], d["test_acc"]))
    ax.axvline(stop / T, color=BLUE, lw=1.5)
    ax.text(stop / T + 0.01, 0.06, "the rule stops here", fontsize=8, color=BLUE)
    if stop < te:
        ax.annotate("", xy=(stop / T, d["test_acc"].max()), xytext=(stop / T, acc),
                    arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
        ax.text(stop / T + 0.01, (acc + 1) / 2, f"accuracy lost\n= {100 * (d['test_acc'].max() - acc):.0f} points",
                fontsize=8.5, color=ORANGE, va="center")
    else:
        ax.annotate("", xy=(te / T, 0.3), xytext=(stop / T, 0.3), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.5))
        ax.text((te / T + stop / T) / 2, 0.34, f"budget wasted = {stop / T - te / T:.0%}\nscore adds 0.1 × {stop / T - te / T:.2f}",
                fontsize=8.5, color=ORANGE, ha="center")
    ax.set_xlim(0, 1)
    ax.set_title(title, loc="left")
    ax.set_xlabel("training time, as a share of the budget")
    style(ax, "both")
axes[0].set_ylabel("test accuracy")
fig.tight_layout()
fig.savefig(FIG / "02_score_explained.png", dpi=150)
plt.close(fig)
say("score example run:", rid, "t_eps/T", te / T)


# ==== Figure 3 and 4: held-out score of every candidate, and its difference from fixed =
def forest(metric, lo, hi, path, xlabel, ref0, fams, note, lam=0.1):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=False)
    for ax, (app, split, name) in zip(axes.flat, TESTS):
        s = summ[(summ["lambda"] == lam) &(summ["app"] == app) & (summ["split"] == split)].set_index("family")
        for k, fam in enumerate(fams):
            r = s.loc[fam]
            ax.plot([r[lo], r[hi]], [k, k], color=COLOR[fam], lw=1.6)
            ax.plot(r[metric], k, "o", color=COLOR[fam], ms=6.5, mec=BG, mew=1.2)
            txt = f"{r[metric]:+.3f}" if ref0 else f"{r[metric]:.3f}"
            ax.text(max(r[hi], r[metric]) + 0.006, k, txt, va="center", fontsize=8, color=INK2)
        if ref0:
            ax.axvline(0, color=ORANGE, lw=1.2)
        else:
            ax.axvline(s.loc["fixed_frac", metric], color=ORANGE, lw=0.8, ls="--")
            ax.axvline(s.loc["horizon", metric], color=GREY, lw=0.8, ls="--")
        ax.set_yticks(range(len(fams)), [LABEL[f] for f in fams], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(f"{name}: {int(s['test_runs'].iloc[0])} held-out runs", loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        style(ax)
        xl = ax.get_xlim()
        ax.set_xlim(xl[0], xl[1] + 0.03)
    fig.supxlabel(xlabel, fontsize=9.5, y=0.035)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


for lam, fname in ((0.1, "03_scores.png"), (0.03, "04a_scores_lambda_0.03.png"), (0.3, "04b_scores_lambda_0.3.png")):
    forest("score", "score_lo", "score_hi", FIG / fname,
           f"score on held-out runs at λ = {lam} (lower is better); every rule re-tuned at this λ", False,
           [f for f, _, _ in CANDS],
           "Dot: average score on runs the rule was not tuned on. Line: 95% confidence interval. Dashed lines: never "
           "stop (grey) and fixed fraction (orange). Verdicts use paired differences, not these overlaps.",
           lam)

# ==== Figure 5: where each rule stops ==================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
for ax, app in zip(axes, ("modular", "mnist")):
    ids = test_ids(app, "seeds")
    ok = sel.loc[list(ids)]
    ok = ok[(ok["max_test"] >= 0.5) & (ok["t_eps"] > 0)]
    fx = C(0.1, app, "seeds", "fixed_frac")
    c = float(fx["setting"].split("=")[1])
    for fam, mk in (("F2 best", "o"), ("oracle_cos", "D")):
        t = picked_rows(0.1, app, "seeds", fam).set_index("run_id").loc[ok.index]
        x = np.clip(ok["tf"].to_numpy(), 1e-3, 1)
        st = t["stop"].to_numpy(dtype=float) / ok["steps"].to_numpy()
        nev = ~np.isfinite(st)
        y = np.where(nev, 1.0, np.clip(st, 1e-3, 1))
        ax.scatter(x[~nev], y[~nev], s=26 if mk == "o" else 18, marker=mk, color=COLOR[fam], alpha=0.85, lw=0,
                   label=f"{LABEL[fam]}")
        ax.scatter(x[nev], y[nev] * (1.12 if mk == "o" else 1.25), s=26 if mk == "o" else 18, marker=mk,
                   facecolor="none", edgecolor=COLOR[fam], lw=1)
    ax.axhline(c, color=ORANGE, lw=2, label=f"Fixed fraction (stops at {c:g})")
    lim = (8e-4, 1.5)
    ax.plot(lim, lim, color=INK, lw=0.8, ls=":")
    ax.fill_between(lim, [lim[0]] * 2, lim, color=MAGENTA, alpha=0.06, lw=0)
    ax.text(0.3, 2e-3, "below the dotted line:\nstopped too early", fontsize=8, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("when test accuracy stopped improving (share of budget)")
    ax.set_title(f"{'Modular' if app == 'modular' else 'MNIST'}, Test A: {len(ok)} held-out runs that learned",
                 loc="left")
    style(ax, "both")
axes[0].set_ylabel("when the rule stopped (share of budget)")
h5, l5 = axes[0].get_legend_handles_labels()
fig.legend(h5, l5, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04))
fig.text(0.01, 0.005, "Hollow markers at the top: the rule never stopped and trained to the end of the budget.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIG / "05_stop_times.png", dpi=150)
plt.close(fig)

# ==== Figure 6: stopped too early, by outcome ==========================================
fams6 = ["fixed_frac"] + ONLINE + ["F2 best", "oracle_cos"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
axes[0].invert_yaxis()
for ax, (app, split, name) in zip(axes, TESTS[:2]):
    s = summ[(summ["lambda"] == 0.1) & (summ["app"] == app) & (summ["split"] == split)].set_index("family")
    for k, fam in enumerate(fams6):
        a, b = s.loc[fam, "too_early_pct"], s.loc[fam, "too_early_grokkers_pct"]
        ax.barh(k - 0.19, a, height=0.36, color=COLOR[fam], alpha=0.45, edgecolor=BG)
        ax.barh(k + 0.19, b, height=0.36, color=COLOR[fam], edgecolor=BG)
        ax.text(a + 0.8, k - 0.19, f"{a:.0f}%", va="center", fontsize=7.5, color=INK2)
        ax.text(b + 0.8, k + 0.19, f"{b:.0f}%", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(fams6)), [LABEL[f] for f in fams6], fontsize=8.5)
    ax.set_xlim(0, 60)
    ng = int(s.loc["fixed_frac", "grokkers"])
    ax.set_title(f"{name}: {ng} of the held-out runs grok", loc="left")
    ax.set_xlabel("share of runs stopped before test accuracy finished improving")
    style(ax)
axes[0].legend(handles=[Patch(color=GREY, alpha=0.45, label="all held-out runs"),
                        Patch(color=GREY, label="runs that grok")], frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "06_too_early.png", dpi=150)
plt.close(fig)

# ==== Figure 7: two example runs ========================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
picks = {}
for app in ("modular", "mnist"):
    f2 = C(0.1, app, "seeds", "const_delta")["setting"]
    ok = []
    for r in sel[(sel["app"] == app) & (sel["seed"] >= 2) & (sel["max_test"] >= 0.8)].index:
        st = rows.loc[(rows["run_id"] == r) & (rows["setting"] == f2), "stop"]
        d = series[series["run_id"] == r]
        early = d.loc[d["step"] < sel.loc[r, "t_eps"] / 2, "align_M"]
        if len(st) and pd.notna(st.iloc[0]) and (early.max() if len(early) else 0) < 0.5:
            ok.append((sel.loc[r, "category"] != "grokker", st.iloc[0] / max(sel.loc[r, "t_eps"], 1), r))
    ok.sort()
    grok = [o for o in ok if not o[0]] or ok
    picks[app] = grok[len(grok) // 2][2]
for ax, app in zip(axes, ("modular", "mnist")):
    r = picks[app]
    d = series[series["run_id"] == r]
    ax.plot(d["step"], d["test_acc"], color=AQUA, lw=2, label="test accuracy")
    ax.plot(d["step"], d["align_M"], color=BLUE, lw=2, label="alignment of the features with their final state")
    ax.axvline(sel.loc[r, "t_eps"], color=INK, lw=0.8, ls=":", label="test accuracy stops improving")
    ax.set_xscale("log")
    ax.set_title(f"{'Modular' if app == 'modular' else 'MNIST'}: one typical grokking run "
                 f"({VAR_NAME[sel.loc[r, 'variant']]})", loc="left")
    ax.set_xlabel("training step")
    style(ax, "both")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "07_example_runs.png", dpi=150)
plt.close(fig)
say("example runs:", picks)

# ==== Figure 8: features vs accuracy, medians over runs ===============================
s8 = series[(series["t_eps"] > 0) & (series["max_test"] >= 0.5)].copy()
s8["x"] = np.log10(np.maximum(s8["step"], 1) / s8["t_eps"])
s8["acc"] = s8["test_acc"] / s8["max_test"]
groups = [("Modular (all three operations)", s8["app"] == "modular"),
          ("MNIST, 3-layer network", s8["variant"] == "mnist d3"),
          ("MNIST, 8-layer network", s8["variant"] == "mnist d8")]
bins = np.arange(-2.0, 1.55, 0.1)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, (title, m) in zip(axes, groups):
    d = s8[m]
    n_runs = d["run_id"].nunique()
    xs = np.arange(-2.0, 1.51, 0.05)
    for col, label, color in (("acc", "test accuracy (share of its best)", AQUA),
                              ("align_M", "alignment with the final features", BLUE)):
        Y = np.full((n_runs, len(xs)), np.nan)
        for k, (_, g) in enumerate(d.groupby("run_id")):
            g = g[g["step"] > 0]
            x, y = g["x"].to_numpy(), g[col].to_numpy()
            inside = (xs >= x.min()) & (xs <= x.max())
            Y[k, inside] = np.interp(xs[inside], x, y)
        ok = np.isfinite(Y).sum(0) >= 0.5 * n_runs
        med, lo, hi = (np.nanpercentile(Y[:, ok], q, axis=0) for q in (50, 25, 75))
        ax.fill_between(10 ** xs[ok], lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(10 ** xs[ok], med, color=color, lw=2, label=label)
    ax.axvline(1, color=INK, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_title(f"{title} ({n_runs} runs that learned)", loc="left")
    ax.set_xlabel("training time ÷ time when test accuracy\nstopped improving (1 = that moment)")
    style(ax, "both")
axes[0].set_ylabel("median over runs (band: middle half)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "08_features_vs_accuracy.png", dpi=150)
plt.close(fig)

diag = []
for r, d in series.groupby("run_id"):
    info = sel.loc[r]
    if not (info["t_eps"] > 0 and info["max_test"] >= 0.5):
        continue
    i = min(int(np.searchsorted(d["step"].to_numpy(), info["t_eps"])), len(d) - 1)
    w = d["w_norm"].to_numpy()
    e = d["erank_G"].to_numpy()
    diag.append({"run_id": r, "variant": info["variant"], "app": info["app"],
                 "align_at_teps": d["align_M"].iloc[i],
                 "weight_norm_ratio_after": w[-1] / w[i] if w[i] > 0 else np.nan,
                 "erank_ratio_after": e[-1] / e[i] if e[i] > 0 else np.nan})
diag = pd.DataFrame(diag)
diag["group"] = np.where(diag["app"] == "modular", "modular", diag["variant"])
say("\nfeature diagnostics at t_eps (medians)\n",
    diag.groupby("group")[["align_at_teps", "weight_norm_ratio_after", "erank_ratio_after"]].median().round(2),
    diag.groupby("group").size())

# ==== Figure 9: the oracle at every threshold ==========================================
fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
ths = ["0.9", "0.95", "0.98", "0.99", "0.995", "0.999"]
for ax, (app, split, name) in zip(axes, TESTS):
    ids = test_ids(app, split)
    t = rows[rows["run_id"].isin(ids)]
    for obj, col, ls in (("M", MAGENTA, "-"), ("G", MAGENTA, "--"), ("W", MAGENTA, ":")):
        y = [t.loc[t["setting"] == f"oracle_cos {obj} c={c}", "cost_0.1"].mean() for c in ths]
        ax.plot(range(len(ths)), y, color=col, ls=ls, lw=1.8, marker="o", ms=4,
                label={"M": "features M", "G": "AGOP G", "W": "weights"}[obj])
    hz = t.loc[t["setting"] == "horizon", "cost_0.1"].mean()
    fxs = [t.loc[t["setting"] == f"fixed_frac c={c}", "cost_0.1"].mean()
           for c in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)]
    ax.axhline(hz, color=GREY, lw=1.2)
    ax.axhline(min(fxs), color=ORANGE, lw=1.2)
    ax.text(len(ths) - 1, hz, "never stop", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.text(0, min(fxs), "best fixed fraction", fontsize=7.5, color=ORANGE, va="bottom")
    ax.set_xticks(range(len(ths)), ths)
    ax.set_xlabel("stop once alignment with the final ≥")
    ax.set_title(name.replace(" (", "\n("), loc="left", fontsize=9.5)
    style(ax, "y")
axes[0].set_ylabel("score on held-out runs")
h9, l9 = axes[0].get_legend_handles_labels()
fig.legend(h9, ["alignment of the features M", "alignment of the AGOP G", "alignment of the weights"], frameon=False,
           fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.05))
fig.text(0.01, 0.005, "Every threshold is shown, so these are the oracle's best possible scores in hindsight; the "
         "orange line is likewise the best fixed fraction in hindsight.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(FIG / "09_oracle_all_thresholds.png", dpi=150)
plt.close(fig)
for app, split, name in TESTS:
    t = rows[rows["run_id"].isin(test_ids(app, split))]
    m = t.groupby("setting")["cost_0.1"].mean()
    say(f"hindsight best, {name}: oracle_cos {m[m.index.str.startswith('oracle_cos')].min():.3f}  "
        f"fixed {m[m.index.str.startswith('fixed_frac')].min():.3f}  never {m['horizon']:.3f}")

# ==== Figure 10: is the online measurement faithful? ===================================
fig, ax = plt.subplots(figsize=(6.6, 4.6))
sp = []
for title, m, color in (("Modular", series["app"] == "modular", BLUE),
                        ("MNIST 3-layer", series["variant"] == "mnist d3", ORANGE),
                        ("MNIST 8-layer", series["variant"] == "mnist d8", AQUA)):
    d = series[m & np.isfinite(series["theta_const"]) & np.isfinite(series["true_M"]) & (series["theta_const"] > 0)]
    b = np.logspace(-3.5, 0.2, 20)
    d = d.assign(b=pd.cut(d["theta_const"], b))
    g = d.groupby("b", observed=True)["true_M"].agg(["median", lambda v: v.quantile(0.25),
                                                       lambda v: v.quantile(0.75), "size"])
    g.columns = ["med", "lo", "hi", "n"]
    g = g[g["n"] >= 30]
    xc = np.array([np.sqrt(iv.left * iv.right) for iv in g.index])
    rho = d[["theta_const", "true_M"]].corr(method="spearman").iloc[0, 1]
    per = d.groupby("run_id")[["theta_const", "true_M"]].apply(
        lambda x: x.corr(method="spearman").iloc[0, 1] if len(x) >= 8 else np.nan).median()
    sp.append((title, rho, per))
    ax.fill_between(xc, g["lo"], g["hi"], color=color, alpha=0.15, lw=0)
    ax.plot(xc, g["med"], color=color, lw=2, label=f"{title}: rank correlation {rho:.2f} pooled, {per:.2f} within a run")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("what F2-C measures during training: how far the features\nrotated over the last 2% of the budget (radians)")
ax.set_ylabel("how far the features still are\nfrom their final state (radians)")
style(ax, "both")
ax.legend(frameon=False, fontsize=7.8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "10_online_vs_final.png", dpi=150)
plt.close(fig)
say("spearman (pooled, median within run):", sp)
dg = series.dropna(subset=["R_default", "true_M"])
for title, m in (("modular", dg["app"] == "modular"), ("mnist d3", dg["variant"] == "mnist d3"),
                 ("mnist d8", dg["variant"] == "mnist d8")):
    d = dg[m & np.isfinite(dg["R_default"]) & (dg["R_default"] > 0)]
    per = d.groupby("run_id")[["R_default", "true_M"]].apply(
        lambda x: x.corr(method="spearman").iloc[0, 1] if len(x) >= 8 else np.nan).median()
    say(f"F2-A bound spearman {title}: pooled {d[['R_default', 'true_M']].corr(method='spearman').iloc[0, 1]:.2f}, "
        f"within run {per:.2f}")

# ==== Figure 11: does it depend on lambda? ============================================
fams11 = ONLINE + ["F2 best", "oracle_cos", "oracle_settle"]
cols = [(app, split, lam) for app, split, _ in TESTS for lam in (0.03, 0.1, 0.3)]
fig, ax = plt.subplots(figsize=(13, 4.6))
vcol = {"better": BLUE, "tie": "#f0efec", "worse": ORANGE}
for i, fam in enumerate(fams11):
    for j, (app, split, lam) in enumerate(cols):
        r = summ[(summ["lambda"] == lam) & (summ["app"] == app) & (summ["split"] == split)
                 & (summ["family"] == fam)].iloc[0]
        v = r["verdict_vs_fixed"]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=vcol[v], edgecolor=BG, lw=2))
        ax.text(j + 0.5, i + 0.5, f"{r['vs_fixed']:+.3f}", ha="center", va="center", fontsize=7.8,
                color="#ffffff" if v != "tie" else INK)
ax.set_xlim(0, len(cols))
ax.set_ylim(len(fams11), 0)
ax.set_yticks(np.arange(len(fams11)) + 0.5, [LABEL[f] for f in fams11], fontsize=8.5)
ax.set_xticks(np.arange(len(cols)) + 0.5, [f"λ={lam}" for _, _, lam in cols], fontsize=8)
for k, (_, _, name) in enumerate(TESTS):
    ax.text(3 * k + 1.5, -0.35, name.replace(" (", "\n("), ha="center", va="bottom", fontsize=8.5, color=INK)
    if k:
        ax.axvline(3 * k, color=INK2, lw=1)
ax.tick_params(length=0)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
ax.legend(handles=[Patch(color=BLUE, label="clearly better than the fixed fraction"),
                   Patch(facecolor="#f0efec", edgecolor=GRID, label="tie (the 95% interval includes 0)"),
                   Patch(color=ORANGE, label="clearly worse than the fixed fraction")],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "11_lambda.png", dpi=150)
plt.close(fig)

# ==== Figure 12: does the tuning carry over? ==========================================
fig, ax = plt.subplots(figsize=(9.0, 5.2))
mk = {"horizon": "s", "fixed_frac": "s", "oracle_cos": "D", "oracle_settle": "D"}
tcol = [BLUE, AQUA, ORANGE, MAGENTA]
for (app, split, name), col in zip(TESTS, tcol):
    s12 = summ[(summ["lambda"] == 0.1) & (summ["app"] == app) & (summ["split"] == split)
               & ~summ["family"].isin(["oracle_teps", "F2 best"])]
    for _, r in s12.iterrows():
        ax.scatter(r["cal_score"], r["score"], marker=mk.get(r["family"], "o"), s=42, color=col, alpha=0.9, lw=0)
    far = s12[s12["score"] - s12["cal_score"] > 0.05].sort_values("score")
    for n, (_, r) in enumerate(far.iterrows()):
        ax.annotate(LABEL[r["family"]].split("  ")[0], (r["cal_score"], r["score"]),
                    xytext=(r["cal_score"] + 0.035, r["score"] + 0.012 * (n - (len(far) - 1) / 2)),
                    fontsize=7.5, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
lim = (0, 0.24)
ax.plot(lim, lim, color=INK, lw=0.8, ls=":")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel("score on the runs it was tuned on")
ax.set_ylabel("score on the held-out runs")
style(ax, "both")
h = [Line2D([], [], color=c, marker="o", ls="", label=n) for (_, _, n), c in zip(TESTS, tcol)]
h += [Line2D([], [], color=INK2, marker=m, ls="", label=l) for m, l in
      (("o", "F2 variant A-E"), ("s", "never stop / fixed fraction"), ("D", "oracle"))]
h += [Line2D([], [], color=INK, ls=":", label="same score on both")]
ax.legend(handles=h, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(FIG / "12_tuned_vs_heldout.png", dpi=150)
plt.close(fig)

(TAB / "numbers.txt").write_text("\n".join(out_lines), encoding="utf-8")

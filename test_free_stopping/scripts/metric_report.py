"""Figures and tables for the plain-language reports of study A (report_N1/, report_R1/).

Usage: python scripts/metric_report.py <dir> <N1|R1|B5>
<dir> is a download of advay333/tfs:phase4/metrics_a/ (rows.parquet, series.parquet,
selection.csv, traces/, report_<M>/tables/calibration.csv). Writes report_<M>/figures/,
report_<M>/tables/ (summary.csv, by_outcome.csv, corpus.csv, numbers.txt).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stopping import metric_study as ms  # noqa: E402
from stopping.agop_final import _cluster_ci  # noqa: E402

DATA, METRIC = Path(sys.argv[1]), sys.argv[2]
OUT = ROOT / f"report_{METRIC}"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
for old in FIG.glob("*.png"):
    old.unlink()

INK, INK2, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"
BLUE, BLUE_D, BLUE_L, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, GREY = (
    "#2a78d6", "#184f95", "#86b6ef", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#898781")
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "axes.titlesize": 10, "axes.titlecolor": INK})

FAM_A, FAM_B = ms.METRICS[METRIC]
BEST = f"{METRIC} best"
NAMES = {
    "N1": {"n1_abs": "N1-A  norm has stopped changing", "n1_shrink": "N1-B  norm has stopped shrinking"},
    "R1": {"r1": "R1-A  rank has stopped changing", "r1_guard": "R1-B  same, after the rank has fallen"},
    "B5": {"b5_level": "B5-A  alpha below a level", "b5_settle": "B5-B  alpha has stopped changing"},
}[METRIC]
CANDS = [("horizon", "Never stop", GREY), ("fixed_frac", "Fixed fraction", ORANGE),
         (FAM_A, NAMES[FAM_A], BLUE), (FAM_B, NAMES[FAM_B], BLUE_L),
         (BEST, f"{METRIC}, tuned pick of A and B", BLUE_D), ("oracle_teps", "Perfect stop (test set)", INK)]
LABEL = {f: l for f, l, _ in CANDS}
COLOR = {f: c for f, _, c in CANDS}
TESTS = [("modular", "seeds", "Modular, Test A (new seeds)"),
         ("modular", "operation", "Modular, Test B (new operations)"),
         ("mnist", "seeds", "MNIST, Test A (new seeds)"),
         ("mnist", "depth", "MNIST, Test B (8-layer network)"),
         ("teacher_student", "seeds", "Teacher-student, Test A (new seeds)"),
         ("teacher_student", "size", "Teacher-student, Test B (N 50, 100 -> 400)")]
APP_NAME = {"modular": "Modular arithmetic", "mnist": "MNIST", "teacher_student": "Teacher-student"}
CATS = [("fast", "learns fast", BLUE), ("grokker", "groks (learns late)", ORANGE),
        ("joint_plateau", "train and test both stall, then rise", AQUA),
        ("late_collapse", "learns, then collapses", YELLOW), ("never", "never learns", MAGENTA),
        ("no_fit", "never fits its training set", GREEN)]
VAR_NAME = {"modular x+y": "x + y", "modular x*y": "x × y", "modular x/y": "x ÷ y",
            "mnist d3": "3-layer, 1k images", "mnist d8": "8-layer, 4k images",
            "teacher_student n50": "N = 50", "teacher_student n100": "N = 100", "teacher_student n400": "N = 400"}
SIGNAL = {"N1": "weight norm", "R1": "effective rank", "B5": "HTSR alpha"}[METRIC]

calib = pd.read_csv(DATA / f"report_{METRIC}" / "tables" / "calibration.csv")
sel = pd.read_csv(DATA / "selection.csv").set_index("run_id")
sel["tf"] = sel["t_eps"] / sel["steps"]
rows = pd.read_parquet(DATA / "rows.parquet")
rows = rows[rows["metric"].isin([METRIC, "ref"])].join(sel[["app", "seed", "variant", "category", "steps"]],
                                                       on="run_id")
series = pd.read_parquet(DATA / "series.parquet").join(sel[["app", "variant", "t_eps", "max_test", "category",
                                                            "steps"]], on="run_id")
out_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    out_lines.append(s)
    print(s)


def C(lam, app, split, fam):
    return calib[(calib["lambda"] == lam) & (calib["app"] == app) & (calib["split"] == split)
                 & (calib["family"] == fam)].iloc[0]


def test_ids(app, split):
    return ms.split_ids(sel.reset_index(), app, split)[1]


def picked(lam, app, split, fam):
    r = C(lam, app, split, fam)
    return rows[rows["run_id"].isin(test_ids(app, split)) & (rows["setting"] == r["setting"])]


def verdict(lo, hi):
    return "better" if hi < 0 else ("worse" if lo > 0 else "tie")


def style(ax, grid="x"):
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def trace(rid):
    return pd.read_parquet(DATA / "traces" / f"{rid}.parquet")


# ==== tables ===========================================================================
corp = []
for (app, var), g in sel.groupby(["app", "variant"]):
    d = {"app": app, "variant": var, "runs": len(g), "cells": g["cell"].nunique(),
         "optimizers": ", ".join(sorted(g["optim"].unique())),
         "lr": ", ".join(f"{x:g}" for x in sorted(g["lr"].unique())),
         "weight_decay": ", ".join(f"{x:g}" for x in sorted(g["gamma"].unique())),
         "init_scale": ", ".join(f"{x:g}" for x in sorted(g["alpha"].unique())),
         "train_size": ", ".join(f"{x:g}" for x in sorted(g["size"].dropna().unique()))}
    for c, _, _ in CATS:
        d[c] = int((g["category"] == c).sum())
    corp.append(d)
corp = pd.DataFrame(corp)
corp.to_csv(TAB / "corpus.csv", index=False)
say("corpus\n", corp.to_string(index=False))
say("t_eps / T by category\n", sel.groupby(["app", "category"])["tf"].describe()[["count", "25%", "50%", "75%"]].round(3))
say("best test acc by category\n", sel.groupby(["app", "category"])["max_test"].median().round(3))

summ = []
for lam in (0.03, 0.1, 0.3):
    for app, split, _ in TESTS:
        for fam, _, _ in CANDS:
            r = C(lam, app, split, fam)
            t = picked(lam, app, split, fam)
            gk = t[t["category"] == "grokker"]
            summ.append({"lambda": lam, "app": app, "split": split, "family": fam, "label": LABEL[fam],
                         "picked_family": r["picked_family"], "setting": r["setting"], "cal_runs": r["cal_runs"],
                         "test_runs": t["run_id"].nunique(), "test_cells": r["test_cells"],
                         "cal_score": r["cal_cost"], "score": r["test_cost"], "score_lo": r["ci_lo"],
                         "score_hi": r["ci_hi"], "acc_lost_pts": 100 * t["regret"].mean(),
                         "budget_wasted_pct": 100 * t["late"].mean(), "too_early_pct": 100 * t["premature"].mean(),
                         "never_stops_pct": 100 * t["never"].mean(), "grokkers": len(gk),
                         "grokkers_too_early": int(gk["premature"].sum()),
                         "vs_never": r["d_horizon"], "vs_never_lo": r["d_horizon_lo"],
                         "vs_never_hi": r["d_horizon_hi"],
                         "verdict_vs_never": verdict(r["d_horizon_lo"], r["d_horizon_hi"]) if fam != "horizon" else "",
                         "vs_fixed": r["d_fixed"], "vs_fixed_lo": r["d_fixed_lo"], "vs_fixed_hi": r["d_fixed_hi"],
                         "verdict_vs_fixed": verdict(r["d_fixed_lo"], r["d_fixed_hi"]) if fam != "fixed_frac" else "",
                         "cells_better_than_fixed": r["cells_better_f"], "cells_worse_than_fixed": r["cells_worse_f"],
                         "hindsight_setting": r["hindsight_setting"], "hindsight_score": r["hindsight_cost"]})
summ = pd.DataFrame(summ)
summ.round(4).to_csv(TAB / "summary.csv", index=False)
show = ["label", "setting", "cal_score", "score", "score_lo", "score_hi", "acc_lost_pts", "budget_wasted_pct",
        "too_early_pct", "never_stops_pct", "grokkers_too_early", "grokkers", "vs_never", "verdict_vs_never",
        "vs_fixed", "vs_fixed_lo", "vs_fixed_hi", "verdict_vs_fixed"]
for lam in (0.1, 0.03, 0.3):
    for app, split, name in TESTS:
        s = summ[(summ["lambda"] == lam) & (summ["app"] == app) & (summ["split"] == split)]
        say(f"\n--- lambda {lam}: {name} (tune {s['cal_runs'].iloc[0]}, held out {s['test_runs'].iloc[0]} runs, "
            f"{s['test_cells'].iloc[0]} settings)")
        say(s[show].round(3).to_string(index=False))

cat_rows = []
for app, split, _ in TESTS:
    for fam in ("horizon", "fixed_frac", BEST):
        t = picked(0.1, app, split, fam)
        for cat, g in t.groupby("category"):
            cat_rows.append({"app": app, "split": split, "family": fam, "category": cat, "runs": len(g),
                             "score": g["cost_0.1"].mean(), "acc_lost_pts": 100 * g["regret"].mean(),
                             "budget_wasted_pct": 100 * g["late"].mean(),
                             "too_early_pct": 100 * g["premature"].mean(), "never_stops_pct": 100 * g["never"].mean()})
cat_tab = pd.DataFrame(cat_rows)
cat_tab.round(4).to_csv(TAB / "by_outcome.csv", index=False)
say("\nby outcome, lambda 0.1\n", cat_tab.round(3).to_string(index=False))

# every fixed fraction against the tuned pick, held-out runs, lambda 0.1
fx_rows = []
for app, split, name in TESTS:
    ids = test_ids(app, split)
    t = rows[rows["run_id"].isin(ids)]
    piv = t.pivot_table(index="run_id", columns="setting", values="cost_0.1", aggfunc="first")
    cl = sel["cell"].reindex(piv.index).to_numpy()
    v = piv[C(0.1, app, split, BEST)["setting"]]
    for c in ms.FIXED_FRACS:
        d = (v - piv[f"fixed_frac c={c}"]).to_numpy()
        lo, hi = _cluster_ci(d, cl)
        fx_rows.append({"app": app, "split": split, "c": c, "fixed_score": piv[f"fixed_frac c={c}"].mean(),
                        "pick_score": v.mean(), "diff": np.nanmean(d), "lo": lo, "hi": hi, "verdict": verdict(lo, hi)})
fx = pd.DataFrame(fx_rows)
fx.round(4).to_csv(TAB / "vs_every_fixed_fraction.csv", index=False)
say("\ntuned pick against every fixed fraction (lambda 0.1)\n", fx.round(3).to_string(index=False))

# ==== Figure 1: the corpus =============================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 9.0), gridspec_kw={"width_ratios": [1, 1.25]})
for i, app in enumerate(("modular", "mnist", "teacher_student")):
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
    ax.set_title(f"{APP_NAME[app]}: what happened in each run", loc="left")
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
    ax.set_title("When learning finished (dot = one run, tick = median)", loc="left")
    style(ax)
axes[-1, 0].set_xlabel("number of runs")
axes[-1, 1].set_xlabel("when test accuracy stopped improving (share of the budget)")
fig.legend(handles=[Patch(color=col, label=lab) for _, lab, col in CATS], loc="lower center", ncol=6,
           frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.025))
fig.text(0.01, 0.005, "Right: runs that never learn are left out, since they have no moment when learning finished.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    ax.plot(d["step"] / T, d["test_acc"], color=AQUA, lw=2)
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


# ==== signal per run ===================================================================
def signal_frame(rid):
    """step, test_acc, and the metric's signal(s) on the summary grid, from the trace."""
    tr = trace(rid)
    s = tr[tr["summary"]].reset_index(drop=True)
    out = pd.DataFrame({"step": s["step"]})
    cur = series[series["run_id"] == rid].set_index("step")["test_acc"]
    out["test_acc"] = cur.reindex(out["step"]).to_numpy()
    if METRIC == "N1":
        out["main"] = s["w_norm"].to_numpy()
    elif METRIC == "R1":
        cols = sorted((c for c in s.columns if c.startswith("erank_l")), key=lambda c: int(c.rsplit("l", 1)[1]))
        E = s[cols].rolling(3, min_periods=1).median().to_numpy()
        out["main"] = np.nanmean(E, axis=1)
        out["last"] = E[:, -1]
    else:
        cols = [c for c in s.columns if c.startswith("alpha_l")]
        out["main"] = s[cols].rolling(3, min_periods=1).median().mean(axis=1).to_numpy()
    return out


# ==== Figure 3: one typical grokking run per task ======================================
fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex="col")
for j, (app, vars_) in enumerate((("modular", None), ("mnist", ["mnist d3"]), ("teacher_student", None))):
    cand = sel[(sel["app"] == app) & (sel["category"] == "grokker") & (sel["seed"] >= 2) & (sel["max_test"] >= 0.8)]
    if vars_:
        cand = cand[cand["variant"].isin(vars_)]
    cand = cand.sort_values("tf")
    r = cand.index[len(cand) // 2]
    f = signal_frame(r)
    te = sel.loc[r, "t_eps"]
    ax = axes[0, j]
    ax.plot(f["step"], f["test_acc"], color=AQUA, lw=2)
    ax.set_title(f"{APP_NAME[app]} ({VAR_NAME[sel.loc[r, 'variant']]}, {sel.loc[r, 'optim']})", loc="left", fontsize=9.5)
    ax.set_ylabel("test accuracy" if j == 0 else "")
    ax = axes[1, j]
    i0 = int(np.argmin(np.abs(f["step"] - te)))
    ax.plot(f["step"], f["main"] / f["main"].iloc[i0], color=BLUE, lw=2,
            label=f"{SIGNAL}" + (" (mean over layers)" if METRIC == "R1" else ""))
    if METRIC == "R1":
        ax.plot(f["step"], f["last"] / f["last"].iloc[i0], color=BLUE_L, lw=1.6, label="last hidden layer")
    ax.set_ylabel(f"{SIGNAL} ÷ its value\nwhen accuracy finished" if j == 0 else "")
    ax.set_xlabel("training step")
    for a in axes[:, j]:
        a.axvline(te, color=INK, lw=0.8, ls=":")
        a.set_xscale("log")
        style(a, "both")
axes[1, 0].legend(frameon=False, fontsize=8)
fig.text(0.01, 0.005, "Dotted line: when test accuracy stopped improving. Steps on a log scale.", fontsize=8,
         color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "03_example_runs.png", dpi=150)
plt.close(fig)

# ==== Figure 4: medians over runs that learned =========================================
groups = [("Modular", sel["app"] == "modular"), ("MNIST 3-layer", sel["variant"] == "mnist d3"),
          ("MNIST 8-layer", sel["variant"] == "mnist d8"), ("Teacher-student", sel["app"] == "teacher_student")]
xs = np.arange(-2.0, 1.51, 0.05)
fig, axes = plt.subplots(2, 4, figsize=(15, 6.0), sharex=True)
diag = []
for j, (title, m) in enumerate(groups):
    ids = sel.index[m & (sel["t_eps"] > 0) & (sel["max_test"] >= 0.5)]
    curves = {"acc": [], "main": [], "last": []}
    for r in ids:
        f = signal_frame(r)
        f = f[f["step"] > 0]
        x = np.log10(f["step"].to_numpy() / sel.loc[r, "t_eps"])
        i0 = int(np.argmin(np.abs(x)))
        ys = {"acc": f["test_acc"].to_numpy() / sel.loc[r, "max_test"]}
        for k in ("main", "last"):
            if k in f:
                y = f[k].to_numpy(dtype=float)
                ys[k] = y / y[i0] if METRIC != "B5" and np.isfinite(y[i0]) and y[i0] > 0 else y
        for k, y in ys.items():
            row = np.full(len(xs), np.nan)
            inside = (xs >= x.min()) & (xs <= x.max())
            ok = np.isfinite(y)
            if ok.sum() >= 2:
                row[inside] = np.interp(xs[inside], x[ok], y[ok])
            curves[k].append(row)
        # diagnostics: signal at the end and at 3 x t_eps, relative to t_eps; change over the last 10% of budget
        y = f["main"].to_numpy(dtype=float)
        st_ = f["step"].to_numpy()
        T_ = sel.loc[r, "steps"]
        j90 = int(np.searchsorted(st_, 0.9 * T_))
        j3 = int(np.searchsorted(st_, min(3 * sel.loc[r, "t_eps"], T_))) - 1
        diag.append({"group": title, "run_id": r, "end_over_teps": y[-1] / y[i0] if y[i0] else np.nan,
                     "at3x_over_teps": y[max(j3, 0)] / y[i0] if y[i0] else np.nan,
                     "last10pct_change": abs(np.log(y[-1] / y[min(j90, len(y) - 1)]))})
    for row_i, (keys, ylab) in enumerate(((["acc"], "test accuracy ÷ its best"),
                                           (["main", "last"], f"{SIGNAL} ÷ value at that moment"))):
        ax = axes[row_i, j]
        for k in keys:
            if not curves[k]:
                continue
            Y = np.vstack(curves[k])
            good = np.isfinite(Y).sum(0) >= 0.5 * len(Y)
            med, lo, hi = (np.nanpercentile(Y[:, good], q, axis=0) for q in (50, 25, 75))
            col = AQUA if k == "acc" else (BLUE if k == "main" else BLUE_L)
            lab = {"acc": None, "main": SIGNAL + (" (mean of layers)" if METRIC == "R1" else ""),
                   "last": "last hidden layer"}[k]
            if k != "last":
                ax.fill_between(10 ** xs[good], lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(10 ** xs[good], med, color=col, lw=2 if k != "last" else 1.6, label=lab)
        ax.axvline(1, color=INK, lw=0.8, ls=":")
        ax.set_xscale("log")
        style(ax, "both")
        if j == 0:
            ax.set_ylabel(ylab)
        if row_i == 0:
            ax.set_title(f"{title} ({len(ids)} runs that learned)", loc="left", fontsize=9.5)
        else:
            ax.set_xlabel("training time ÷ time when test\naccuracy stopped improving")
axes[1, 0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "04_signal_vs_accuracy.png", dpi=150)
plt.close(fig)
diag = pd.DataFrame(diag)
say("\nsignal diagnostics (medians): value at 3x t_eps and at the end, relative to t_eps; |log change| over the "
    "last 10% of the budget\n", diag.groupby("group")[["at3x_over_teps", "end_over_teps", "last10pct_change"]]
    .median().round(3), diag.groupby("group").size())


# ==== Figures 5, 6a, 6b: held-out scores at each lambda ================================
def forest(lam, path):
    fams = [f for f, _, _ in CANDS]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, (app, split, name) in zip(axes.flat, TESTS):
        s = summ[(summ["lambda"] == lam) & (summ["app"] == app) & (summ["split"] == split)].set_index("family")
        for k, fam in enumerate(fams):
            r = s.loc[fam]
            ax.plot([r["score_lo"], r["score_hi"]], [k, k], color=COLOR[fam], lw=1.6)
            ax.plot(r["score"], k, "o", color=COLOR[fam], ms=6.5, mec=BG, mew=1.2)
            ax.text(max(r["score_hi"], r["score"]) + 0.004, k, f"{r['score']:.3f}", va="center", fontsize=8,
                    color=INK2)
        ax.axvline(s.loc["fixed_frac", "score"], color=ORANGE, lw=0.8, ls="--")
        ax.axvline(s.loc["horizon", "score"], color=GREY, lw=0.8, ls="--")
        ax.set_yticks(range(len(fams)), [LABEL[f] for f in fams], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(f"{name}: {int(s['test_runs'].iloc[0])} runs", loc="left", fontsize=9.5)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
        style(ax)
        xl = ax.get_xlim()
        ax.set_xlim(min(0, xl[0]), xl[1] + 0.03)
    fig.supxlabel(f"score on held-out runs at λ = {lam} (lower is better); every rule tuned at this λ",
                  fontsize=9.5, y=0.03)
    fig.text(0.01, 0.005, "Dot: average. Line: 95% confidence interval. Dashed: never stop (grey), fixed fraction "
             "(orange). Verdicts use paired differences, not these overlaps.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


forest(0.1, FIG / "05_scores.png")
forest(0.03, FIG / "06a_scores_lambda_0.03.png")
forest(0.3, FIG / "06b_scores_lambda_0.3.png")

# ==== Figure 7: the tuned pick against every fixed fraction ============================
fig, axes = plt.subplots(2, 3, figsize=(14, 6.6))
for ax, (app, split, name) in zip(axes.T.flat, TESTS):
    g = fx[(fx["app"] == app) & (fx["split"] == split)]
    r = C(0.1, app, split, BEST)
    tuned_c = float(C(0.1, app, split, "fixed_frac")["setting"].split("=")[1])
    ax.plot(g["c"], g["fixed_score"], color=ORANGE, marker="o", ms=4, lw=1.6, label="fixed fraction, at each fraction")
    tc = g[g["c"] == tuned_c]
    ax.plot(tc["c"], tc["fixed_score"], "o", ms=10, mfc="none", mec=ORANGE, mew=2,
            label="the fraction tuned on the other runs")
    ax.axhspan(r["ci_lo"], r["ci_hi"], color=BLUE, alpha=0.12, lw=0)
    ax.axhline(r["test_cost"], color=BLUE, lw=2, label=f"{METRIC} tuned pick (band: 95% CI)")
    top = min(0.45, max(0.2, g["fixed_score"].max() * 1.05, r["ci_hi"] * 1.05))
    for _, q in g.iterrows():
        if q["verdict"] != "tie" and q["fixed_score"] <= top:
            ax.text(q["c"], q["fixed_score"], "  ✓" if q["verdict"] == "better" else "  ✗", fontsize=9,
                    color=BLUE_D if q["verdict"] == "better" else ORANGE, va="center", clip_on=True)
    ax.set_xscale("log")
    ax.set_xticks(ms.FIXED_FRACS, [f"{c:g}" for c in ms.FIXED_FRACS], fontsize=7.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, top)
    ax.set_title(name.replace(", Test", "\nTest"), loc="left", fontsize=9.5)
    ax.set_xlabel("stop at this fraction of the budget")
    style(ax, "both")
axes[0, 0].set_ylabel("held-out score")
axes[1, 0].set_ylabel("held-out score")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.03))
fig.text(0.01, 0.005, f"✓: the {METRIC} pick is clearly better than stopping at that fraction (paired 95% CI); "
         f"✗: clearly worse; no mark: tie. λ = 0.1. Scores above the top of a panel are cut off (marks there "
         f"are omitted; see tables/vs_every_fixed_fraction.csv).", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(FIG / "07_vs_every_fixed_fraction.png", dpi=150)
plt.close(fig)

# ==== Figure 8: where the rules stop (Test A) ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, app in zip(axes, ("modular", "mnist", "teacher_student")):
    ids = list(test_ids(app, "seeds"))
    ok = sel.loc[ids]
    ok = ok[(ok["max_test"] >= 0.5) & (ok["t_eps"] > 0)]
    c = float(C(0.1, app, "seeds", "fixed_frac")["setting"].split("=")[1])
    t = picked(0.1, app, "seeds", BEST).set_index("run_id").loc[ok.index]
    x = np.clip(ok["tf"].to_numpy(), 1e-3, 1)
    st_ = t["stop"].to_numpy(dtype=float) / ok["steps"].to_numpy()
    nev = ~np.isfinite(st_)
    gr = (ok["category"] == "grokker").to_numpy()
    for mask, mk, lab in ((~gr, "o", "learns fast or other"), (gr, "^", "groks")):
        ax.scatter(x[mask & ~nev], np.clip(st_[mask & ~nev], 1e-3, 1), s=24, marker=mk, color=BLUE_D, alpha=0.8, lw=0,
                   label=f"{METRIC} tuned pick: run that {lab}")
        ax.scatter(x[mask & nev], np.full((mask & nev).sum(), 1.12), s=24, marker=mk, facecolor="none",
                   edgecolor=BLUE_D, lw=1)
    ax.axhline(c, color=ORANGE, lw=2, label="fixed fraction (tuned)")
    lim = (8e-4, 1.5)
    ax.plot(lim, lim, color=INK, lw=0.8, ls=":")
    ax.fill_between(lim, [lim[0]] * 2, lim, color=MAGENTA, alpha=0.06, lw=0)
    ax.text(0.3, 2e-3, "below the dotted line:\nstopped too early", fontsize=8, color=INK2, ha="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("when test accuracy stopped improving (share of budget)")
    ax.set_title(f"{APP_NAME[app]}, Test A: {len(ok)} held-out runs that learned", loc="left", fontsize=9.5)
    style(ax, "both")
axes[0].set_ylabel("when the rule stopped (share of budget)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.04))
fig.text(0.01, 0.005, "Hollow markers at the top: the rule never stopped and trained to the end of the budget.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.11, 1, 1))
fig.savefig(FIG / "08_stop_times.png", dpi=150)
plt.close(fig)

# ==== Figure 9: stopped too early ======================================================
fams9 = ["fixed_frac", FAM_A, FAM_B, BEST]
fig, axes = plt.subplots(2, 3, figsize=(14, 6.0), sharey=True)
axes[0, 0].invert_yaxis()
for ax, (app, split, name) in zip(axes.T.flat, TESTS):
    s = summ[(summ["lambda"] == 0.1) & (summ["app"] == app) & (summ["split"] == split)].set_index("family")
    for k, fam in enumerate(fams9):
        a = s.loc[fam, "too_early_pct"]
        ng = s.loc[fam, "grokkers"]
        b = 100 * s.loc[fam, "grokkers_too_early"] / ng if ng else np.nan
        ax.barh(k - 0.19, a, height=0.36, color=COLOR[fam], alpha=0.45, edgecolor=BG)
        ax.barh(k + 0.19, b, height=0.36, color=COLOR[fam], edgecolor=BG)
        ax.text(a + 1, k - 0.19, f"{a:.0f}%", va="center", fontsize=7.5, color=INK2)
        ax.text((b if np.isfinite(b) else 0) + 1, k + 0.19,
                f"{int(s.loc[fam, 'grokkers_too_early'])} of {int(ng)}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(fams9)), [LABEL[f] for f in fams9], fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_title(name.replace(", Test", "\nTest"), loc="left", fontsize=9.5)
    style(ax)
for ax in axes[1]:
    ax.set_xlabel("% of held-out runs stopped before\ntest accuracy finished improving")
axes[0, 0].legend(handles=[Patch(color=GREY, alpha=0.45, label="all held-out runs"),
                           Patch(color=GREY, label="runs that grok (count)")], frameon=False, fontsize=8,
                  loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "09_too_early.png", dpi=150)
plt.close(fig)

# ==== Figure 10: verdict grid over lambda ==============================================
fams10 = [FAM_A, FAM_B, BEST]
cols = [(app, split, lam) for app, split, _ in TESTS for lam in (0.03, 0.1, 0.3)]
fig, ax = plt.subplots(figsize=(15, 3.3))
vcol = {"better": BLUE, "tie": "#f0efec", "worse": ORANGE}
for i, fam in enumerate(fams10):
    for j, (app, split, lam) in enumerate(cols):
        r = summ[(summ["lambda"] == lam) & (summ["app"] == app) & (summ["split"] == split)
                 & (summ["family"] == fam)].iloc[0]
        v = r["verdict_vs_fixed"]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=vcol[v], edgecolor=BG, lw=2))
        ax.text(j + 0.5, i + 0.5, f"{r['vs_fixed']:+.3f}", ha="center", va="center", fontsize=7.5,
                color="#ffffff" if v != "tie" else INK)
ax.set_xlim(0, len(cols))
ax.set_ylim(len(fams10), 0)
ax.set_yticks(np.arange(len(fams10)) + 0.5, [LABEL[f] for f in fams10], fontsize=8.5)
ax.set_xticks(np.arange(len(cols)) + 0.5, [f"{lam}" for _, _, lam in cols], fontsize=7.5)
for k, (_, _, name) in enumerate(TESTS):
    ax.text(3 * k + 1.5, -0.25, name.replace(" (", "\n("), ha="center", va="bottom", fontsize=8, color=INK)
    if k:
        ax.axvline(3 * k, color=INK2, lw=1)
ax.tick_params(length=0)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
ax.set_xlabel("λ", fontsize=8.5)
ax.legend(handles=[Patch(color=BLUE, label="clearly better than the fixed fraction"),
                   Patch(facecolor="#f0efec", edgecolor=GRID, label="tie (95% interval includes 0)"),
                   Patch(color=ORANGE, label="clearly worse")],
          loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "10_lambda.png", dpi=150)
plt.close(fig)

# ==== Figure 11: does the tuning carry over? ===========================================
fig, ax = plt.subplots(figsize=(9.0, 5.2))
tcol = [BLUE, AQUA, ORANGE, MAGENTA, GREEN, YELLOW]
mk = {"horizon": "s", "fixed_frac": "s"}
for (app, split, name), col in zip(TESTS, tcol):
    s11 = summ[(summ["lambda"] == 0.1) & (summ["app"] == app) & (summ["split"] == split)
               & summ["family"].isin(["horizon", "fixed_frac", FAM_A, FAM_B])]
    for _, r in s11.iterrows():
        ax.scatter(r["cal_score"], r["score"], marker=mk.get(r["family"], "o"), s=42, color=col, alpha=0.9, lw=0)
    far = s11[s11["score"] - s11["cal_score"] > 0.04]
    for _, r in far.iterrows():
        ax.annotate(LABEL[r["family"]].split("  ")[0], (r["cal_score"], r["score"]), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="center")
lim = (0, max(0.25, summ.loc[summ["lambda"] == 0.1, "score"].max() * 1.05))
ax.plot(lim, lim, color=INK, lw=0.8, ls=":")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel("score on the runs it was tuned on")
ax.set_ylabel("score on the held-out runs")
style(ax, "both")
from matplotlib.lines import Line2D  # noqa: E402

h = [Line2D([], [], color=c, marker="o", ls="", label=n) for (_, _, n), c in zip(TESTS, tcol)]
h += [Line2D([], [], color=INK2, marker="o", ls="", label=f"{METRIC} rule A or B"),
      Line2D([], [], color=INK2, marker="s", ls="", label="never stop / fixed fraction"),
      Line2D([], [], color=INK, ls=":", label="same score on both")]
ax.legend(handles=h, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
fig.savefig(FIG / "11_tuned_vs_heldout.png", dpi=150)
plt.close(fig)

# tuned settings, every test and lambda
tun = summ[summ["family"].isin([FAM_A, FAM_B, BEST, "fixed_frac"])][["lambda", "app", "split", "family", "setting"]]
tun.to_csv(TAB / "tuned_settings.csv", index=False)
say("\ntuned settings\n", tun.to_string(index=False))
(TAB / "numbers.txt").write_text("\n".join(out_lines), encoding="utf-8")

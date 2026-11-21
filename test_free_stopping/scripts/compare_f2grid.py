"""Recovered 2026-11-17 from the session that ran it (2026-11-16); see results/phase4/RESULTS.md,
"Re-score on the F2 grid". The paths A (download of HF phase4/metrics_a) and F (download of
HF phase4/agop_final) are hard-coded below: edit them before running."""
"""F2, N1 and R1 on one grid (F2's 223 points): validation, calibration, direct paired
comparisons between the tuned picks."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\advay\Desktop\BTP\test_free_stopping")
from stopping import metric_study as ms
from stopping.agop_final import _cluster_ci
warnings.simplefilter("ignore")

A = Path(r"C:\Users\advay\AppData\Local\Temp\claude\9666543bc33064d0\b072e2a5-970c-4f2c-93cf-c820373f95c4\scratchpad\hf\phase4\metrics_a")
F = Path(r"C:\Users\advay\AppData\Local\Temp\claude\9666543bc33064d0\9157bee3-9ceb-4e56-b96a-d32f694a130e\scratchpad\final\phase4\agop_final")
mine = pd.read_parquet(A.parent / "rescore_f2grid" / "rows_f2grid.parquet")
f2 = pd.read_parquet(F / "rows.parquet")
meta = pd.read_csv(A / "selection.csv")
meta = meta[meta.run_id.isin(mine.run_id)]

# 1. validation: references identical to the F2 study's?
cols = ["cost_0.03", "cost_0.1", "cost_0.3", "stop"]
for fam in ("fixed_frac", "horizon", "oracle_teps"):
    a = mine[mine.family == fam].set_index(["run_id", "setting"])[cols]
    b = f2[f2.family == fam].copy()
    if fam != "fixed_frac":
        b["setting"] = fam
    b = b.set_index(["run_id", "setting"])[cols]
    j = a.join(b, rsuffix="_f2", how="inner")
    diff = max(float(np.nanmax(np.abs(j[c] - j[c + "_f2"]))) for c in ["cost_0.03", "cost_0.1", "cost_0.3"])
    print(f"validation {fam}: {len(j)} rows matched, max |cost diff| = {diff:.2e}")

# 2. one row table: F2 families from the F2 study, N1/R1 from the re-score, shared refs
F2FAM = ("bound", "rel_delta", "const_delta", "msd", "geweke")
ms.METRICS["F2"] = F2FAM
f2r = f2[f2.family.isin(F2FAM)].assign(metric="F2")
rows = pd.concat([mine, f2r[mine.columns]], ignore_index=True)

LAB = {("modular", "seeds"): "Modular, new seeds", ("modular", "operation"): "Modular, new ops",
       ("mnist", "seeds"): "MNIST, new seeds", ("mnist", "depth"): "MNIST, 8-layer"}
picks, res = {}, []
for lam in (0.03, 0.1, 0.3):
    for (app, split), lab in LAB.items():
        cal, test = ms.split_ids(meta, app, split)
        for M in ("F2", "N1", "R1"):
            c = ms.calibrate(rows, meta, cal, test, lam, M).set_index("family")
            for fam in ("horizon", "fixed_frac", f"{M} best"):
                r = c.loc[fam]
                res.append({"lam": lam, "test": lab, "rule": fam if fam != f"{M} best" else M,
                            "setting": r.setting, "score": r.test_cost, "lo": r.ci_lo, "hi": r.ci_hi,
                            "vs_fixed": r.d_fixed, "vf_lo": r.d_fixed_lo, "vf_hi": r.d_fixed_hi,
                            "vs_never": r.d_horizon, "vn_lo": r.d_horizon_lo, "vn_hi": r.d_horizon_hi,
                            "premature": r.premature, "never": r.never})
            picks[(lam, app, split, M)] = c.loc[f"{M} best", "setting"]
res = pd.DataFrame(res).drop_duplicates(["lam", "test", "rule"])
v = lambda lo, hi: "better" if hi < 0 else ("worse" if lo > 0 else "tie")
res["verdict_fixed"] = [v(a, b) if r not in ("fixed_frac",) else "" for a, b, r in zip(res.vf_lo, res.vf_hi, res.rule)]
res["verdict_never"] = [v(a, b) if r not in ("horizon",) else "" for a, b, r in zip(res.vn_lo, res.vn_hi, res.rule)]
pd.set_option("display.width", 250)
for lam in (0.1, 0.03, 0.3):
    print(f"\n=== lambda {lam}")
    print(res[res.lam == lam].drop(columns="lam").round(3).to_string(index=False))

# 3. direct paired comparisons between the tuned picks, and grokkers cut off
cost = rows.merge(meta[["run_id", "cell", "category"]], on="run_id")
print("\n=== paired differences between tuned picks (held-out runs, cluster bootstrap)")
for lam in (0.03, 0.1, 0.3):
    col = f"cost_{lam}"
    for (app, split), lab in LAB.items():
        _, test = ms.split_ids(meta, app, split)
        t = cost[cost.run_id.isin(test)]
        piv = t.pivot_table(index="run_id", columns="setting", values=col, aggfunc="first")
        cells = t.drop_duplicates("run_id").set_index("run_id")["cell"].reindex(piv.index).to_numpy()
        s = {M: piv[picks[(lam, app, split, M)]].to_numpy() for M in ("F2", "N1", "R1")}
        out = []
        for a, b in (("N1", "F2"), ("R1", "F2"), ("R1", "N1")):
            d = s[a] - s[b]
            lo, hi = _cluster_ci(d, cells)
            out.append(f"{a}-{b} {np.nanmean(d):+.3f} [{lo:+.3f},{hi:+.3f}] {v(lo, hi)}")
        print(f"lam {lam:<4} {lab:<18} " + " | ".join(out))

print("\n=== grokkers stopped before t_eps (held-out), lambda 0.1")
for (app, split), lab in LAB.items():
    _, test = ms.split_ids(meta, app, split)
    t = cost[cost.run_id.isin(test) & (cost.category == "grokker")]
    n = t.run_id.nunique()
    parts = []
    for M in ("F2", "N1", "R1"):
        parts.append(f"{M} {int(t[t.setting == picks[(0.1, app, split, M)]].premature.sum())}/{n}")
    ffs = res[(res.lam == 0.1) & (res.test == lab) & (res.rule == "fixed_frac")].setting.iloc[0]
    parts.append(f"fixed({ffs}) {int(t[t.setting == ffs].premature.sum())}/{n}")
    print(f"{lab:<18} " + "  ".join(parts))

print("\n=== tuned settings at lambda 0.1")
for k, s in picks.items():
    if k[0] == 0.1:
        print(k[1:], s)
res.to_csv(A.parent / "rescore_f2grid" / "summary_f2grid.csv", index=False)

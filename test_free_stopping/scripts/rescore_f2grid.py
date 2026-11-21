"""Recovered 2026-11-17 from the session that ran it (2026-11-16); see results/phase4/RESULTS.md,
"Re-score on the F2 grid". The paths A (download of HF phase4/metrics_a) and F (download of
HF phase4/agop_final) are hard-coded below: edit them before running."""
"""Re-score study A's N1 and R1 rules at the F2 study's 223 check points per run, so
F2, N1 and R1 can be compared on one grid. Reads local copies of the HF outputs only."""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, r"C:\Users\advay\Desktop\BTP\test_free_stopping")
from stopping import metric_study as ms, agop_study as st

A = Path(r"C:\Users\advay\AppData\Local\Temp\claude\9666543bc33064d0\b072e2a5-970c-4f2c-93cf-c820373f95c4\scratchpad\hf\phase4\metrics_a")
F = Path(r"C:\Users\advay\AppData\Local\Temp\claude\9666543bc33064d0\9157bee3-9ceb-4e56-b96a-d32f694a130e\scratchpad\final\phase4\agop_final")
OUT = A.parent / "rescore_f2grid"
OUT.mkdir(exist_ok=True)

sel = pd.read_csv(A / "selection.csv")
fser = pd.read_parquet(F / "series.parquet", columns=["run_id", "step"])
grids = {r: g["step"].to_numpy(np.int64) for r, g in fser.groupby("run_id")}
print("F2 runs", len(grids), "study A runs in F2 grid", sel.run_id.isin(grids).sum())


def analyze_on_grid(trace, curves, info, G):
    T = float(curves["step"][-1])
    Gf = G.astype(float)
    rows = []

    def add(metric, family, setting, cond, patience=ms.PATIENCE):
        stop = st.first_stop(G, np.asarray(cond, dtype=bool), patience=patience)
        rows.append({"run_id": info["run_id"], "metric": metric, "family": family, "setting": setting,
                     **st.score_stop(curves, stop, info)})

    # N1, exactly as ms.analyze_run but evaluated at G
    lg = trace[["step", "w_norm"]].dropna()
    s1 = lg["step"].to_numpy(dtype=float)
    y1 = np.log(np.maximum(lg["w_norm"].to_numpy(dtype=float), 1e-300))
    momentum = 0.9 if info.get("optim") == "sgd_momentum" else 0.0
    lr_eff = float(info.get("lr") or 0) / (1 - momentum)
    gamma = float(info.get("gamma") or 0)
    for kind, x in ms._lags():
        d, gap = ms.lag_change(s1, y1, Gf, kind, x, T)
        for norm in ("none", "decay"):
            use_decay = norm == "decay" and gamma > 0 and lr_eff > 0
            v = d / (lr_eff * gamma * gap) if use_decay else d
            for k in range(len(ms.EPS_LOG)):
                eps_name = ms.EPS_DECAY[k] if norm == "decay" else ms.EPS_LOG[k]
                e = ms.EPS_DECAY[k] if use_decay else ms.EPS_LOG[k]
                s = f"{kind} {ms._tag(kind, x)} norm={norm} eps={eps_name}"
                with np.errstate(invalid="ignore"):
                    add("N1", "n1_abs", "N1 abs " + s, np.abs(v) <= e)
                    add("N1", "n1_shrink", "N1 shrink " + s, v >= -e)

    # R1: smoothing and guard on the summary grid (as in study A), read causally at G
    summ = trace[trace["summary"]].reset_index(drop=True)
    S = summ["step"].to_numpy(dtype=float)
    erc = ms._layer_cols(summ, "erank_")
    E = ms._med3(summ[erc].to_numpy(dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logE = np.log(np.maximum(E, 1e-300))
        mean = np.nanmean(E, axis=1) if E.shape[1] > 1 else E[:, 0]
    runmax = np.fmax.accumulate(np.where(np.isfinite(mean), mean, -np.inf))
    guard_s = np.isfinite(mean) & (mean <= (1 - ms.GUARD_DROP) * runmax)
    gi = np.searchsorted(S, Gf, side="right") - 1
    guard = np.where(gi >= 0, guard_s[np.maximum(gi, 0)], False)
    for kind, x in ms._lags():
        dm = np.abs(ms.lag_change(S, np.log(np.maximum(mean, 1e-300)), Gf, kind, x, T)[0])
        dl = np.abs(ms.lag_change(S, logE[:, -1], Gf, kind, x, T)[0])
        per = np.vstack([np.abs(ms.lag_change(S, logE[:, j], Gf, kind, x, T)[0]) for j in range(E.shape[1])])
        dall = np.where(np.isnan(per).all(0), np.nan, np.nanmax(np.where(np.isnan(per), -np.inf, per), axis=0))
        for layers, dd in (("mean", dm), ("last", dl), ("all", dall)):
            for eps in ms.EPS_LOG:
                s = f"R1 {layers} {kind} {ms._tag(kind, x)} eps={eps}"
                with np.errstate(invalid="ignore"):
                    base = dd <= eps
                add("R1", "r1", s, base)
                add("R1", "r1_guard", s + " guard", base & guard)

    # references: fixed fraction with patience 3 as in the F2 study
    for c in ms.FIXED_FRACS:
        add("ref", "fixed_frac", f"fixed_frac c={c}", Gf >= c * T)
    for b in st.baseline_rows(curves, info):
        fam = "horizon" if b["setting"].startswith("horizon") else "oracle_teps"
        rows.append({"run_id": info["run_id"], "metric": "ref", "family": fam, "setting": fam,
                     **{k: b[k] for k in b if k not in ("run_id", "app", "category", "optim", "family", "setting")}})
    return pd.DataFrame(rows)


out = []
for info in sel[sel.run_id.isin(grids)].to_dict("records"):
    rid = info["run_id"]
    trace = pd.read_parquet(A / "traces" / f"{rid}.parquet")
    curves = json.loads((A / "curves" / f"{rid}.json").read_text())
    out.append(analyze_on_grid(trace, curves, info, grids[rid]))
rows = pd.concat(out, ignore_index=True)
rows.to_parquet(OUT / "rows_f2grid.parquet", index=False)
print("rows", len(rows), "runs", rows.run_id.nunique())

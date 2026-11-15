"""Phase 4 study of three cheap signals as stopping rules: N1 (weight norm), R1 (effective
rank of the activations) and B5 (HTSR alpha of the weight spectra). Same protocol as the
final F2 study (stopping/agop_final.py): tune each rule on some runs, score it on held-out
runs, against never stopping, a tuned fixed fraction of the budget and the perfect stop.

All three come from data recorded during training (telemetry every log step, summaries
every summary step), so no replay is needed: each run is downloaded (telemetry,
summaries, test curve), its signals are computed once with the causal signal classes
(stopping.signals), and every rule setting is evaluated offline on the summary-step grid.
B5's correlation traps need the full weight matrices (replay) and are left to the replay
study of B2 and S1b.

Evaluation side (reads test curves to score stops), so exempt from the quarantine test,
like stopping.agop_final. Outputs are organized per metric (one report each); see
build_reports and molab/phase4_metrics_a.py.

Rule grid (metric -> families -> settings; every rule needs PATIENCE consecutive
summary rows; the "change over a lag" is |log y(t) - log y(t')|, t' = (1 - f) t for a lag
growing with t, or t - c T for a fixed share c of the budget T):
- N1 (140): n1_abs  |change| <= eps;  n1_shrink  change >= -eps (stopped shrinking).
  lag: f in GROWING or c in FIXED (7) x normalization {none, decay: change / (lr_eff gamma
  dt), the change in units of pure weight decay; at gamma = 0 it falls back to none}
  (2) x eps (5).
- R1 (210): r1 and r1_guard (only after the mean erank has fallen GUARD_DROP below its
  running max: the end of the early rise must not read as settling).
  layers {mean, last hidden, all must pass} (3) x lag (7) x eps (5).
- B5 (28): b5_level  alpha <= a (5 levels);  b5_settle  |change of log alpha| <= eps
  over a lag growing with t (3 x 3); each for layers {mean, all must pass} (2).
  Layers with fewer than B5_MIN_EIGS singular values are not fitted.
References in every metric's report: horizon (never stop), fixed_frac (stop at c T),
oracle_teps (stop exactly at t_eps).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import agop_study as st
from .agop_final import _cluster_ci, _sign_test, pick_final_runs
from .compute import compute_run
from .signals import ERankConfig, HtsrAlphaConfig, WeightNormConfig

APPS = ("modular", "mnist", "teacher_student")
METRICS = {"N1": ("n1_abs", "n1_shrink"), "R1": ("r1", "r1_guard"), "B5": ("b5_level", "b5_settle")}
METRIC_NAMES = {"N1": "weight norm", "R1": "effective rank of the activations", "B5": "HTSR alpha"}
REFS = ("horizon", "fixed_frac", "oracle_teps")
SEEDS = (0, 1, 2, 3)
SPLITS = {
    "modular": {"seeds": ("seed", (0, 1), (2, 3)),
                "operation": ("variant", ("modular x+y",), ("modular x*y", "modular x/y"))},
    "mnist": {"seeds": ("seed", (0, 1), (2, 3)), "depth": ("variant", ("mnist d3",), ("mnist d8",))},
    "teacher_student": {"seeds": ("seed", (0, 1), (2, 3)),
                        "size": ("variant", ("teacher_student n50", "teacher_student n100"),
                                 ("teacher_student n400",))},
}
SPLIT_NAMES = {("modular", "seeds"): "new seeds", ("modular", "operation"): "new operations (x+y -> x*y, x/y)",
               ("mnist", "seeds"): "new seeds", ("mnist", "depth"): "deeper network (d3 -> d8)",
               ("teacher_student", "seeds"): "new seeds",
               ("teacher_student", "size"): "more data (N 50, 100 -> 400)"}
DEFAULT_CELLS = {"modular": {"modular x+y": 6, "modular x*y": 2, "modular x/y": 2},
                 "mnist": {"mnist d3": 2, "mnist d8": 2},
                 "teacher_student": {"teacher_student n50": 2, "teacher_student n100": 2, "teacher_student n400": 2}}
GROWING = (0.1, 0.25, 0.5)  # lag = f t
FIXED = (0.01, 0.02, 0.05, 0.1)  # lag = c T
EPS_LOG = (0.001, 0.003, 0.01, 0.03, 0.1)  # thresholds on a change of log y over the lag
EPS_DECAY = (0.01, 0.03, 0.1, 0.3, 1.0)  # the same, in units of the pure weight-decay shrinkage
R1_LAYERS = ("mean", "last", "all")
GUARD_DROP = 0.05
B5_LEVELS = (2.0, 2.5, 3.0, 3.5, 4.0)
B5_EPS = (0.01, 0.03, 0.1)
B5_MIN_EIGS = 50
FIXED_FRACS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
PATIENCE = st.PATIENCE
MAIN_LAMBDA = st.MAIN_LAMBDA
META_COLS = ("run_id", "app", "variant", "optim", "lr", "gamma", "alpha", "size", "seed", "category", "cell",
             "cell_category", "max_test", "t_eps", "train99_step", "wall_s", "steps")


# ---- run selection ----------------------------------------------------------------------

def prepare_rows(rows: list[dict], app: str) -> list[dict]:
    """Corpus summary rows of one application, with teacher-student's variant split by
    training-set size (its hyperparameter cells are chosen per (size, optimizer))."""
    out = []
    for r in rows:
        r = dict(r)
        if app == "teacher_student":
            r["variant"] = f"teacher_student n{int(r['size'])}"
        out.append(r)
    return out


def select_runs(rows_by_app: dict, cells: Optional[dict] = None) -> list[dict]:
    """Whole cells (all 4 seeds), round-robin over seed-0 categories per (variant,
    optimizer), as in the F2 study (so modular and MNIST pick the same runs)."""
    cells = cells or DEFAULT_CELLS
    out = []
    for app in APPS:
        if app in rows_by_app and cells.get(app):
            out += pick_final_runs(prepare_rows(rows_by_app[app], app), app, cells[app])
    return out


def build_meta(selection: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: r.get(k) for k in META_COLS} for r in selection])


# ---- one run: signals from the recorded data ---------------------------------------------

def signal_configs(min_eigs: int = B5_MIN_EIGS) -> tuple:
    return WeightNormConfig(), ERankConfig(), HtsrAlphaConfig(min_eigs=min_eigs)


_LAYER = re.compile(r"^l(\d+)$")
_KS = re.compile(r"^ks_l(\d+)$")


def trace_run(handle, min_eigs: int = B5_MIN_EIGS) -> tuple[pd.DataFrame, dict]:
    """(trace, curves). trace: one row per log step with w_norm, and at summary steps the
    per-layer erank (erank_l{i}), HTSR alpha and its KS distance (alpha_l{i}, ks_l{i}).
    curves: step, train_acc, test_acc (scoring only)."""
    cfgs = signal_configs(min_eigs)
    tr = compute_run(handle, cfgs)
    n1, r1, b5 = (tr[c.key] for c in cfgs)
    df = n1[["step", "value"]].rename(columns={"value": "w_norm"})
    er = r1[["step"] + [c for c in r1.columns if _LAYER.match(c)]]
    er = er.rename(columns={c: f"erank_{c}" for c in er.columns if c != "step"})
    al = b5[["step"] + [c for c in b5.columns if _LAYER.match(c) or _KS.match(c)]]
    al = al.rename(columns={c: f"alpha_{c}" if _LAYER.match(c) else c for c in al.columns if c != "step"})
    df = df.merge(er, on="step", how="left").merge(al, on="step", how="left")
    df["summary"] = df["step"].isin(r1["step"])
    tel = handle.telemetry()
    test = handle.heldout().set_index("step")["test_acc"].reindex(tel["step"]).to_numpy(dtype=float)
    curves = {"step": tel["step"].tolist(), "train_acc": tel["train_acc"].tolist(), "test_acc": test.tolist()}
    return df, curves


# ---- one run: every rule setting ---------------------------------------------------------

def _layer_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted((c for c in df.columns if re.match(rf"^{prefix}l\d+$", c)), key=lambda c: int(c.rsplit("l", 1)[1]))


def _med3(a: np.ndarray) -> np.ndarray:
    """Causal median of the last 3 rows, per column (alpha fits and erank jitter)."""
    return pd.DataFrame(a).rolling(3, min_periods=1).median().to_numpy()


def lag_change(src: np.ndarray, logy: np.ndarray, ev: np.ndarray, kind: str, x: float, T: float):
    """Signed logy(t) - logy(t') at each eval step t, with t' the last src step <= the lag
    target ((1 - x) t for "growing", t - x T for "fixed"), and the step gap t - t'. NaN
    where no earlier point exists."""
    target = ev * (1 - x) if kind == "growing" else ev - x * T
    i = np.searchsorted(src, ev, side="right") - 1
    j = np.searchsorted(src, target, side="right") - 1
    ok = (i >= 0) & (j >= 0) & (j < i) & (target >= 0)
    d, gap = np.full(len(ev), np.nan), np.full(len(ev), np.nan)
    d[ok] = logy[i[ok]] - logy[j[ok]]
    gap[ok] = src[i[ok]] - src[j[ok]]
    return d, gap


def _lags():
    return [("growing", f) for f in GROWING] + [("fixed", c) for c in FIXED]


def _tag(kind: str, x: float) -> str:
    return f"f={x}" if kind == "growing" else f"c={x}"


def analyze_run(trace: pd.DataFrame, curves: dict, info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows: one per metric x family x setting, scored; series: the signals on the
    summary grid, for plots)."""
    T = float(curves["step"][-1])
    summ = trace[trace["summary"]].reset_index(drop=True)
    G = summ["step"].to_numpy(dtype=np.int64)
    Gf = G.astype(float)
    rows: list[dict] = []

    def add(metric, family, setting, cond, patience=PATIENCE):
        stop = st.first_stop(G, np.asarray(cond, dtype=bool), patience=patience)
        rows.append({"run_id": info["run_id"], "metric": metric, "family": family, "setting": setting,
                     **st.score_stop(curves, stop, info)})

    # N1: weight norm at every log step
    lg = trace[["step", "w_norm"]].dropna()
    s1 = lg["step"].to_numpy(dtype=float)
    y1 = np.log(np.maximum(lg["w_norm"].to_numpy(dtype=float), 1e-300))
    momentum = 0.9 if info.get("optim") == "sgd_momentum" else 0.0
    lr_eff = float(info.get("lr") or 0) / (1 - momentum)
    gamma = float(info.get("gamma") or 0)
    for kind, x in _lags():
        d, gap = lag_change(s1, y1, Gf, kind, x, T)
        for norm in ("none", "decay"):
            use_decay = norm == "decay" and gamma > 0 and lr_eff > 0
            v = d / (lr_eff * gamma * gap) if use_decay else d
            for k in range(len(EPS_LOG)):
                eps_name = EPS_DECAY[k] if norm == "decay" else EPS_LOG[k]
                e = EPS_DECAY[k] if use_decay else EPS_LOG[k]
                s = f"{kind} {_tag(kind, x)} norm={norm} eps={eps_name}"
                with np.errstate(invalid="ignore"):
                    add("N1", "n1_abs", "N1 abs " + s, np.abs(v) <= e)
                    add("N1", "n1_shrink", "N1 shrink " + s, v >= -e)

    # R1: effective rank per hidden layer, at summary steps
    erc = _layer_cols(summ, "erank_")
    if erc:
        E = _med3(summ[erc].to_numpy(dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            logE = np.log(np.maximum(E, 1e-300))
            mean = np.nanmean(E, axis=1) if E.shape[1] > 1 else E[:, 0]
        runmax = np.fmax.accumulate(np.where(np.isfinite(mean), mean, -np.inf))
        guard = np.isfinite(mean) & (mean <= (1 - GUARD_DROP) * runmax)
        for kind, x in _lags():
            dm = np.abs(lag_change(Gf, np.log(np.maximum(mean, 1e-300)), Gf, kind, x, T)[0])
            dl = np.abs(lag_change(Gf, logE[:, -1], Gf, kind, x, T)[0])
            per = np.vstack([np.abs(lag_change(Gf, logE[:, j], Gf, kind, x, T)[0]) for j in range(E.shape[1])])
            dall = np.where(np.isnan(per).all(0), np.nan, np.nanmax(np.where(np.isnan(per), -np.inf, per), axis=0))
            for layers, dd in (("mean", dm), ("last", dl), ("all", dall)):
                for eps in EPS_LOG:
                    s = f"R1 {layers} {kind} {_tag(kind, x)} eps={eps}"
                    with np.errstate(invalid="ignore"):
                        base = dd <= eps
                    add("R1", "r1", s, base)
                    add("R1", "r1_guard", s + " guard", base & guard)
    else:
        mean = np.full(len(G), np.nan)

    # B5: HTSR alpha per eligible layer, at summary steps
    alc = _layer_cols(summ, "alpha_")
    A = _med3(summ[alc].to_numpy(dtype=float)) if alc else np.full((len(G), 1), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (no eligible layer yet)
        a_mean = np.nanmean(A, axis=1)
        a_all = np.nanmax(A, axis=1)
    for layers, agg in (("mean", a_mean), ("all", a_all)):
        for a in B5_LEVELS:
            with np.errstate(invalid="ignore"):
                add("B5", "b5_level", f"B5 {layers} alpha<={a}", agg <= a)
    logA = np.log(np.maximum(A, 1e-300))
    for f in GROWING:
        per = np.vstack([np.abs(lag_change(Gf, logA[:, j], Gf, "growing", f, T)[0]) for j in range(A.shape[1])])
        dmean = np.abs(lag_change(Gf, np.log(np.maximum(a_mean, 1e-300)), Gf, "growing", f, T)[0])
        dall = np.where(np.isnan(per).all(0), np.nan, np.nanmax(np.where(np.isnan(per), -np.inf, per), axis=0))
        for layers, dd in (("mean", dmean), ("all", dall)):
            for eps in B5_EPS:
                with np.errstate(invalid="ignore"):
                    add("B5", "b5_settle", f"B5 {layers} settle f={f} eps={eps}", dd <= eps)

    # references
    for c in FIXED_FRACS:
        add("ref", "fixed_frac", f"fixed_frac c={c}", Gf >= c * T, patience=1)
    for b in st.baseline_rows(curves, info):
        fam = "horizon" if b["setting"].startswith("horizon") else "oracle_teps"
        rows.append({"run_id": info["run_id"], "metric": "ref", "family": fam, "setting": fam,
                     **{k: b[k] for k in b if k not in ("run_id", "app", "category", "optim", "family", "setting")}})

    test = np.array([st._acc_at(curves, int(s)) for s in G])
    w_at = np.interp(Gf, s1, np.exp(y1)) if len(s1) else np.full(len(G), np.nan)
    series = pd.DataFrame({"run_id": info["run_id"], "step": G, "t_frac": Gf / T, "test_acc": test,
                           "w_norm": w_at, "erank_mean": mean, "alpha_mean": a_mean})
    return pd.DataFrame(rows), series


# ---- many runs, in parallel --------------------------------------------------------------

def _paths(work: Path, rid: str) -> dict:
    work = Path(work)
    return {"trace": work / "traces" / f"{rid}.parquet", "curves": work / "curves" / f"{rid}.json",
            "rows": work / "analysis" / "rows" / f"{rid}.parquet",
            "series": work / "analysis" / "series" / f"{rid}.parquet"}


def traced(work: Path, rid: str) -> bool:
    p = _paths(work, rid)
    return p["trace"].exists() and p["curves"].exists()


def analyzed(work: Path, rid: str) -> bool:
    p = _paths(work, rid)
    return p["rows"].exists() and p["series"].exists()


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


_W: dict = {}


def _init_worker(repo: str, token: Optional[str], work: str) -> None:
    _W.clear()
    _W.update(repo=repo, token=token, work=work)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _hf_source(app: str):
    if app not in _W:
        from .tfs_source import HfRunSource
        _W[app] = HfRunSource(_W["repo"], f"phase3/{app}", work=Path(_W["work"]) / f"runs_{os.getpid()}",
                              device="cpu", token=_W.get("token"))
    return _W[app]


def process_run(info: dict, work: str, min_eigs: int = B5_MIN_EIGS, source=None) -> dict:
    """Trace (download + signals) unless done, then analyze unless done. Never raises."""
    rid, work = info["run_id"], Path(work)
    p = _paths(work, rid)
    t0, err, did = time.time(), "", []
    try:
        if not traced(work, rid):
            src = source if source is not None else _hf_source(info["app"])
            with src.open(rid, {"telemetry", "summaries", "heldout"}) as h:
                trace, curves = trace_run(h, min_eigs)
            _atomic(p["trace"], lambda t: trace.to_parquet(t, index=False))
            _atomic(p["curves"], lambda t: t.write_text(json.dumps(curves)))
            did.append("trace")
        if not analyzed(work, rid):
            rows, series = analyze_run(pd.read_parquet(p["trace"]), json.loads(p["curves"].read_text()), info)
            _atomic(p["rows"], lambda t: rows.to_parquet(t, index=False))
            _atomic(p["series"], lambda t: series.to_parquet(t, index=False))
            did.append("analysis")
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    return {"run_id": rid, "app": info["app"], "ok": not err, "did": "+".join(did) or "cached",
            "seconds": round(time.time() - t0, 1), "error": err}


def run_all(selection: list[dict], work: Path, workers: int = 8, repo: str = "advay333/tfs",
            token: Optional[str] = None, out_remote=None, upload_every: int = 20, min_eigs: int = B5_MIN_EIGS,
            source=None, log: Callable[[str], None] = print) -> list[dict]:
    """process_run for every selected run. With `source` (tests) everything runs in this
    process; otherwise `workers` processes download from HF (token passed in memory,
    never written). New traces and curves are uploaded to out_remote every
    `upload_every` runs, so an interrupted session resumes (resume_from)."""
    work = Path(work)
    selection = sorted(selection, key=lambda r: -float(r.get("wall_s") or 0))  # long runs first
    status, pending = [], {}

    def flush(msg):
        if out_remote is not None and pending:
            out_remote.commit(dict(pending), message=msg)
            log(f"uploaded {len(pending)} file(s)")
            pending.clear()

    def done(s, n):
        status.append(s)
        if s["ok"] and "trace" in s["did"]:
            pending[f"traces/{s['run_id']}.parquet"] = _paths(work, s["run_id"])["trace"]
            pending[f"curves/{s['run_id']}.json"] = _paths(work, s["run_id"])["curves"]
        el = time.time() - t0
        log(f"[{n}/{len(selection)}] {s['run_id']}: {'ok' if s['ok'] else 'FAILED ' + s['error']} "
            f"({s['did']}, {s['seconds']} s; {el / 60:.1f} min elapsed)")
        if len(pending) >= 2 * upload_every:
            flush("metrics_a: traces")

    t0 = time.time()
    if source is not None or workers <= 1:
        if source is None:
            _init_worker(repo, token, str(work))
        for n, r in enumerate(selection, 1):
            done(process_run(r, str(work), min_eigs, source), n)
    else:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(repo, token, str(work))) as ex:
            futs = [ex.submit(process_run, r, str(work), min_eigs) for r in selection]
            for n, f in enumerate(as_completed(futs), 1):
                done(f.result(), n)
    flush("metrics_a: traces")
    return status


def resume_from(remote, work: Path, run_ids: set, log: Callable[[str], None] = print) -> int:
    """Download traces and curves this study already uploaded, for the given runs."""
    got = 0
    for folder, ext in (("traces", ".parquet"), ("curves", ".json")):
        try:
            files = remote.list_files(folder)
        except Exception as e:
            log(f"nothing to resume in {folder}/ ({type(e).__name__})")
            continue
        for path in files:
            rid = Path(path).name[: -len(ext)]
            if rid not in run_ids:
                continue
            dest = Path(work) / folder / Path(path).name
            if not dest.exists():
                remote.download(path, dest)
                got += 1
    log(f"resumed {got} file(s) from the remote")
    return got


def load_analysis(work: Path, run_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    R, S = [], []
    for rid in run_ids:
        if analyzed(work, rid):
            p = _paths(work, rid)
            R.append(pd.read_parquet(p["rows"]))
            S.append(pd.read_parquet(p["series"]))
    return (pd.concat(R, ignore_index=True) if R else pd.DataFrame(),
            pd.concat(S, ignore_index=True) if S else pd.DataFrame())


# ---- calibration: tune on some runs, score on held-out runs ------------------------------

def split_ids(meta: pd.DataFrame, app: str, split: str) -> tuple[set, set]:
    col, a, b = SPLITS[app][split]
    m = meta[meta["app"] == app]
    return set(m.loc[m[col].isin(a), "run_id"]), set(m.loc[m[col].isin(b), "run_id"])


def family_order(metric: str) -> tuple:
    return ("horizon", "fixed_frac") + METRICS[metric] + (f"{metric} best", "oracle_teps")


def calibrate(rows: pd.DataFrame, meta: pd.DataFrame, cal: set, test: set, lam: float, metric: str) -> pd.DataFrame:
    """Per family: the setting with the lowest mean cost on the calibration runs, scored
    on the test runs, with 95% cluster-bootstrap CIs (over hyperparameter cells) and
    paired differences against never stopping and the tuned fixed fraction. '<metric>
    best' is the tuned pick across the metric's families. Also the hindsight-best
    setting on the test runs."""
    col = f"cost_{lam}"
    fams = family_order(metric)
    own = METRICS[metric]
    df = rows[rows["family"].isin(set(fams) | set(own))].merge(meta[["run_id", "cell"]], on="run_id", how="left")
    c_df, t_df = df[df["run_id"].isin(cal)], df[df["run_id"].isin(test)]
    if c_df.empty or t_df.empty:
        return pd.DataFrame()
    cal_mean = c_df.groupby(["family", "setting"])[col].mean()
    test_mean = t_df.groupby(["family", "setting"])[col].mean()
    tpiv = t_df.pivot_table(index="run_id", columns="setting", values=col, aggfunc="first")
    cells = t_df.drop_duplicates("run_id").set_index("run_id")["cell"].reindex(tpiv.index).to_numpy()
    picks = {}
    for fam in fams:
        sub = cal_mean[cal_mean.index.get_level_values(0).isin(own if fam == f"{metric} best" else [fam])]
        if len(sub):
            picks[fam] = sub.idxmin()
    hz = tpiv[picks["horizon"][1]].to_numpy()
    ff = tpiv[picks["fixed_frac"][1]].to_numpy()
    out = []
    for fam, (src_fam, setting) in picks.items():
        v = tpiv[setting].to_numpy()
        tr = t_df[t_df["setting"] == setting]
        lo, hi = _cluster_ci(v, cells)
        dh, dfx = v - hz, v - ff
        w, l, p = _sign_test(dh, cells)
        wf, lf, pf = _sign_test(dfx, cells)
        fam_test = test_mean[test_mean.index.get_level_values(0).isin(own if fam == f"{metric} best" else [fam])]
        dhl, dhh = _cluster_ci(dh, cells)
        dfl, dfh = _cluster_ci(dfx, cells)
        out.append({
            "metric": metric, "family": fam, "picked_family": src_fam, "setting": setting, "lambda": lam,
            "cal_runs": c_df["run_id"].nunique(), "test_runs": int(np.isfinite(v).sum()),
            "test_cells": len(set(cells)), "cal_cost": float(cal_mean.loc[(src_fam, setting)]),
            "test_cost": float(np.nanmean(v)), "ci_lo": lo, "ci_hi": hi,
            "premature": float(tr["premature"].mean()), "never": float(tr["never"].mean()),
            "regret": float(tr["regret"].mean()), "late": float(tr["late"].mean()),
            "d_horizon": float(np.nanmean(dh)), "d_horizon_lo": dhl, "d_horizon_hi": dhh,
            "cells_better_h": w, "cells_worse_h": l, "p_h": p,
            "d_fixed": float(np.nanmean(dfx)), "d_fixed_lo": dfl, "d_fixed_hi": dfh,
            "cells_better_f": wf, "cells_worse_f": lf, "p_f": pf,
            "hindsight_setting": fam_test.idxmin()[1] if len(fam_test) else None,
            "hindsight_cost": float(fam_test.min()) if len(fam_test) else np.nan,
        })
    return pd.DataFrame(out)


def calibration_table(rows: pd.DataFrame, meta: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = []
    for app in SPLITS:
        if not (meta["app"] == app).any():
            continue
        for split in SPLITS[app]:
            cal, test = split_ids(meta, app, split)
            if not cal or not test:
                continue
            for lam in st.LAMBDAS:
                t = calibrate(rows[rows["run_id"].isin(cal | test)], meta, cal, test, lam, metric)
                if len(t):
                    out.append(t.assign(app=app, split=split))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def verdict(lo: float, hi: float) -> str:
    return "better" if hi < 0 else ("worse" if lo > 0 else "tie")


# ---- reports, one per metric -------------------------------------------------------------

_COL = {"horizon": "#898781", "fixed_frac": "#eb6834", "oracle_teps": "#0b0b0b"}


def _fig_costs(calib: pd.DataFrame, metric: str, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = calib[calib["lambda"] == MAIN_LAMBDA]
    tests = list(c.groupby(["app", "split"], sort=False).groups)
    fams = [f for f in family_order(metric) if f in set(c["family"])]
    fig, axes = plt.subplots(len(tests), 1, figsize=(8, 2.3 * len(tests) + 0.6), squeeze=False)
    for ax, (app, split) in zip(axes[:, 0], tests):
        g = c[(c["app"] == app) & (c["split"] == split)].set_index("family")
        for k, fam in enumerate(fams):
            if fam not in g.index:
                continue
            r = g.loc[fam]
            col = _COL.get(fam, "#184f95" if fam.endswith("best") else "#2a78d6")
            ax.plot([r["ci_lo"], r["ci_hi"]], [k, k], color=col, lw=1.6)
            ax.plot(r["test_cost"], k, "o", color=col, ms=6)
            ax.text(r["ci_hi"], k, f"  {r['test_cost']:.3f}", va="center", fontsize=8)
        ax.set_yticks(range(len(fams)), fams, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{app}: {SPLIT_NAMES[(app, split)]} ({int(g['test_runs'].iloc[0])} held-out runs)",
                     loc="left", fontsize=9)
        ax.grid(axis="x", color="#e1e0d9")
    axes[-1, 0].set_xlabel(f"held-out cost at lambda = {MAIN_LAMBDA} (dot: mean, line: 95% CI)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _fig_signal(series: pd.DataFrame, meta: pd.DataFrame, metric: str, path: Path) -> Path:
    """Median over runs that learned of test accuracy (share of best) and the metric's
    signal, against step / t_eps, per application (per-run interpolation, then median)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col, label = {"N1": ("w_norm", "weight norm / its value at t_eps"),
                  "R1": ("erank_mean", "mean effective rank / its value at t_eps"),
                  "B5": ("alpha_mean", "mean HTSR alpha")}[metric]
    m = meta[(meta["t_eps"] > 0) & (meta["max_test"] >= 0.5)][["run_id", "app", "t_eps", "max_test"]]
    df = series.merge(m, on="run_id")
    apps = [a for a in APPS if (df["app"] == a).any()]
    fig, axes = plt.subplots(2, max(1, len(apps)), figsize=(4.2 * max(1, len(apps)), 6), squeeze=False)
    xs = np.arange(-2.0, 1.51, 0.05)
    for j, app in enumerate(apps):
        d = df[(df["app"] == app) & (df["step"] > 0)]
        runs = list(d.groupby("run_id"))
        for row, (ycol, ylab, color) in enumerate((("acc", "test accuracy / best", "#1baf7a"), (col, label, "#2a78d6"))):
            Y = np.full((len(runs), len(xs)), np.nan)
            for k, (_, g) in enumerate(runs):
                x = np.log10(g["step"].to_numpy() / g["t_eps"].iloc[0])
                if ycol == "acc":
                    y = g["test_acc"].to_numpy() / g["max_test"].iloc[0]
                else:
                    y = g[ycol].to_numpy(dtype=float)
                    if metric != "B5":
                        i0 = int(np.argmin(np.abs(x)))
                        y = y / y[i0] if np.isfinite(y[i0]) and y[i0] > 0 else np.full(len(y), np.nan)
                inside = (xs >= x.min()) & (xs <= x.max())
                ok = np.isfinite(y)
                if ok.sum() >= 2:
                    Y[k, inside] = np.interp(xs[inside], x[ok], y[ok])
            good = np.isfinite(Y).sum(0) >= max(1, 0.5 * len(runs))
            ax = axes[row, j]
            if good.any():
                med, lo, hi = (np.nanpercentile(Y[:, good], q, axis=0) for q in (50, 25, 75))
                ax.fill_between(10 ** xs[good], lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(10 ** xs[good], med, color=color, lw=2)
            ax.axvline(1, color="#0b0b0b", lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_ylabel(ylab, fontsize=8)
            ax.grid(color="#e1e0d9")
            if row == 0:
                ax.set_title(f"{app} ({len(runs)} runs that learned)", loc="left", fontsize=9)
            else:
                ax.set_xlabel("step / t_eps")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def build_reports(rows: pd.DataFrame, series: pd.DataFrame, meta: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """One folder per metric: report_<M>/REPORT_auto.md, figures/, tables/ (calibration,
    every candidate x test x lambda; selection). Returns remote path -> local file."""
    out_dir = Path(out_dir)
    files: dict[str, Path] = {}
    for metric in METRICS:
        d = out_dir / f"report_{metric}"
        (d / "tables").mkdir(parents=True, exist_ok=True)
        calib = calibration_table(rows, meta, metric)
        if calib.empty:
            continue
        calib.to_csv(d / "tables" / "calibration.csv", index=False)
        meta.to_csv(d / "tables" / "selection.csv", index=False)
        figs = [_fig_costs(calib, metric, d / "figures" / "cost_heldout.png"),
                _fig_signal(series, meta, metric, d / "figures" / "signal_vs_accuracy.png")]
        md = [f"# {metric} ({METRIC_NAMES[metric]}) as a stopping rule: machine summary", "",
              f"{meta['run_id'].nunique()} runs ({', '.join(f'{a} {n}' for a, n in meta['app'].value_counts().items())}). "
              f"cost = regret + lambda * (stop - t_eps)+ / T. Each family's setting is tuned on the calibration "
              f"runs and scored on held-out runs; 95% CIs by cluster bootstrap over cells; verdicts from the "
              f"paired CI against the tuned fixed fraction.", ""]
        for (app, split), g in calib[calib["lambda"] == MAIN_LAMBDA].groupby(["app", "split"], sort=False):
            t = g.assign(vs_fixed=[verdict(a, b) if f != "fixed_frac" else "" for a, b, f in
                                   zip(g["d_fixed_lo"], g["d_fixed_hi"], g["family"])],
                         vs_never=[verdict(a, b) if f != "horizon" else "" for a, b, f in
                                   zip(g["d_horizon_lo"], g["d_horizon_hi"], g["family"])])
            md += [f"## {app}, {SPLIT_NAMES[(app, split)]}, lambda = {MAIN_LAMBDA}", "",
                   _md(t[["family", "setting", "cal_cost", "test_cost", "ci_lo", "ci_hi", "premature", "never",
                          "d_fixed", "vs_fixed", "vs_never"]]), ""]
        md += ["## Every lambda: the tuned pick against the fixed fraction", "",
               _md(calib[calib["family"] == f"{metric} best"][["app", "split", "lambda", "setting", "test_cost",
                                                               "d_fixed", "d_fixed_lo", "d_fixed_hi"]]), ""]
        md += [f"![{p.stem}](figures/{p.name})" for p in figs]
        (d / "REPORT_auto.md").write_text("\n".join(md), encoding="utf-8")
        for p in [d / "REPORT_auto.md", d / "tables" / "calibration.csv", d / "tables" / "selection.csv", *figs]:
            files[str(p.relative_to(out_dir)).replace(os.sep, "/")] = p
    return files

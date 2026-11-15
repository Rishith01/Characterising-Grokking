"""Final F2 study (2026-11-15): calibrate on some runs, score on held-out runs, and
compare against the simpler and the oracle versions of the same idea.

Questions (see report_AGOP/REPORT.md):
1. Modular: does a calibrated F2 rule beat training to the horizon, and the best
   calibrated fixed-time stop, on held-out seeds and on held-out operations?
2. MNIST: show that it does not, even with oracle knowledge of the final features.
3. Related metrics: Mallinar's alignment with the *final* AGOP, cos(M_t, M_T), used as
   an oracle stopping rule and as the ground truth the online estimates are compared to.
4. The constant-delta rule: angle between M_t and M_{t - Delta} for a fixed Delta,
   below a threshold. No extrapolation, no kappa: the simplest version of F2.

Evaluation side (reads test curves to score stops), so it is exempt from the quarantine
test, like stopping.agop_study. The traces are the agop_path signal's, so traces from
the earlier F2 experiment (phase4/agop_study) are reused.

Speed: a replay retrains a run step by step from step 0 (keyframes are 10k steps apart),
and one small MLP leaves the GPU mostly idle. replay_parallel therefore runs several
replays at once, each in its own process (`python -m stopping.agop_final worker ...`),
the way the corpus itself was trained. analyze_parallel does the same for the offline
analysis (CPU only).

Families scored (each with a grid of settings; the setting is chosen on the calibration
runs, then scored on the test runs):
- bound:       remaining-angle bound R = theta q / (1 - q) (stopping.agop_study)
- rel_delta:   angle over the last lag f*t below eps (the bound without kappa)
- const_delta: angle over a fixed lag Delta = c*T below eps (T = the training budget)
- msd, geweke: settling tests (stopping.agop_study)
- fixed_frac:  stop at c*T whatever happens (not F2; the bar F2 has to clear)
- oracle_cos:  first t with cos(M_t, M_T) >= c (needs the final M: not causal)
- oracle_settle: first t after which the angle to M_T stays below delta (not causal)
- oracle_teps: stop exactly at t_eps; horizon: never stop.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import agop_study as st
from .signals.agop import AgopPathConfig
from .store import TraceStore

CAT_ORDER = ("grokker", "fast", "never", "late_collapse", "joint_plateau", "no_fit")
SEEDS = (0, 1, 2, 3)
CONST_FRACS = (0.005, 0.01, 0.02, 0.05, 0.1)  # Delta = c * T
CONST_WINDOWS = ((1, False), (3, True))
EPS_THETA = (0.003, 0.01, 0.03, 0.1, 0.3)  # thresholds on a measured angle (rad)
FIXED_FRACS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
ORACLE_COS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)
ORACLE_DELTA = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
ONLINE = ("bound", "rel_delta", "const_delta", "msd", "geweke")
ORACLES = ("oracle_cos", "oracle_settle", "oracle_teps")
FAMILY_ORDER = ("horizon", "fixed_frac") + ONLINE + ("F2 best",) + ORACLES
SPLITS = {"modular": {"seeds": ("seed", (0, 1), (2, 3)),
                      "operation": ("variant", ("modular x+y",), ("modular x*y", "modular x/y"))},
          "mnist": {"seeds": ("seed", (0, 1), (2, 3)),
                    "depth": ("variant", ("mnist d3",), ("mnist d8",))}}
DEFAULT_CELLS = {"modular": {"modular x+y": 6, "modular x*y": 2, "modular x/y": 2},
                 "mnist": {"mnist d3": 2, "mnist d8": 2}}
META_COLS = ("run_id", "app", "variant", "optim", "lr", "gamma", "alpha", "size", "seed", "category", "cell",
             "cell_category", "max_test", "t_eps", "train99_step", "wall_s")


# ---- run selection ----------------------------------------------------------------------

def cell_key(r: dict) -> str:
    return f"{r['variant']}|{r['optim']}|lr{r['lr']}|g{r['gamma']}|a{r['alpha']}|s{r.get('size')}"


def pick_final_runs(rows: list[dict], app: str, cells_per_group, variants: Optional[Sequence[str]] = None,
                    categories: Sequence[str] = CAT_ORDER, seeds: Sequence[int] = SEEDS) -> list[dict]:
    """Whole hyperparameter cells (all `seeds`), so that calibration and test runs share
    hyperparameters in the seed split. Per (variant, optimizer), `cells_per_group` cells
    (an int, or a dict variant -> int), picked round-robin over the seed-0 categories in
    `categories` order, each time the middle remaining candidate (sorted by alpha, gamma,
    size). Adds app, cell and cell_category (the seed-0 category)."""
    ok = [r for r in rows if r.get("status", "complete") == "complete" and r.get("category") != "diverged"]
    if variants:
        ok = [r for r in ok if r["variant"] in set(variants)]
    cells: dict[str, dict[int, dict]] = {}
    for r in ok:
        cells.setdefault(cell_key(r), {})[int(r["seed"])] = r
    full = {k: v for k, v in cells.items() if all(s in v for s in seeds)}
    out = []
    for v, o in sorted({(c[0]["variant"], c[0]["optim"]) for c in full.values()}):
        n = cells_per_group.get(v, 0) if isinstance(cells_per_group, dict) else int(cells_per_group)
        group = [c for c in full.values() if c[0]["variant"] == v and c[0]["optim"] == o]
        by_cat = {cat: sorted((c for c in group if c[0]["category"] == cat),
                              key=lambda c: (c[0]["alpha"], c[0]["gamma"], c[0].get("size") or 0, c[0]["run_id"]))
                  for cat in categories}
        chosen = []
        while len(chosen) < n and any(by_cat.values()):
            for cat in categories:
                if len(chosen) >= n:
                    break
                if by_cat[cat]:
                    chosen.append((cat, by_cat[cat].pop(len(by_cat[cat]) // 2)))
        for cat, c in chosen:
            for s in seeds:
                out.append({**c[s], "app": app, "cell": cell_key(c[s]), "cell_category": cat})
    return out


# ---- process pool over a shared run queue ------------------------------------------------
#
# The main process writes spec.json (no secrets: the HF token goes in the environment) and
# queue.json, then starts N workers. Each worker claims the next unclaimed run with an
# exclusive file create, so the work balances itself and nothing is done twice.

def _claim(work: Path, mode: str, rid: str) -> bool:
    d = Path(work) / f"claims_{mode}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / rid, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _trace_done(store: TraceStore, cfg: AgopPathConfig, work: Path, rid: str) -> bool:
    return store.has(cfg, rid) and st._curves_path(work, rid).exists()


def _analysis_paths(out: Path, rid: str) -> dict:
    out = Path(out)
    return {"rows": out / "rows" / f"{rid}.parquet", "series": out / "series" / f"{rid}.parquet",
            "diag": out / "diag" / f"{rid}.json"}


def _analysis_done(out: Path, rid: str) -> bool:
    return all(p.exists() for p in _analysis_paths(out, rid).values())


def _launch(mode: str, spec: dict, selection: list[dict], workers: int, work: Path, project_dir: Path,
            done: Callable[[str], bool], on_poll: Callable[[list[str]], None], poll: float,
            log: Callable[[str], None], env: Optional[dict]) -> list[dict]:
    work = Path(work)
    for d in (f"claims_{mode}", f"status_{mode}", f"logs_{mode}"):
        (work / d).mkdir(parents=True, exist_ok=True)
    for p in (work / f"claims_{mode}").iterdir():  # stale claims of an earlier, interrupted call
        p.unlink()
    for p in (work / f"status_{mode}").iterdir():
        p.unlink()
    spec_path = work / f"spec_{mode}.json"
    spec_path.write_text(json.dumps(spec, indent=2, default=str))
    (work / f"queue_{mode}.json").write_text(json.dumps([{k: r.get(k) for k in META_COLS} for r in selection],
                                                       default=str))
    todo = [r["run_id"] for r in selection if not done(r["run_id"])]
    log(f"{mode}: {len(selection) - len(todo)} of {len(selection)} runs already done; {len(todo)} to do "
        f"with {workers} worker(s)")
    if not todo:
        on_poll([])
        return []
    procs = []
    for i in range(min(workers, len(todo))):
        fh = open(work / f"logs_{mode}" / f"w{i}.log", "a")
        procs.append((subprocess.Popen([sys.executable, "-m", "stopping.agop_final", mode, str(spec_path), str(i)],
                                       cwd=str(project_dir), env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    t0, seen = time.time(), set()
    while True:
        running = [p for p, _ in procs if p.poll() is None]
        new = [rid for rid in todo if rid not in seen and done(rid)]
        seen.update(new)
        on_poll(new)
        el = time.time() - t0
        eta = el / len(seen) * (len(todo) - len(seen)) if seen else float("nan")
        if new or not running:
            log(f"{mode}: {len(seen)}/{len(todo)} done, {el / 60:.1f} min elapsed, ETA {eta / 60:.0f} min, "
                f"{len(running)} worker(s) running")
        if not running:
            break
        time.sleep(poll)
    for _, fh in procs:
        fh.close()
    failed = [i for i, (p, _) in enumerate(procs) if p.returncode != 0]
    if failed:
        tail = (work / f"logs_{mode}" / f"w{failed[0]}.log").read_text()[-3000:]
        log(f"{mode}: worker(s) {failed} exited with an error; log tail of w{failed[0]}:\n{tail}")
    status = []
    for f in sorted((work / f"status_{mode}").glob("*.jsonl")):
        status += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return status


def replay_parallel(selection: list[dict], store: TraceStore, cfg: AgopPathConfig, work: Path, project_dir: Path,
                    workers: int = 4, source: Optional[dict] = None, eval_kw: Optional[dict] = None,
                    device: str = "cuda", out_remote=None, upload_every: int = 10, poll: float = 15.0,
                    mem_fraction: Optional[float] = None, env: Optional[dict] = None,
                    log: Callable[[str], None] = print) -> list[dict]:
    """Replay every selected run that has no trace yet, `workers` runs at a time, each in
    its own process. source: {"kind": "hf", "repo": ..., "allow_env": bool} or
    {"kind": "dir", "root": ...} (tests); add "verify": True to check every replay against
    the recorded telemetry, predictions and keyframes (a run that diverges fails). The HF
    token, if any, must be in env["HF_TOKEN"]. mem_fraction caps each process's share of
    GPU memory. Runs are queued longest first (corpus wall time), so the last runs to
    finish are short ones. Uploads the new trace parts and curves every `upload_every`
    finished runs."""
    selection = sorted(selection, key=lambda r: -float(r.get("wall_s") or 0))
    work = Path(work)
    rel = f"signals/{cfg.name}/{cfg.hash()}"
    cfg_path = store.dir(cfg) / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    (store.dir(cfg) / "parts").mkdir(parents=True, exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    spec = {"store": str(store.root), "cfg": {k: v for k, v in cfg.to_dict().items() if k not in ("signal", "version")},
            "work": str(work), "source": source or {"kind": "hf", "repo": "advay333/tfs"}, "eval_kw": eval_kw,
            "device": device, "mem_fraction": mem_fraction}
    pending: dict[str, Path] = {}

    def flush(msg):
        if out_remote is not None and pending:
            out_remote.commit(dict(pending), message=msg)
            log(f"uploaded {len(pending)} file(s)")
            pending.clear()

    def on_poll(new):
        for rid in new:
            pending[f"{rel}/parts/{rid}.parquet"] = store.dir(cfg) / "parts" / f"{rid}.parquet"
            pending[f"curves/{rid}.json"] = st._curves_path(work, rid)
            pending[f"{rel}/config.json"] = cfg_path
        if len([k for k in pending if k.startswith("curves/")]) >= upload_every:
            flush("agop final: traces")

    status = _launch("worker", spec, selection, workers, work, project_dir,
                     lambda rid: _trace_done(store, cfg, work, rid), on_poll, poll, log, env)
    flush("agop final: traces")
    return status


def analyze_parallel(selection: list[dict], store: TraceStore, cfg: AgopPathConfig, work: Path, out: Path,
                     project_dir: Path, workers: int = 4, poll: float = 5.0, env: Optional[dict] = None,
                     log: Callable[[str], None] = print) -> list[dict]:
    """analyze_final_run for every selected run with a trace, in `workers` processes.
    Per-run results go to out/rows, out/series and out/diag (cached)."""
    have = [r for r in selection if _trace_done(store, cfg, work, r["run_id"])]
    spec = {"store": str(store.root), "cfg": {k: v for k, v in cfg.to_dict().items() if k not in ("signal", "version")},
            "work": str(work), "out": str(out)}
    return _launch("analyze", spec, have, workers, work, project_dir, lambda rid: _analysis_done(out, rid),
                   lambda new: None, poll, log, env)


def _worker_main(mode: str, spec_path: str, idx: int) -> None:
    spec = json.loads(Path(spec_path).read_text())
    work = Path(spec["work"])
    cfg = AgopPathConfig(**spec["cfg"])
    store = TraceStore(Path(spec["store"]))
    queue = json.loads((work / f"queue_{mode}.json").read_text())
    status_f = work / f"status_{mode}" / f"w{idx}.jsonl"

    def log(msg):
        print(f"[w{idx} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    sources = {}
    if mode == "worker":
        from tfs.training import determinism
        determinism.apply()
        import torch
        from .compute import compute
        from .pilot import _curves
        src = spec["source"]
        verify = bool(src.get("verify"))
        if torch.cuda.is_available() and spec.get("mem_fraction"):
            # a cap per process, so one runaway replay cannot starve the others
            torch.cuda.set_per_process_memory_fraction(float(spec["mem_fraction"]))
        for app in sorted({r["app"] for r in queue}):
            if src["kind"] == "dir":
                from .tfs_source import DirRunSource
                sources[app] = DirRunSource(Path(src["root"]), device=spec["device"], verify_replay=verify)
            else:
                if app == "mnist":
                    from tfs.tasks.mnist import ensure_mnist
                    ensure_mnist()
                from .tfs_source import HfRunSource
                sources[app] = HfRunSource(src["repo"], f"phase3/{app}", work=work / f"runs_w{idx}",
                                           device=spec["device"], token=os.environ.get("HF_TOKEN"),
                                           allow_env_mismatch=bool(src.get("allow_env")), verify_replay=verify)
    for r in queue:
        rid = r["run_id"]
        if mode == "worker":
            if _trace_done(store, cfg, work, rid) or not _claim(work, mode, rid):
                continue
            t0 = time.time()
            for attempt in range(3):  # out of GPU memory (other replays peaking at once): wait and retry
                s = compute(sources[r["app"]], [rid], [cfg], store, eval_kw=spec.get("eval_kw"), log=log)[0]
                if s["ok"] or "out of memory" not in s["error"].lower():
                    break
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log(f"{rid}: out of GPU memory, retry {attempt + 1} in 60 s")
                time.sleep(60)
            err = s["error"]
            if s["ok"]:
                try:
                    cp = st._curves_path(work, rid)
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    tmp = cp.with_suffix(".tmp")
                    tmp.write_text(json.dumps(_curves(sources[r["app"]], rid)))
                    tmp.replace(cp)
                except Exception as e:
                    err = f"curves: {type(e).__name__}: {e}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if _analysis_done(Path(spec["out"]), rid) or not _claim(work, mode, rid):
                continue
            t0 = time.time()
            err = ""
            try:
                write_analysis(store.read(cfg, rid), json.loads(st._curves_path(work, rid).read_text()), r,
                               Path(spec["out"]))
            except Exception as e:
                import traceback
                err = f"{type(e).__name__}: {e}"
                log(traceback.format_exc())
        row = {"run_id": rid, "app": r["app"], "ok": not err, "seconds": round(time.time() - t0, 1), "error": err,
               "worker": idx}
        with open(status_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"{rid}: {'ok' if not err else 'FAILED ' + err} in {row['seconds']} s")


# ---- offline analysis of one run ---------------------------------------------------------

def const_lags(steps: np.ndarray, D: np.ndarray, delta: float, w: int, floor: bool) -> np.ndarray:
    """Angle between the window mean ending at eval index i and the one ending at the last
    step <= t_i - delta (constant lag). NaN until t_i >= delta (and w points fit)."""
    k = len(steps)
    th = np.full(k, np.nan)
    for i in range(k):
        if i - w + 1 < 0:
            continue
        A = np.arange(i - w + 1, i + 1)
        B = st._ending_at(steps, steps[i] - delta, w, A[0])
        if B is None:
            continue
        t = float(st.angle(st.mean_chord2(D, A, B)))
        if w > 1 and floor:
            s2 = (D[np.ix_(A, A)].mean() / 2 + D[np.ix_(B, B)].mean() / 2) / (w - 1)
            t = math.sqrt(max(t * t - s2, 0.0))
        th[i] = t
    return th


def scalar_const(steps: np.ndarray, y: np.ndarray, delta: float, w: int, floor: bool) -> np.ndarray:
    k = len(steps)
    out = np.full(k, np.nan)
    for i in range(k):
        if i - w + 1 < 0:
            continue
        A = np.arange(i - w + 1, i + 1)
        B = st._ending_at(steps, steps[i] - delta, w, A[0])
        if B is None:
            continue
        d = abs(y[A].mean() - y[B].mean())
        if floor and w > 1:
            d = math.sqrt(max(d * d - (y[A].var() + y[B].var()) / (w - 1), 0.0))
        out[i] = d
    return out


def _mats(tr: st.PathTrace, obj: str) -> list:
    return tr.D["W"] if obj == "W" else [tr.D[obj]]


def _stop_rows(steps, cond, curves, info, family, setting, extra=None) -> dict:
    stop = st.first_stop(steps, cond)
    return {"run_id": info["run_id"], "family": family, "setting": setting, **(extra or {}),
            **st.score_stop(curves, stop, info)}


def analyze_final_run(df: pd.DataFrame, curves: dict, info: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(rows: one per family x setting, series for plots, per-run diagnostics)."""
    tr = st.from_trace(df)
    steps = tr.steps
    T = float(curves["step"][-1])
    logn = np.log(np.maximum(tr.w_norm, 1e-300))
    rows: list[dict] = []
    keep: dict = {}

    def nn(x):
        return np.nan_to_num(x, nan=np.inf)

    # bound and rel_delta: one path_lags per (object, lag, window), both kappa estimators
    for obj in st.OBJECTS:
        for lag in st.LAG_FRACS:
            for w, floor in st.WINDOWS:
                per = [st.path_lags(steps, D, lag, w, floor) for D in _mats(tr, obj)]
                th1 = np.max(np.vstack([p[0] for p in per]), axis=0)
                n1, nd1, n2, nd2 = st.scalar_lags(steps, logn, lag, w, floor)
                tag = f"{obj} lag={lag} w={w} floor={int(floor)}"
                for eps in EPS_THETA:
                    for gate in st.GATES:
                        cond = nn(th1) <= eps
                        if gate == "norm":
                            cond &= nn(n1) <= eps
                        rows.append(_stop_rows(steps, cond, curves, info, "rel_delta",
                                               f"rel_delta {tag} eps={eps} gate={gate}"))
                for est in st.ESTIMATORS:
                    R = np.max(np.vstack([st.bound(steps, *p[:4], est)[1] for p in per]), axis=0)
                    Rn = st.bound(steps, n1, nd1, n2, nd2, est)[1]
                    if (obj, lag, w, floor, est) == st._DEFAULT_KEY:
                        keep["R"] = R
                    for eps in st.EPS_R:
                        for gate in st.GATES:
                            cond = nn(R) <= eps
                            if gate == "norm":
                                cond &= nn(Rn) <= st.EPS_NORM
                            rows.append(_stop_rows(steps, cond, curves, info, "bound",
                                                   f"bound {tag} est={est} eps={eps} gate={gate}"))
    # const_delta
    for obj in st.OBJECTS:
        for c in CONST_FRACS:
            for w, floor in CONST_WINDOWS:
                th = np.max(np.vstack([const_lags(steps, D, c * T, w, floor) for D in _mats(tr, obj)]), axis=0)
                nth = scalar_const(steps, logn, c * T, w, floor)
                if (obj, c, w) == ("M", 0.02, 3):
                    keep["theta_const"] = th
                for eps in EPS_THETA:
                    for gate in st.GATES:
                        cond = nn(th) <= eps
                        if gate == "norm":
                            cond &= nn(nth) <= eps
                        rows.append(_stop_rows(steps, cond, curves, info, "const_delta",
                                               f"const_delta {obj} c={c} w={w} floor={int(floor)} eps={eps} gate={gate}"))
    # settling tests (family msd / geweke)
    for r in st.settle_rows(tr, curves, info):
        rows.append({k: r[k] for k in ("run_id", "family", "setting")} | st.score_stop(curves, r["stop"], info))
    # fixed fraction of the budget
    for c in FIXED_FRACS:
        rows.append(_stop_rows(steps, steps >= c * T, curves, info, "fixed_frac", f"fixed_frac c={c}"))
    # oracles that know the final features
    align = {}
    for obj in st.OBJECTS:
        cos = np.min(np.vstack([1 - D[:, -1] / 2 for D in _mats(tr, obj)]), axis=0)
        ang = np.max(np.vstack([st.angle(D[:, -1]) for D in _mats(tr, obj)]), axis=0)
        align[obj] = cos
        for c in ORACLE_COS:
            stop = st.first_stop(steps, cos >= c, patience=1)
            rows.append({"run_id": info["run_id"], "family": "oracle_cos", "setting": f"oracle_cos {obj} c={c}",
                         **st.score_stop(curves, stop, info)})
        later_max = np.maximum.accumulate(ang[::-1])[::-1]  # max angle to M_T from i onwards
        for d in ORACLE_DELTA:
            stop = st.first_stop(steps, later_max <= d, patience=1)
            rows.append({"run_id": info["run_id"], "family": "oracle_settle",
                         "setting": f"oracle_settle {obj} delta={d}", **st.score_stop(curves, stop, info)})
        if obj == "M":
            keep["true_M"] = ang
    for b in st.baseline_rows(curves, info):
        fam = "horizon" if b["setting"].startswith("horizon") else "oracle_teps"
        rows.append({"run_id": info["run_id"], "family": fam, "setting": fam,
                     **{k: b[k] for k in b if k not in ("run_id", "app", "category", "optim", "family", "setting")}})
    out = pd.DataFrame(rows)

    test = np.array([st._acc_at(curves, int(s)) for s in steps])
    series = pd.DataFrame({"run_id": info["run_id"], "step": steps, "t_frac": steps / T, "test_acc": test,
                           "align_M": align["M"], "align_G": align["G"], "align_W": align["W"],
                           "true_M": keep["true_M"], "R_default": keep["R"], "theta_const": keep["theta_const"],
                           "w_norm": tr.w_norm, "erank_G": tr.erank_G})
    diag = {"run_id": info["run_id"], "eval_rows": int(len(steps))}
    t_eps, t_mem = info.get("t_eps"), info.get("train99_step")
    for name, t in (("teps", t_eps), ("tmem", t_mem)):
        if t is not None and not pd.isna(t) and t > 0:
            i = min(int(np.searchsorted(steps, t)), len(steps) - 1)
            diag[f"align_M_at_{name}"] = float(align["M"][i])
            diag[f"angle_M_at_{name}"] = float(keep["true_M"][i])
            diag[f"logw_change_after_{name}"] = float(logn[-1] - logn[i])
            diag[f"erank_ratio_after_{name}"] = float(tr.erank_G[-1] / tr.erank_G[i]) if tr.erank_G[i] > 0 else np.nan
    after = steps >= (t_mem or 0)
    for name in ("R_default", "theta_const"):
        x, y = series[name].to_numpy(), keep["true_M"]
        m = after & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        diag[f"spearman_{name}"] = float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank())) if m.sum() >= 8 else np.nan
    return out, series, diag


def write_analysis(df: pd.DataFrame, curves: dict, info: dict, out: Path) -> None:
    rows, series, diag = analyze_final_run(df, curves, info)
    p = _analysis_paths(out, info["run_id"])
    for k in p:
        p[k].parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(p["rows"], index=False)
    series.to_parquet(p["series"], index=False)
    p["diag"].write_text(json.dumps(diag, default=float))


def load_analysis(out: Path, run_ids: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, series, diag = [], [], []
    for rid in run_ids:
        if not _analysis_done(out, rid):
            continue
        p = _analysis_paths(out, rid)
        rows.append(pd.read_parquet(p["rows"]))
        series.append(pd.read_parquet(p["series"]))
        diag.append(json.loads(p["diag"].read_text()))
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()  # noqa: E731
    return cat(rows), cat(series), pd.DataFrame(diag)


# ---- calibration and held-out scoring ----------------------------------------------------

def _cluster_ci(values: np.ndarray, clusters: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the mean, resampling whole hyperparameter cells (the seeds of
    one cell are not independent)."""
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) < 2:
        return (np.nan, np.nan)
    u, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u)).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(B, len(u)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _sign_test(diff: np.ndarray, clusters: np.ndarray) -> tuple[int, int, float]:
    """Two-sided sign test on the per-cell mean difference (cells, not runs, are the
    independent units). Returns (cells better, cells worse, p)."""
    d = pd.Series(diff).groupby(clusters).mean()
    d = d[np.isfinite(d)]
    w, l = int((d < -1e-9).sum()), int((d > 1e-9).sum())
    n = w + l
    if n == 0:
        return w, l, 1.0
    k = min(w, l)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return w, l, p


def calibrate(rows: pd.DataFrame, meta: pd.DataFrame, cal: set, test: set, lam: float) -> pd.DataFrame:
    """Per family: the setting with the lowest mean cost_lam on the calibration runs,
    scored on the test runs, plus the hindsight-best setting on the test runs (an upper
    bound for that family). 'F2 best' picks across all online F2 families."""
    col = f"cost_{lam}"
    df = rows.merge(meta[["run_id", "cell", "category"]], on="run_id", how="left")
    c_df, t_df = df[df["run_id"].isin(cal)], df[df["run_id"].isin(test)]
    cal_mean = c_df.groupby(["family", "setting"])[col].mean()
    test_mean = t_df.groupby(["family", "setting"])[col].mean()
    tpiv = t_df.pivot_table(index="run_id", columns="setting", values=col, aggfunc="first")
    cells = t_df.drop_duplicates("run_id").set_index("run_id")["cell"].reindex(tpiv.index).to_numpy()
    picks = {}
    for fam in FAMILY_ORDER:
        if fam == "F2 best":
            sub = cal_mean[cal_mean.index.get_level_values(0).isin(ONLINE)]
        else:
            sub = cal_mean[cal_mean.index.get_level_values(0) == fam]
        if len(sub):
            picks[fam] = sub.idxmin()[1]
    hz = tpiv[picks["horizon"]].to_numpy()
    ff = tpiv[picks["fixed_frac"]].to_numpy() if "fixed_frac" in picks else np.full(len(hz), np.nan)
    out = []
    for fam, setting in picks.items():
        v = tpiv[setting].to_numpy()
        tr = t_df[t_df["setting"] == setting]
        lo, hi = _cluster_ci(v, cells)
        dh, dfx = v - hz, v - ff
        w, l, p = _sign_test(dh, cells)
        wf, lf, pf = _sign_test(dfx, cells)
        fam_test = test_mean[test_mean.index.get_level_values(0).isin(ONLINE if fam == "F2 best" else [fam])]
        out.append({
            "family": fam, "setting": setting, "lambda": lam, "cal_runs": c_df["run_id"].nunique(),
            "test_runs": int(np.isfinite(v).sum()), "test_cells": len(set(cells)),
            "cal_cost": float(cal_mean.loc[(tr["family"].iloc[0], setting)]) if len(tr) else np.nan,
            "test_cost": float(np.nanmean(v)), "ci_lo": lo, "ci_hi": hi,
            "premature": float(tr["premature"].mean()), "never": float(tr["never"].mean()),
            "regret": float(tr["regret"].mean()), "late": float(tr["late"].mean()),
            "d_horizon": float(np.nanmean(dh)), "d_horizon_lo": _cluster_ci(dh, cells)[0],
            "d_horizon_hi": _cluster_ci(dh, cells)[1], "cells_better_h": w, "cells_worse_h": l, "p_h": p,
            "d_fixed": float(np.nanmean(dfx)), "d_fixed_lo": _cluster_ci(dfx, cells)[0],
            "d_fixed_hi": _cluster_ci(dfx, cells)[1], "cells_better_f": wf, "cells_worse_f": lf, "p_f": pf,
            "hindsight_setting": fam_test.idxmin()[1] if len(fam_test) else None,
            "hindsight_cost": float(fam_test.min()) if len(fam_test) else np.nan,
        })
    return pd.DataFrame(out)


def split_ids(meta: pd.DataFrame, app: str, split: str) -> tuple[set, set]:
    col, a, b = SPLITS[app][split]
    m = meta[meta["app"] == app]
    return set(m.loc[m[col].isin(a), "run_id"]), set(m.loc[m[col].isin(b), "run_id"])


def calibration_table(rows: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = []
    for app in SPLITS:
        if not (meta["app"] == app).any():
            continue
        for split in SPLITS[app]:
            cal, test = split_ids(meta, app, split)
            if not cal or not test:
                continue
            for lam in st.LAMBDAS:
                t = calibrate(rows[rows["run_id"].isin(cal | test)], meta, cal, test, lam)
                out.append(t.assign(app=app, split=split))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def category_table(rows: pd.DataFrame, meta: pd.DataFrame, calib: pd.DataFrame, lam: float = st.MAIN_LAMBDA) -> pd.DataFrame:
    """Seed split, test runs: mean cost per seed-0 category of the picked F2 rule and the
    references."""
    out = []
    for app in calib["app"].unique():
        c = calib[(calib["app"] == app) & (calib["split"] == "seeds") & (calib["lambda"] == lam)].set_index("family")
        _, test = split_ids(meta, app, "seeds")
        df = rows[rows["run_id"].isin(test)].merge(meta[["run_id", "cell_category"]], on="run_id")
        for fam in ("horizon", "fixed_frac", "F2 best", "oracle_cos", "oracle_teps"):
            if fam not in c.index:
                continue
            sub = df[df["setting"] == c.loc[fam, "setting"]]
            for cat, g in sub.groupby("cell_category"):
                out.append({"app": app, "family": fam, "category": cat, "runs": len(g),
                            "cost": g[f"cost_{lam}"].mean(), "premature": g["premature"].mean(),
                            "regret": g["regret"].mean(), "late": g["late"].mean()})
    return pd.DataFrame(out)


# ---- report -------------------------------------------------------------------------------

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
    return plt


def _fam_color(fam: str) -> str:
    return {"horizon": INK2, "fixed_frac": "#8a8984", "bound": PALETTE[0], "rel_delta": PALETTE[6],
            "const_delta": PALETTE[1], "msd": PALETTE[2], "geweke": PALETTE[3], "F2 best": PALETTE[7],
            "oracle_cos": PALETTE[4], "oracle_settle": PALETTE[5], "oracle_teps": INK}.get(fam, INK2)


def fig_costs(calib: pd.DataFrame, split: str, lam: float, path: Path) -> Path:
    plt = _style()
    apps = [a for a in SPLITS if ((calib["app"] == a) & (calib["split"] == split)).any()]
    if not apps:  # e.g. the variant split has its own name per app
        apps = list(calib["app"].unique())
    fig, axes = plt.subplots(1, len(apps), figsize=(5.2 * len(apps), 4.2), squeeze=False)
    for ax, app in zip(axes[0], apps):
        sp = split if split in SPLITS[app] else list(SPLITS[app])[1]
        c = calib[(calib["app"] == app) & (calib["split"] == sp) & (calib["lambda"] == lam)].set_index("family")
        if "horizon" not in c.index:
            ax.set_axis_off()
            ax.set_title(f"{app}: no runs for the {sp} split", fontsize=9, color=INK2)
            continue
        fams = [f for f in FAMILY_ORDER if f in c.index][::-1]
        y = np.arange(len(fams))
        for i, f in enumerate(fams):
            r = c.loc[f]
            ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=_fam_color(f), lw=2, solid_capstyle="round")
            ax.plot(r["test_cost"], i, "o", ms=8, color=_fam_color(f), mec="#fcfcfb", mew=2)
            ax.plot(r["hindsight_cost"], i, "|", ms=10, color=INK2)
        ax.axvline(c.loc["horizon", "test_cost"], color=INK2, lw=0.8, ls=":")
        ax.set_yticks(y, [f"{f}\n(oracle)" if f.startswith("oracle") else f for f in fams], fontsize=8)
        ax.set_xlabel(f"mean test cost, lambda = {lam} (lower is better)")
        ax.set_title(f"{app}: calibrate {SPLITS[app][sp][1]}, test {SPLITS[app][sp][2]}", fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, "dot: calibrated setting on test runs, line: 95% cell-bootstrap CI, tick: best setting "
             "chosen on the test runs themselves (hindsight)", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_alignment(series: pd.DataFrame, meta: pd.DataFrame, path: Path) -> Path:
    """Median (IQR band) alignment with the final features and normalized test accuracy
    against t / t_eps, per application, generalizing runs only."""
    plt = _style()
    m = meta[(meta["t_eps"] > 0) & (meta["max_test"] >= 0.5)][["run_id", "app", "t_eps", "max_test"]]
    df = series.merge(m, on="run_id")
    df["x"] = np.log10(np.maximum(df["step"], 1) / df["t_eps"])
    df["acc_n"] = df["test_acc"] / df["max_test"]
    apps = sorted(df["app"].unique())
    fig, axes = plt.subplots(2, max(len(apps), 1), figsize=(5 * max(len(apps), 1), 6), sharex=True, squeeze=False)
    bins = np.linspace(-2, 1.5, 36)
    for j, app in enumerate(apps):
        g = df[df["app"] == app]
        g = g.assign(b=pd.cut(g["x"], bins, labels=(bins[:-1] + bins[1:]) / 2))
        for row, (col, lab) in enumerate((("align_M", "cos(M_t, M_final)"), ("acc_n", "test acc / max test acc"))):
            ax = axes[row, j]
            for k, (obj, color) in enumerate((("align_M", PALETTE[0]), ("align_G", PALETTE[1]), ("align_W", PALETTE[2]))
                                             if row == 0 else ((col, PALETTE[0]),)):
                gb = g.groupby("b", observed=True)
                q = gb[obj].quantile([0.25, 0.5, 0.75]).unstack()
                q = q[gb["run_id"].nunique().reindex(q.index) >= 5]  # bins with at least 5 runs
                xs = q.index.astype(float)
                ax.fill_between(xs, q[0.25], q[0.75], color=color, alpha=0.15, lw=0)
                ax.plot(xs, q[0.5], color=color, lw=2, label={"align_M": "M = G^1/2", "align_G": "G",
                                                              "align_W": "weights (min layer)"}.get(obj, None))
            ax.axvline(0, color=INK, lw=0.8, ls=":")
            ax.set_ylabel(lab)
            if row == 0:
                ax.set_title(f"{app} ({g['run_id'].nunique()} generalizing runs)")
                ax.legend(fontsize=7, loc="lower right")
        axes[1, j].set_xlabel("log10(step / t_eps)  (0 = test accuracy within eps of its best)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_online_vs_true(series: pd.DataFrame, meta: pd.DataFrame, path: Path) -> Path:
    """Online estimates (bound R, constant-delta angle) against the true remaining angle
    to the final M, after memorization."""
    plt = _style()
    df = series.merge(meta[["run_id", "app", "train99_step"]], on="run_id")
    df = df[df["step"] >= df["train99_step"].fillna(0)]
    apps = sorted(df["app"].unique())
    fig, axes = plt.subplots(2, max(len(apps), 1), figsize=(4.6 * max(len(apps), 1), 8), squeeze=False)
    for j, app in enumerate(apps):
        g = df[df["app"] == app]
        for row, (col, lab, color) in enumerate((("R_default", "bound R (M, lag 0.25 t, w 3)", PALETTE[0]),
                                                 ("theta_const", "constant-delta angle (M, Delta 0.02 T, w 3)", PALETTE[1]))):
            ax = axes[row, j]
            ok = np.isfinite(g[col]) & (g[col] > 0) & (g["true_M"] > 1e-4)
            x, y = g.loc[ok, col].to_numpy(), g.loc[ok, "true_M"].to_numpy()
            if len(x) > 6000:
                sel = np.random.default_rng(0).choice(len(x), 6000, replace=False)
                x, y = x[sel], y[sel]
            ax.scatter(x, y, s=6, alpha=0.25, color=color, lw=0)
            lo, hi = 1e-4, 4
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, 1e2 if row == 0 else hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(lab)
            ax.set_ylabel("true remaining angle to final M (rad)")
            if row == 0:
                ax.set_title(app)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_stops(rows: pd.DataFrame, meta: pd.DataFrame, calib: pd.DataFrame, path: Path, lam: float = st.MAIN_LAMBDA) -> Path:
    """Stop time against t_eps on the seed-split test runs, for the picked F2 rule and the
    picked fixed-fraction rule."""
    plt = _style()
    apps = list(calib["app"].unique())
    fig, axes = plt.subplots(1, len(apps), figsize=(4.8 * len(apps), 4.4), squeeze=False)
    for ax, app in zip(axes[0], apps):
        c = calib[(calib["app"] == app) & (calib["split"] == "seeds") & (calib["lambda"] == lam)].set_index("family")
        _, test = split_ids(meta, app, "seeds")
        T = meta.set_index("run_id")
        for k, fam in enumerate(("fixed_frac", "F2 best")):
            sub = rows[(rows["setting"] == c.loc[fam, "setting"]) & rows["run_id"].isin(test)].set_index("run_id")
            Tn = np.array([float(T.loc[r, "steps"]) if "steps" in T.columns else 1.0 for r in sub.index])
            te = T.loc[sub.index, "t_eps"].to_numpy(float) / Tn
            sp = sub["stop"].astype(float).to_numpy()
            sp = np.where(np.isfinite(sp), sp, Tn) / Tn
            ax.scatter(np.maximum(te, 1e-3), sp, s=22, color=_fam_color(fam), alpha=0.75, lw=0,
                       label=f"{fam}: {c.loc[fam, 'setting']}")
        ax.plot([1e-3, 1], [1e-3, 1], color=INK2, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t_eps / T")
        ax.set_ylabel("stop / T  (no stop = 1)")
        ax.set_title(f"{app}, seed-split test runs")
        ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_lambda(calib: pd.DataFrame, path: Path) -> Path:
    plt = _style()
    apps = list(calib["app"].unique())
    fig, axes = plt.subplots(1, len(apps), figsize=(4.8 * len(apps), 3.8), squeeze=False)
    for ax, app in zip(axes[0], apps):
        c = calib[(calib["app"] == app) & (calib["split"] == "seeds")]
        for k, fam in enumerate(("fixed_frac", "F2 best", "const_delta", "oracle_cos")):
            s = c[c["family"] == fam].sort_values("lambda")
            if s.empty:
                continue
            x = np.arange(len(s)) + (k - 1.5) * 0.12
            ax.errorbar(x, s["d_horizon"], yerr=[s["d_horizon"] - s["d_horizon_lo"], s["d_horizon_hi"] - s["d_horizon"]],
                        fmt="o", ms=6, color=_fam_color(fam), lw=1.5, label=fam)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(st.LAMBDAS)), [f"lambda={l}" for l in st.LAMBDAS])
        ax.set_ylabel("test cost minus horizon cost")
        ax.set_title(f"{app}, seed split (recalibrated per lambda)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_oracle_curve(rows: pd.DataFrame, meta: pd.DataFrame, path: Path, lam: float = st.MAIN_LAMBDA) -> Path:
    """Mean cost (all runs) of the oracle rules over their thresholds, against the horizon
    and the best fixed fraction: the most any F2-type rule could achieve."""
    plt = _style()
    df = rows.merge(meta[["run_id", "app"]], on="run_id")
    apps = sorted(df["app"].unique())
    fig, axes = plt.subplots(1, len(apps), figsize=(4.8 * len(apps), 3.8), squeeze=False)
    col = f"cost_{lam}"
    for ax, app in zip(axes[0], apps):
        g = df[df["app"] == app].groupby("setting")[col].mean()
        for k, (fam, vals, fmt) in enumerate((("oracle_cos", ORACLE_COS, "oracle_cos M c={}"),
                                              ("oracle_settle", ORACLE_DELTA, "oracle_settle M delta={}"))):
            ys = [g.get(fmt.format(v), np.nan) for v in vals]
            ax.plot(range(len(vals)), ys, "o-", color=_fam_color(fam), lw=2, ms=6,
                    label=f"{fam}: " + ("c = " if fam == "oracle_cos" else "delta = ") + ", ".join(map(str, vals)))
        ax.axhline(g["horizon"], color=INK2, lw=1, ls=":", label="horizon")
        best_ff = g[[s for s in g.index if s.startswith("fixed_frac")]].min()
        ax.axhline(best_ff, color="#8a8984", lw=1, ls="--", label="best fixed fraction (hindsight)")
        ax.axhline(g["oracle_teps"], color=INK, lw=1, label="stop at t_eps")
        ax.set_xticks(range(6), [f"#{i + 1}" for i in range(6)])
        ax.set_xlabel("threshold index (stricter to the right)")
        ax.set_ylabel(f"mean cost, lambda = {lam}")
        ax.set_title(f"{app}: oracles that know the final features")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_examples(series: pd.DataFrame, rows: pd.DataFrame, meta: pd.DataFrame, calib: pd.DataFrame, path: Path,
                 lam: float = st.MAIN_LAMBDA) -> Path:
    """Two typical grokking runs per application: test accuracy, alignment with the final
    features, and the two online estimates, with the stops of the picked rules."""
    plt = _style()
    picks = []
    for app in calib["app"].unique():
        g = meta[(meta["app"] == app) & (meta["cell_category"] == "grokker") & (meta["seed"] == 2)]
        picks += list(g.sort_values("t_eps")["run_id"].iloc[[len(g) // 4, (3 * len(g)) // 4]]) if len(g) >= 2 else list(g["run_id"])
    if not picks:
        return path
    fig, axes = plt.subplots(3, len(picks), figsize=(3.6 * len(picks), 7.5), sharex="col", squeeze=False)
    for j, rid in enumerate(picks):
        s = series[series["run_id"] == rid]
        info = meta.set_index("run_id").loc[rid]
        c = calib[(calib["app"] == info["app"]) & (calib["split"] == "seeds") & (calib["lambda"] == lam)].set_index("family")
        x = np.maximum(s["step"].to_numpy(float), 1)
        axes[0, j].plot(x, s["test_acc"], color=PALETTE[0], lw=2)
        axes[0, j].set_title(f"{info['app']} {info['optim']}\na={info['alpha']} g={info['gamma']} seed {info['seed']}", fontsize=8)
        axes[1, j].plot(x, s["align_M"], color=PALETTE[0], lw=2, label="M")
        axes[1, j].plot(x, s["align_W"], color=PALETTE[2], lw=2, label="weights")
        axes[2, j].plot(x, s["true_M"].where(s["true_M"] > 0), color=INK, lw=1.5, ls="--", label="true remaining (M)")
        axes[2, j].plot(x, s["R_default"].where(s["R_default"] > 0), color=PALETTE[0], lw=1.5, label="bound R")
        axes[2, j].plot(x, s["theta_const"].where(s["theta_const"] > 0), color=PALETTE[1], lw=1.5, label="const-delta angle")
        for fam in ("F2 best", "fixed_frac", "oracle_cos"):
            if fam in c.index:
                r = rows[(rows["run_id"] == rid) & (rows["setting"] == c.loc[fam, "setting"])]
                if len(r) and pd.notna(r["stop"].iloc[0]):
                    for a in axes[:, j]:
                        a.axvline(max(float(r["stop"].iloc[0]), 1), color=_fam_color(fam), lw=1.2, ls="-.")
        for a in axes[:, j]:
            a.axvline(max(float(info["t_eps"]), 1), color=INK, lw=0.8, ls=":")
        axes[2, j].set_yscale("log")
        axes[2, j].set_xscale("log")
        axes[2, j].set_xlabel("step")
    axes[0, 0].set_ylabel("test accuracy")
    axes[1, 0].set_ylabel("cos(., final)")
    axes[2, 0].set_ylabel("angle (rad)")
    axes[1, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=6)
    fig.text(0.01, 0.005, "dotted black: t_eps; dash-dot: stops of the picked rules (red F2 best, grey fixed "
             "fraction, pink final-AGOP oracle)", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _md_table(df: pd.DataFrame) -> str:
    from tfs.evaluation.report import rows_to_markdown
    return rows_to_markdown(df.to_dict("records"))


def build_report(rows: pd.DataFrame, series: pd.DataFrame, diag: pd.DataFrame, meta: pd.DataFrame,
                 out_dir: Path) -> dict:
    """report_AGOP/: REPORT_auto.md, figures/*.png, tables/*.csv. Returns {relative path: Path}."""
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    calib = calibration_table(rows, meta)
    cats = category_table(rows, meta, calib)
    d = diag.merge(meta[["run_id", "app", "cell_category", "optim"]], on="run_id")
    tables = {"calibration": calib, "categories": cats, "diagnostics": d, "selection": meta}
    files = {}
    for name, t in tables.items():
        p = out_dir / "tables" / f"{name}.csv"
        t.to_csv(p, index=False)
        files[f"tables/{name}.csv"] = p
    figs = [fig_costs(calib, "seeds", st.MAIN_LAMBDA, out_dir / "figures" / "cost_seed_split.png"),
            fig_costs(calib, "variant", st.MAIN_LAMBDA, out_dir / "figures" / "cost_variant_split.png"),
            fig_alignment(series, meta, out_dir / "figures" / "alignment_to_final.png"),
            fig_online_vs_true(series, meta, out_dir / "figures" / "online_vs_true.png"),
            fig_stops(rows, meta, calib, out_dir / "figures" / "stops_vs_teps.png"),
            fig_lambda(calib, out_dir / "figures" / "lambda_sensitivity.png"),
            fig_oracle_curve(rows, meta, out_dir / "figures" / "oracle_curves.png"),
            fig_examples(series, rows, meta, calib, out_dir / "figures" / "examples.png")]
    for p in figs:
        if Path(p).exists():
            files[f"figures/{Path(p).name}"] = Path(p)
    cols = ["family", "setting", "test_cost", "ci_lo", "ci_hi", "premature", "never", "d_horizon", "d_horizon_lo",
            "d_horizon_hi", "cells_better_h", "cells_worse_h", "p_h", "d_fixed", "d_fixed_lo", "d_fixed_hi", "p_f",
            "hindsight_cost"]
    md = ["# F2 final study: automatic summary", "",
          f"Runs analyzed: {meta.groupby('app').size().to_dict()}; cells: {meta.groupby('app')['cell'].nunique().to_dict()}.",
          "Cost = regret + lambda x (training after t_eps) / T. Settings are chosen on the calibration runs and "
          "scored on the test runs. CIs: 95% bootstrap over hyperparameter cells. p: two-sided sign test over cells.", ""]
    for (app, split), g in calib[calib["lambda"] == st.MAIN_LAMBDA].groupby(["app", "split"], sort=False):
        a, b = SPLITS[app][split][1], SPLITS[app][split][2]
        md += [f"## {app}, {split} split (calibrate {a}, test {b}), lambda = {st.MAIN_LAMBDA}", "",
               _md_table(g[cols].round(4)), ""]
    dd = d.groupby("app")[[c for c in d.columns if c.startswith(("align_", "angle_", "spearman_", "logw_", "erank_"))]].median()
    md += ["## Diagnostics (medians per application)", "", _md_table(dd.reset_index().round(3)), "",
           "## Per category (seed split, test runs)", "", _md_table(cats.round(4)), ""]
    p = out_dir / "REPORT_auto.md"
    p.write_text("\n".join(md))
    files["REPORT_auto.md"] = p
    return files


def build_meta(selection: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**{k: r.get(k) for k in META_COLS}, "steps": r.get("steps")} for r in selection])


def resume_from(remote, prefix_rel: str, store: TraceStore, cfg: AgopPathConfig, work: Path, run_ids: set,
                log: Callable[[str], None] = print) -> int:
    """Download trace parts and curves for `run_ids` that `remote` already has (this
    study's own output, or the earlier phase4/agop_study)."""
    rel = f"signals/{cfg.name}/{cfg.hash()}"
    got = 0
    try:
        files = dict(remote.list_files(rel + "/parts"))
        files.update(remote.list_files("curves"))
    except Exception as e:
        log(f"{prefix_rel}: nothing to resume ({type(e).__name__})")
        return 0
    for path in files:
        rid = Path(path).stem
        if rid not in run_ids:
            continue
        if path.startswith(rel + "/parts/"):
            dest = store.dir(cfg) / "parts" / Path(path).name
        elif path.startswith("curves/"):
            dest = st._curves_path(work, rid)
        else:
            continue
        if not dest.exists():
            remote.download(path, dest)
            got += 1
    log(f"{prefix_rel}: downloaded {got} file(s)")
    return got


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] in ("worker", "analyze"):
        _worker_main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        raise SystemExit("usage: python -m stopping.agop_final {worker|analyze} <spec.json> <index>")

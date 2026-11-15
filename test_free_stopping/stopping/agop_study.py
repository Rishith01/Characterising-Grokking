"""F2 experiment: is the remaining-angle bound a usable stopping signal on real runs?
(AGOP.md §4, THEORY.md §4; questions Q1-Q8 in the F2 discussion of 2026-11-14.)

Evaluation side: it reads test curves to score stops, so, like stopping.pilot, it is
exempt from the quarantine test. The signal itself (stopping.signals.agop) never sees
test data.

Pipeline:
1. pick_study_runs: a small, spread-out set of corpus runs per application.
2. run_study: replay each run once and record the agop_path trace (pairwise distances
   between unit-normalized G, M = G^{1/2} and per-layer weights at every eval step),
   plus its train/test curves. Resumable; uploads as it goes.
3. analyze_run / analyze_all: rebuild offline the angle over the last lag, the decay
   rate kappa of the speed, and the bound R = theta * q / (1 - q), q = exp(-kappa lag)
   (= omega / kappa for short lags), for a grid of settings (object, lag, window,
   noise-floor correction, kappa estimator, threshold, norm gate), and score every one.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from tfs.evaluation.report import rows_to_markdown

from .compute import compute
from .pilot import _acc_at, _curves
from .signals.agop import AgopPathConfig
from .store import TraceStore

# ---- settings grid ----------------------------------------------------------------------

OBJECTS = ("M", "G", "W")  # M = G^{1/2} (recommended), G, W = per-layer weight directions (max over layers)
LAG_FRACS = (0.15, 0.25, 0.33)  # lag = frac * t; two lags back must fit, so frac < 0.5
WINDOWS = ((1, False), (3, False), (3, True))  # (eval points per window mean, noise-floor correction)
EPS_R = (0.03, 0.1, 0.3)  # thresholds on R, in radians
ESTIMATORS = ("ratio", "fit")  # how kappa, the decay rate of the speed, is estimated
GATES = ("none", "norm")  # "norm": also require the remaining log-norm bound < EPS_NORM (THEORY.md §4.3-4.4)
EPS_NORM = 0.05
FIT_FACTOR = 2.0  # kappa is fitted over eval steps in [t / FIT_FACTOR, t]
MIN_FIT = 4  # minimum points in that fit
PATIENCE = 3  # consecutive eval rows below the threshold
DEFAULT = {"obj": "M", "lag": 0.25, "w": 3, "floor": True, "est": "ratio"}  # the setting the plots show
_DEFAULT_KEY = tuple(DEFAULT[k] for k in ("obj", "lag", "w", "floor", "est"))
CATEGORIES = ("grokker", "fast", "late_collapse")
LAMBDAS = (0.03, 0.1, 0.3)  # late-stop penalty per horizon of extra training (score_stop)
MAIN_LAMBDA = 0.1


# ---- geometry: angles from squared chord distances --------------------------------------

def angle(c2) -> np.ndarray:
    """Angle between unit vectors from their squared chord distance."""
    c = np.sqrt(np.clip(np.asarray(c2, dtype=np.float64), 0.0, 4.0))
    return 2.0 * np.arcsin(c / 2.0)


def full_d2(rows: Sequence) -> np.ndarray:
    """Symmetric k x k matrix from the causal rows (row i: distances to rows 0..i-1)."""
    k = len(rows)
    D = np.zeros((k, k))
    for i, r in enumerate(rows):
        r = np.asarray(r if r is not None else [], dtype=np.float64)
        if len(r) != i:
            raise ValueError(f"row {i} has {len(r)} distances, expected {i}")
        D[i, :i] = r
        D[:i, i] = r
    return D


def mean_chord2(D: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Squared chord between the normalized means of unit vectors A and B, from pairwise
    squared distances only. With s_A = 1 - |mean_A|^2 = mean(D_AA)/2:
    |mean_A - mean_B|^2 = mean(D_AB) - s_A - s_B, and the chord between the normalized
    means is (|mean_A - mean_B|^2 - (|mean_A| - |mean_B|)^2) / (|mean_A| |mean_B|)."""
    sA, sB = D[np.ix_(A, A)].mean() / 2, D[np.ix_(B, B)].mean() / 2
    nA, nB = math.sqrt(max(1 - sA, 1e-300)), math.sqrt(max(1 - sB, 1e-300))
    diff2 = D[np.ix_(A, B)].mean() - sA - sB
    dn = (sB - sA) / (nA + nB)
    return max(diff2 - dn * dn, 0.0) / (nA * nB)


# ---- the estimator ----------------------------------------------------------------------
#
# Banach's a posteriori bound on a sampled path (THEORY.md §4.1): if the angle travelled
# over each lag shrinks by a factor q < 1 per lag, the angle still to travel is at most
# theta_last * q / (1 - q). q is estimated from the last two lags (est "ratio", the
# textbook contraction-factor estimate), or from a regression of log speed on time over
# [t / FIT_FACTOR, t] (est "fit", smoother but biased when the lag grows with t). With
# q = exp(-kappa * lag) this is omega / kappa when kappa * lag is small. For lags
# proportional to t, kappa * lag is not small late in the run, and omega / kappa would
# overstate the remaining angle by a factor that grows with t.

def _ending_at(steps: np.ndarray, target: float, w: int, before: int) -> Optional[np.ndarray]:
    """The w eval indices ending at the last step <= target, if they all lie before index
    `before` (so consecutive windows do not overlap)."""
    j = int(np.searchsorted(steps, target, side="right")) - 1
    if j - w + 1 < 0 or j >= before:
        return None
    return np.arange(j - w + 1, j + 1)


def _lag_windows(steps: np.ndarray, i: int, lag: float, w: int, n: int):
    """n + 1 windows: the first ends at i, the m-th at the last step <= (1 - m lag) t."""
    if i - w + 1 < 0:
        return None
    wins = [np.arange(i - w + 1, i + 1)]
    for m in range(1, n + 1):
        nxt = _ending_at(steps, steps[i] * (1 - m * lag), w, wins[-1][0])
        if nxt is None:
            return None
        wins.append(nxt)
    return wins


def _gap(steps, A, B) -> float:
    return float(steps[A].mean() - steps[B].mean())


def path_lags(steps: np.ndarray, D: np.ndarray, lag: float, w: int, floor: bool):
    """For every eval index: the angle over the last lag (th1) and over the lag before it
    (th2) between window means, their durations dt1 and dt2, and the within-window
    scatter angle (noise-floor estimate; NaN for w = 1). th2 and dt2 are NaN while the
    run is too short for two lags."""
    k = len(steps)
    th1, dt1, th2, dt2, sc = (np.full(k, np.nan) for _ in range(5))

    def ang(A, B):
        th = float(angle(mean_chord2(D, A, B)))
        if w == 1:
            return th, np.nan
        s2 = (D[np.ix_(A, A)].mean() / 2 + D[np.ix_(B, B)].mean() / 2) / (w - 1)
        if floor:
            th = math.sqrt(max(th * th - s2, 0.0))
        return th, math.sqrt(s2)

    for i in range(k):
        wins = _lag_windows(steps, i, lag, w, 1)
        if wins is None:
            continue
        th1[i], sc[i] = ang(wins[0], wins[1])
        dt1[i] = _gap(steps, wins[0], wins[1])
        wins = _lag_windows(steps, i, lag, w, 2)
        if wins is not None:
            th2[i] = ang(wins[1], wins[2])[0]
            dt2[i] = _gap(steps, wins[1], wins[2])
    return th1, dt1, th2, dt2, sc


def scalar_lags(steps: np.ndarray, y: np.ndarray, lag: float, w: int, floor: bool):
    """path_lags for a scalar series: |difference of window means|."""
    k = len(steps)
    th1, dt1, th2, dt2 = (np.full(k, np.nan) for _ in range(4))

    def dif(A, B):
        d = abs(y[A].mean() - y[B].mean())
        if floor and w > 1:
            d = math.sqrt(max(d * d - (y[A].var() + y[B].var()) / (w - 1), 0.0))
        return d

    for i in range(k):
        wins = _lag_windows(steps, i, lag, w, 1)
        if wins is None:
            continue
        th1[i], dt1[i] = dif(wins[0], wins[1]), _gap(steps, wins[0], wins[1])
        wins = _lag_windows(steps, i, lag, w, 2)
        if wins is not None:
            th2[i], dt2[i] = dif(wins[1], wins[2]), _gap(steps, wins[1], wins[2])
    return th1, dt1, th2, dt2


def kappa_ratio(th1, dt1, th2, dt2) -> np.ndarray:
    """kappa from the last two lags: speeds w1 = th1/dt1 (recent) and w2 = th2/dt2,
    kappa = ln(w2 / w1) / mean lag. NaN where either speed is missing or zero."""
    out = np.full(len(th1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        w1, w2 = th1 / dt1, th2 / dt2
        ok = np.isfinite(w1) & np.isfinite(w2) & (w1 > 0) & (w2 > 0)
        out[ok] = np.log(w2[ok] / w1[ok]) / ((dt1[ok] + dt2[ok]) / 2)
    return out


def kappa_fit(steps: np.ndarray, th1: np.ndarray, dt1: np.ndarray, fit_factor: float = FIT_FACTOR,
              min_pts: int = MIN_FIT) -> np.ndarray:
    """kappa(t) = -slope of log(th1 / dt1) against the lag's midpoint time, over eval
    steps in [t / fit_factor, t] (causal)."""
    k = len(steps)
    out = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sp = th1 / dt1
    ok = np.isfinite(sp) & (sp > 0)
    logs = np.where(ok, np.log(np.where(ok, sp, 1.0)), np.nan)
    mid = steps - np.nan_to_num(dt1) / 2
    for i in range(k):
        m = ok[: i + 1] & (steps[: i + 1] >= steps[i] / fit_factor)
        if m.sum() < min_pts:
            continue
        x, y = mid[: i + 1][m], logs[: i + 1][m]
        x = x - x.mean()
        if (x * x).sum() > 0:
            out[i] = -float((x * (y - y.mean())).sum() / (x * x).sum())
    return out


def remaining_bound(th1: np.ndarray, dt1: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """R = th1 * q / (1 - q) with q = exp(-kappa dt1). inf when the speed is not falling
    (kappa <= 0, or unknown while th1 is known); 0 when no motion is detected (th1 == 0
    after the floor correction); NaN while th1 is undefined (start of the run)."""
    R = np.full(len(th1), np.nan)
    known = np.isfinite(th1)
    R[known] = np.inf
    falling = known & np.isfinite(kappa) & (kappa > 0)
    q = np.exp(-kappa[falling] * dt1[falling])
    with np.errstate(divide="ignore", invalid="ignore"):  # q -> 1 (tiny kappa) gives inf, as it should
        R[falling] = th1[falling] * q / (1 - q)
    R[known & (th1 == 0)] = 0.0
    return R


def bound(steps, th1, dt1, th2, dt2, est: str):
    """(kappa, R) with kappa from the estimator `est` ("ratio" or "fit")."""
    ka = kappa_ratio(th1, dt1, th2, dt2) if est == "ratio" else kappa_fit(steps, th1, dt1)
    return ka, remaining_bound(th1, dt1, ka)


def first_stop(steps: np.ndarray, cond: np.ndarray, patience: int = PATIENCE) -> Optional[int]:
    run = 0
    for i, c in enumerate(cond):
        run = run + 1 if c else 0
        if run >= patience:
            return int(steps[i])
    return None


# ---- settling tests (2026-11-15) --------------------------------------------------------
#
# The bound failed on real runs because it extrapolates the speed's recent trend, and the
# speed falls fast at the end of memorization although a slow drift follows (RESULTS.md,
# F2 experiment). These tests extrapolate nothing; they ask whether the path over a
# trailing window [t / factor, t] (log-time, so lr-free) is *confined* (settled: jitter
# around a point) or *directed* (still drifting), the standard question of
#  - mean-squared-displacement analysis of trajectories: MSD(lag) ~ lag^alpha with
#    alpha = 2 for directed motion, 1 for a free random walk, ~0 for confined motion
#    (Michalet 2010; Metzler et al. 2014);
#  - MCMC convergence diagnostics: Geweke 1992 compares the mean of an early and a late
#    segment of the chain, in units of their standard errors (Gelman & Rubin 1992 is the
#    multi-chain version). Here: the angle between the two segments' mean directions,
#    over the squared standard error of those means.
# Both are computed from the stored squared chord distances alone.

SETTLE_FACTORS = (2.0, 4.0)  # trailing window [t / factor, t]
MSD_THRESHOLDS = (0.5, 1.0)  # settled when the MSD exponent alpha <= threshold
GEWEKE_THRESHOLDS = (2.0, 5.0, 20.0)  # settled when Z <= threshold (iid jitter: Z ~ 1)
SETTLE_OBJECTS = ("M", "G", "W")
MIN_WINDOW = 6


def _trail(steps: np.ndarray, i: int, factor: float) -> np.ndarray:
    lo = int(np.searchsorted(steps, steps[i] / factor, side="left"))
    return np.arange(lo, i + 1)


def _msd_slope(lag: np.ndarray, val: np.ndarray, nbins: int) -> float:
    """Slope of log MSD against log lag over the longest lags of the window
    (lags in [L/8, L]: short lags are dominated by jitter)."""
    L = lag.max()
    edges = np.geomspace(L / 8, L, nbins + 1)
    mids, ms = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (lag >= e0) & (lag <= e1)
        if m.any():
            mids.append(lag[m].mean())
            ms.append(val[m].mean())
    if len(ms) < 3:
        return float("nan")
    ms = np.asarray(ms)
    if ms.max() <= 1e-14:
        return 0.0  # no motion at all
    x, y = np.log(mids), np.log(np.maximum(ms, 1e-300))
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x * x).sum())


def msd_exponent(steps: np.ndarray, D: np.ndarray, factor: float, nbins: int = 4,
                 min_pts: int = MIN_WINDOW) -> np.ndarray:
    """MSD exponent alpha of the path over each trailing window; D holds squared chord
    distances (the squared displacement between two points)."""
    out = np.full(len(steps), np.nan)
    for i in range(len(steps)):
        W = _trail(steps, i, factor)
        if len(W) < min_pts:
            continue
        a, b = np.triu_indices(len(W), 1)
        a, b = W[a], W[b]
        out[i] = _msd_slope((steps[b] - steps[a]).astype(float), D[a, b], nbins)
    return out


def geweke_z(steps: np.ndarray, D: np.ndarray, factor: float, min_half: int = 3) -> np.ndarray:
    """Geweke-style Z over each trailing window: split at t / sqrt(factor) into an early
    segment A and a late segment B; Z = chord^2(mean A, mean B) / (s_A^2/n_A + s_B^2/n_B),
    with s^2 the segment's total variance (mean squared distance to its mean, unbiased).
    Z ~ 1 for independent jitter around a point; large for a drift."""
    out = np.full(len(steps), np.nan)
    for i in range(len(steps)):
        W = _trail(steps, i, factor)
        split = steps[i] / math.sqrt(factor)
        A, B = W[steps[W] < split], W[steps[W] >= split]
        if len(A) < min_half or len(B) < min_half:
            continue
        c2 = mean_chord2(D, A, B)
        vA = D[np.ix_(A, A)].mean() / 2 * len(A) / (len(A) - 1)
        vB = D[np.ix_(B, B)].mean() / 2 * len(B) / (len(B) - 1)
        den = vA / len(A) + vB / len(B)
        out[i] = c2 / den if den > 0 else (0.0 if c2 == 0 else np.inf)
    return out


def _scalar_d2(y: np.ndarray) -> np.ndarray:
    """Squared differences, so the scalar log-norm goes through the same tests."""
    return (y[:, None] - y[None, :]) ** 2


def _scalar_geweke(steps: np.ndarray, y: np.ndarray, factor: float, min_half: int = 3) -> np.ndarray:
    out = np.full(len(steps), np.nan)
    for i in range(len(steps)):
        W = _trail(steps, i, factor)
        split = steps[i] / math.sqrt(factor)
        A, B = y[W[steps[W] < split]], y[W[steps[W] >= split]]
        if len(A) < min_half or len(B) < min_half:
            continue
        den = A.var(ddof=1) / len(A) + B.var(ddof=1) / len(B)
        d2 = (A.mean() - B.mean()) ** 2
        out[i] = d2 / den if den > 0 else (0.0 if d2 == 0 else np.inf)
    return out


def settle_stats(tr: PathTrace, test: str, obj: str, factor: float) -> np.ndarray:
    """The test statistic for one object (W: max over layers, all must be settled)."""
    mats = tr.D["W"] if obj == "W" else [tr.D[obj]]
    f = msd_exponent if test == "msd" else geweke_z
    return np.max(np.vstack([f(tr.steps, D, factor) for D in mats]), axis=0)


def settle_rows(tr: PathTrace, curves: dict, info: dict) -> list[dict]:
    """Score every settling-test setting: test x object x window factor x threshold x
    norm gate (the same test on log |w|, same threshold)."""
    rows = []
    logn = np.log(np.maximum(tr.w_norm, 1e-300))
    for test, thresholds in (("msd", MSD_THRESHOLDS), ("geweke", GEWEKE_THRESHOLDS)):
        for factor in SETTLE_FACTORS:
            if test == "msd":
                norm_stat = msd_exponent(tr.steps, _scalar_d2(logn), factor)
            else:
                norm_stat = _scalar_geweke(tr.steps, logn, factor)
            for obj in SETTLE_OBJECTS:
                stat = settle_stats(tr, test, obj, factor)
                for thr in thresholds:
                    for gate in GATES:
                        cond = np.nan_to_num(stat, nan=np.inf) <= thr
                        if gate == "norm":
                            cond &= np.nan_to_num(norm_stat, nan=np.inf) <= thr
                        stop = first_stop(tr.steps, cond)
                        rows.append({**_ids(info), "family": test,
                                     "setting": f"{test} {obj} factor={factor:g} thr={thr:g} gate={gate}",
                                     "obj": obj, "factor": factor, "thr": thr, "gate": gate,
                                     **score_stop(curves, stop, info)})
    return rows


# ---- scoring ----------------------------------------------------------------------------

def score_stop(curves: dict, stop: Optional[int], info: dict) -> dict:
    """Score one stop. A rule that never stops trains to the horizon T (the last logged
    step), so it is scored as a stop at T. Besides the regret max_test - acc(tau), every
    step trained after t_eps (the first step within EPS of the best test accuracy) costs
    lambda / T:

        cost_lambda = (max_test - acc(tau)) + lambda * max(0, tau - t_eps) / T

    so training the whole horizon after converging at once costs about lambda in every
    task, whatever its length. Dividing by T keeps the penalty comparable across tasks
    with different horizons and learning rates."""
    T = int(curves["step"][-1])
    tau = T if stop is None else int(stop)
    acc = _acc_at(curves, tau)
    max_test, t_eps = info.get("max_test"), info.get("t_eps")
    regret = (max_test - acc) if max_test is not None else np.nan
    late = max(0, tau - t_eps) / T if t_eps is not None else np.nan
    out = {"stop": stop, "test_at_stop": acc, "never": stop is None,
           "premature": stop is not None and t_eps is not None and stop < t_eps,
           "rel_lag": (stop - t_eps) / max(t_eps, 1) if stop is not None and t_eps is not None else np.nan,
           "regret": regret, "late": late}
    for lam in LAMBDAS:
        out[f"cost_{lam}"] = regret + lam * late
    return out


# ---- one run ----------------------------------------------------------------------------

@dataclass
class PathTrace:
    steps: np.ndarray
    D: dict  # "G", "M" -> k x k; "W" -> list of k x k (one per layer)
    w_norm: np.ndarray
    tr_G: np.ndarray
    fro_G: np.ndarray
    erank_G: np.ndarray


def from_trace(df: pd.DataFrame) -> PathTrace:
    df = df.sort_values("step").reset_index(drop=True)
    layers = sorted(int(c[4:]) for c in df.columns if c.startswith("d2_w"))
    D = {"G": full_d2(df["d2_G"]), "M": full_d2(df["d2_M"]),
         "W": [full_d2(df[f"d2_w{i}"]) for i in layers]}
    return PathTrace(df["step"].to_numpy(np.int64), D, df["w_norm"].to_numpy(float),
                     df["tr_G"].to_numpy(float), df["fro_G"].to_numpy(float), df["erank_G"].to_numpy(float))


def series(tr: PathTrace, obj: str, lag: float, w: int, floor: bool, est: str) -> dict:
    """omega (= th1 / dt1), kappa, R and the true remaining angle for one object and
    setting. For W every layer's R is computed and the maximum is taken (all layers must
    be settled)."""
    mats = tr.D["W"] if obj == "W" else [tr.D[obj]]
    out = {"omega": [], "kappa": [], "R": [], "theta": [], "scatter": [], "true": []}
    for D in mats:
        th1, dt1, th2, dt2, sc = path_lags(tr.steps, D, lag, w, floor)
        ka, R = bound(tr.steps, th1, dt1, th2, dt2, est)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["omega"].append(th1 / dt1)
        out["kappa"].append(ka)
        out["R"].append(R)
        out["theta"].append(th1)
        out["scatter"].append(sc)
        out["true"].append(angle(D[:, -1]))
    agg = {k: (np.max(np.vstack(v), axis=0) if obj == "W" else v[0]) for k, v in out.items()}
    n1, nd1, n2, nd2 = scalar_lags(tr.steps, np.log(np.maximum(tr.w_norm, 1e-300)), lag, w, floor)
    agg["R_norm"] = bound(tr.steps, n1, nd1, n2, nd2, est)[1]
    return agg


def d_chord(tr: PathTrace, anchor: int, lag: float) -> np.ndarray:
    """The proposal's chord test (its Thm. 4.1) on the trace-normalized AGOP
    G~ = G / tr G, anchored at eval index `anchor`: 1 - cos(G~_t - G~_0, G~_{t-lag} - G~_0)."""
    k = len(tr.steps)
    a = tr.fro_G / tr.tr_G  # |G~|
    U = np.outer(a, a) * (1 - tr.D["G"] / 2)  # <G~_i, G~_j>
    out = np.full(k, np.nan)
    for i in range(anchor + 1, k):
        j = int(np.searchsorted(tr.steps, tr.steps[i] * (1 - lag), side="right")) - 1
        if j <= anchor:
            continue
        m = anchor
        xy = U[i, j] - U[i, m] - U[j, m] + U[m, m]
        xx = U[i, i] - 2 * U[i, m] + U[m, m]
        yy = U[j, j] - 2 * U[j, m] + U[m, m]
        if xx > 0 and yy > 0:
            out[i] = 1 - xy / math.sqrt(xx * yy)
    return out


def _settings():
    for obj in OBJECTS:
        for lag in LAG_FRACS:
            for w, floor in WINDOWS:
                for est in ESTIMATORS:
                    yield obj, lag, w, floor, est


def analyze_run(df: pd.DataFrame, curves: dict, info: dict) -> tuple[list[dict], dict, dict]:
    """(setting rows, question row, default-setting series for plotting)."""
    tr = from_trace(df)
    steps = tr.steps
    t_eps, t_mem = info.get("t_eps"), info.get("train99_step")
    max_test = info.get("max_test")
    rows, default = [], None
    for obj, lag, w, floor, est in _settings():
        s = series(tr, obj, lag, w, floor, est)
        if (obj, lag, w, floor, est) == _DEFAULT_KEY:
            default = s
        for eps in EPS_R:
            for gate in GATES:
                cond = np.nan_to_num(s["R"], nan=np.inf) <= eps
                if gate == "norm":
                    cond &= np.nan_to_num(s["R_norm"], nan=np.inf) <= EPS_NORM
                stop = first_stop(steps, cond)
                i = None if stop is None else int(np.searchsorted(steps, stop))
                rows.append({
                    **_ids(info), "family": "bound",
                    "setting": f"bound {obj} lag={lag} w={w} floor={int(floor)} est={est} eps={eps} gate={gate}",
                    "obj": obj, "lag": lag, "w": w, "floor": floor, "est": est, "eps": eps, "gate": gate,
                    **score_stop(curves, stop, info),
                    "R_at_stop": float(s["R"][i]) if i is not None else np.nan,
                    "true_angle_at_stop": float(s["true"][i]) if i is not None else np.nan,
                    "bound_ok": bool(s["R"][i] >= s["true"][i]) if i is not None else np.nan,
                })
    rows += settle_rows(tr, curves, info)
    q = question_row(tr, default, info)
    return rows, q, {"steps": steps, "w_norm": tr.w_norm, "erank_G": tr.erank_G, **default,
                     "series_G": series(tr, "G", *_DEFAULT_KEY[1:]),
                     "series_W": series(tr, "W", *_DEFAULT_KEY[1:]),
                     "d_chord": _dchord_for(tr, info)}


def _ids(info: dict) -> dict:
    return {"run_id": info["run_id"], "app": info.get("app"), "category": info.get("category"),
            "optim": info.get("optim")}


def _dchord_for(tr: PathTrace, info: dict) -> np.ndarray:
    t_mem = info.get("train99_step")
    if t_mem is None:
        return np.full(len(tr.steps), np.nan)
    anchor = int(np.searchsorted(tr.steps, t_mem))
    if anchor >= len(tr.steps) - 1:
        return np.full(len(tr.steps), np.nan)
    return d_chord(tr, anchor, DEFAULT["lag"])


def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _cv(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(x.std() / x.mean()) if x.size >= 3 else float("nan")


def question_row(tr: PathTrace, s: dict, info: dict) -> dict:
    """Per-run answers to Q1-Q4 and Q8 at the default setting (DEFAULT)."""
    steps, D = tr.steps, tr.D["M"]
    k = len(steps)
    t_eps, t_mem = info.get("t_eps"), info.get("train99_step")
    step_angle = np.r_[np.nan, angle(D[np.arange(1, k), np.arange(k - 1)])]
    total = np.nansum(step_angle)
    pre = steps < t_eps if t_eps is not None else np.zeros(k, bool)
    plateau = pre & (steps >= t_mem) if t_mem is not None else np.zeros(k, bool)
    kt = s["kappa"] * steps
    tail = np.arange(k) >= int(0.7 * k)
    last = np.arange(k) >= int(0.8 * k)
    R = np.nan_to_num(s["R"], nan=np.inf)
    dch = _dchord_for(tr, info)
    after = np.isfinite(dch)
    return {
        "run_id": info["run_id"], "app": info.get("app"), "category": info.get("category"),
        "optim": info.get("optim"), "gamma": info.get("gamma"), "alpha": info.get("alpha"),
        "t_mem": t_mem, "t_eps": t_eps, "max_test": info.get("max_test"), "eval_rows": k,
        # Q1: do the features move on the plateau?
        "plateau_rows": int(plateau.sum()),
        "path_total_rad": float(total),
        "path_plateau_frac": float(np.nansum(step_angle[plateau]) / total) if total > 0 and plateau.any() else np.nan,
        "kt_plateau_median": _median(kt[plateau]),
        # Q3: does R dip before generalization?
        "minR_plateau": float(R[plateau].min()) if plateau.any() else np.nan,
        "minR_pre_teps": float(R[pre & np.isfinite(s["omega"])].min()) if (pre & np.isfinite(s["omega"])).any() else np.nan,
        # Q2: exponential tail? (constant kappa) or power law (constant kappa * t)?
        "tail_cv_kappa": _cv(s["kappa"][tail]), "tail_cv_kt": _cv(kt[tail]),
        "tail_kt_median": _median(kt[tail]),
        # Q4: noise floor, as within-window scatter over the lagged angle, late in the run
        "floor_ratio_late": _median((s["scatter"] / s["theta"])[last]),
        "final_R": float(R[-1]),
        # Q8: is the AGOP path a straight chord after memorization?
        "dchord_median": _median(dch[after]),
        "dchord_frac_below_0.05": float((dch[after] < 0.05).mean()) if after.any() else np.nan,
    }


# ---- many runs --------------------------------------------------------------------------

def baseline_rows(curves: dict, info: dict) -> list[dict]:
    """Reference stops scored the same way: train to the horizon, and stop exactly at
    t_eps (an oracle, and the best any rule can do on this score)."""
    return [{**_ids(info), "family": "baseline", "setting": "horizon (never stop)", **score_stop(curves, None, info)},
            {**_ids(info), "family": "baseline", "setting": "oracle stop at t_eps",
             **score_stop(curves, info.get("t_eps"), info)}]


def summarize_settings(settings: pd.DataFrame) -> pd.DataFrame:
    """One row per (app or "all", setting): premature and never rates, regret, the
    late-penalized costs and lag statistics, sorted by the main cost."""
    df = pd.concat([settings, settings.assign(app="all")], ignore_index=True)
    if "bound_ok" not in df:
        df["bound_ok"] = np.nan
    g = df.groupby(["app", "family", "setting"], dropna=False)
    agg = {"runs": ("run_id", "count"), "premature": ("premature", "mean"), "never": ("never", "mean"),
           "regret_mean": ("regret", "mean"), "regret_median": ("regret", "median"),
           "late_mean": ("late", "mean"), "rel_lag_median": ("rel_lag", "median"),
           "bound_violation": ("bound_ok", lambda x: float((x == False).sum()) / max(int(x.notna().sum()), 1))}  # noqa: E712
    for lam in LAMBDAS:
        agg[f"cost_{lam}"] = (f"cost_{lam}", "mean")
    out = g.agg(**agg).reset_index()
    for c in out.columns[4:]:
        out[c] = out[c].astype(float).round(4)
    return out.sort_values(["app", f"cost_{MAIN_LAMBDA}"]).reset_index(drop=True)


def headline(q: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Short automatic answers to Q1-Q8, for summary.md."""
    lines = []
    gk = q[q["category"] == "grokker"]
    lines.append(f"- **Q1 (features move on the plateau):** {len(gk)} grokkers; median share of the total M path "
                 f"travelled on the plateau = {_median(gk['path_plateau_frac']):.2f}; median kappa*t on the plateau "
                 f"= {_median(gk['kt_plateau_median']):.2f} (near 0 means a steady drift, so R = inf: safe).")
    for eps in EPS_R:
        lines.append(f"- **Q3 (dip before t_eps), eps = {eps}:** R fell below eps before t_eps in "
                     f"{int((q['minR_pre_teps'] <= eps).sum())}/{q['minR_pre_teps'].notna().sum()} runs "
                     f"(grokkers on the plateau: {int((gk['minR_plateau'] <= eps).sum())}/{gk['minR_plateau'].notna().sum()}).")
    lines.append(f"- **Q2 (tail shape):** median coefficient of variation in the last 30% of the run: kappa "
                 f"{_median(q['tail_cv_kappa']):.2f} vs kappa*t {_median(q['tail_cv_kt']):.2f} (the smaller one is the "
                 f"better description: constant kappa = exponential, constant kappa*t = power law).")
    for app, g in q.groupby("app"):
        lines.append(f"- **Q4 (noise floor), {app}:** median late scatter / lagged angle = "
                     f"{_median(g['floor_ratio_late']):.2f} (above ~0.5 the floor dominates the measured speed).")
    allr = summary[summary["app"] == "all"]
    best = pd.concat([allr[allr["family"] == "baseline"]]
                     + [allr[allr["family"] == f].head(3) for f in allr["family"].unique() if f != "baseline"])
    lines.append(f"- **Q5-Q7:** the table below has the baselines and the best three settings of each family by "
                 f"mean cost at lambda = {MAIN_LAMBDA} (regret + lambda x training after t_eps / horizon); "
                 f"settings_summary.csv has every setting.")
    lines.append(f"- **Q8 (chord model):** median share of post-memorization rows with D_chord < 0.05 = "
                 f"{_median(q['dchord_frac_below_0.05']):.2f}.")
    cols = ["family", "setting", "runs", "premature", "never", "regret_mean", "late_mean"] + [f"cost_{l}" for l in LAMBDAS]
    return "\n".join(lines) + "\n\n" + rows_to_markdown(best[cols].to_dict("records"))


def plot_run(curves: dict, s: dict, info: dict, settings: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.maximum(s["steps"].astype(float), 1)
    fig, ax = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
    cs = np.maximum(np.asarray(curves["step"], float), 1)
    ax[0].plot(cs, curves["train_acc"], label="train acc")
    ax[0].plot(cs, curves["test_acc"], label="test acc")
    for t, lab, c in ((info.get("train99_step"), "t_mem", "grey"), (info.get("t_eps"), "t_eps", "black")):
        if t:
            for a in ax:
                a.axvline(max(t, 1), color=c, lw=0.8, ls=":")
    settings = settings[settings["family"] == "bound"]
    me = settings[(settings["obj"] == DEFAULT["obj"]) & (settings["lag"] == DEFAULT["lag"])
                  & (settings["w"] == DEFAULT["w"]) & (settings["floor"] == DEFAULT["floor"])
                  & (settings["est"] == DEFAULT["est"]) & (settings["eps"] == 0.1)]
    for _, r in me.iterrows():
        if r["stop"] is not None and not pd.isna(r["stop"]):
            ax[0].axvline(max(r["stop"], 1), color="red" if r["gate"] == "none" else "purple", lw=1.2, ls="--",
                          label=f"stop eps 0.1, gate {r['gate']}")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend(fontsize=6, loc="lower right")
    ax[0].set_ylabel("accuracy")
    ax[1].plot(steps, s["w_norm"])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("|w|")
    for lab, ser in (("M", s), ("G", s["series_G"]), ("W (max layer)", s["series_W"])):
        om = np.where(ser["omega"] > 0, ser["omega"], np.nan)
        ax[2].plot(steps, om, label=lab, lw=1)
        R = np.where(np.isfinite(ser["R"]) & (ser["R"] > 0), ser["R"], np.nan)
        ax[4].plot(steps, R, label=f"R {lab}", lw=1)
    ax[2].set_yscale("log")
    ax[2].set_ylabel("omega (rad/step)")
    ax[2].legend(fontsize=6)
    ax[3].plot(steps, s["kappa"] * steps, lw=1)
    ax[3].axhline(0, color="grey", lw=0.6)
    ax[3].set_ylabel("kappa * t (M)")
    ax[4].plot(steps, np.where(s["true"] > 0, s["true"], np.nan), "k--", lw=1, label="true remaining angle (M)")
    for e in EPS_R:
        ax[4].axhline(e, color="grey", lw=0.5, ls=":")
    ax[4].set_yscale("log")
    ax[4].set_ylabel("R (rad)")
    ax[4].legend(fontsize=6)
    ax[5].plot(steps, s["d_chord"], lw=1, label="D_chord (G~)")
    ax[5].axhline(0.05, color="grey", lw=0.5, ls=":")
    ax5 = ax[5].twinx()
    ax5.plot(steps, s["erank_G"], color="tab:green", lw=0.8)
    ax5.set_ylabel("erank G", fontsize=7, color="tab:green")
    ax[5].set_ylabel("D_chord")
    ax[-1].set_xscale("log")
    ax[-1].set_xlabel("step")
    fig.suptitle(f"{info.get('app')} {info.get('variant')} {info.get('optim')} a={info.get('alpha')} "
                 f"g={info.get('gamma')} [{info.get('category')}]\n{info['run_id']}", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


# ---- run selection ----------------------------------------------------------------------

def _cell_key(r: dict) -> tuple:
    return (r["variant"], r["optim"], r["lr"], r["gamma"], r["alpha"], r.get("size"))


def pick_study_runs(rows: list[dict], app: str, categories: Sequence[str] = CATEGORIES, per_cell: int = 1,
                    variants: Optional[Sequence[str]] = None, gamma0: bool = True,
                    seed_grokkers: int = 2, extra_seeds: Sequence[int] = (1, 2)) -> list[dict]:
    """Seed-0 runs: per_cell per (variant, optimizer, category), the middle one of the
    sorted candidates; plus one gamma = 0 run per (variant, optimizer) if none was
    picked (Adam+L2 skipped: at gamma = 0 it is the same run as AdamW); plus extra seeds
    of the first `seed_grokkers` grokkers, spread over optimizers. Adds "app" and "role"."""
    rows = [r for r in rows if r.get("status", "complete") == "complete"]
    if variants:
        rows = [r for r in rows if r["variant"] in set(variants)]
    seed0 = [r for r in rows if r["seed"] == 0]
    out: list[dict] = []
    for (v, o) in sorted({(r["variant"], r["optim"]) for r in seed0}):
        cell = [r for r in seed0 if r["variant"] == v and r["optim"] == o]
        for cat in categories:
            cand = sorted((r for r in cell if r["category"] == cat), key=lambda r: (r["alpha"], r["gamma"],
                                                                                   r.get("size") or 0, r["run_id"]))
            for k in range(min(per_cell, len(cand))):
                pick = cand[(len(cand) * (2 * k + 1)) // (2 * per_cell)]
                out.append({**pick, "app": app, "role": "base"})
        if gamma0 and o != "adam_l2" and not any(r["gamma"] == 0 for r in out if r["variant"] == v and r["optim"] == o):
            cand = sorted((r for r in cell if r["gamma"] == 0 and r["category"] != "diverged"),
                          key=lambda r: (r["alpha"], r.get("size") or 0, r["run_id"]))
            if cand:
                out.append({**cand[len(cand) // 2], "app": app, "role": "gamma0"})
    grok, seen_opt = [], set()
    for r in out:
        if r["category"] == "grokker" and r["optim"] not in seen_opt:
            grok.append(r)
            seen_opt.add(r["optim"])
    for r in grok[:seed_grokkers]:
        for s in extra_seeds:
            twin = [x for x in rows if x["seed"] == s and _cell_key(x) == _cell_key(r)]
            if twin:
                out.append({**twin[0], "app": app, "role": f"seed{s}"})
    seen, uniq = set(), []
    for r in out:
        if r["run_id"] not in seen:
            seen.add(r["run_id"])
            uniq.append(r)
    return uniq


# ---- driver -----------------------------------------------------------------------------

def _curves_path(work: Path, run_id: str) -> Path:
    return Path(work) / "curves" / f"{run_id}.json"


def run_study(sources: dict, selection: list[dict], store: TraceStore, cfg: AgopPathConfig, work: Path,
              out_remote=None, eval_kw: Optional[dict] = None, upload_every: int = 5,
              log: Callable[[str], None] = print) -> list[dict]:
    """Compute the agop_path trace and the curves of every selected run (skipping what is
    cached), uploading the per-run parts every `upload_every` runs. sources: app ->
    RunSource. Returns one status row per run."""
    work = Path(work)
    rel = f"signals/{cfg.name}/{cfg.hash()}"
    pending_files: dict[str, Path] = {}
    status = []

    def flush(msg: str):
        if out_remote is not None and pending_files:
            out_remote.commit(dict(pending_files), message=msg)
            log(f"uploaded {len(pending_files)} file(s)")
            pending_files.clear()

    for n, r in enumerate(selection, 1):
        rid, src = r["run_id"], sources[r["app"]]
        t0 = time.time()
        st = compute(src, [rid], [cfg], store, eval_kw=eval_kw, log=log)[0]
        cpath = _curves_path(work, rid)
        if st["ok"] and not cpath.exists():
            try:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(json.dumps(_curves(src, rid)))
            except Exception as e:  # curves are needed only for scoring; retried next time
                st = {**st, "ok": False, "error": f"curves: {type(e).__name__}: {e}"}
        if st["ok"] and st["signals"]:  # newly computed
            pending_files[f"{rel}/parts/{rid}.parquet"] = store.dir(cfg) / "parts" / f"{rid}.parquet"
            pending_files[f"curves/{rid}.json"] = cpath
            pending_files[f"{rel}/config.json"] = store.dir(cfg) / "config.json"
        status.append({**{k: r.get(k) for k in ("run_id", "app", "category", "optim", "role")},
                       "ok": st["ok"], "seconds": round(time.time() - t0, 1), "error": st["error"]})
        log(f"[{n}/{len(selection)}] {r['app']} {r['optim']} {r['category']} ({r['role']}): "
            f"{'ok' if st['ok'] else 'FAILED'} in {status[-1]['seconds']} s")
        if n % upload_every == 0:
            flush(f"agop study: traces after {n} runs")
    flush("agop study: traces")
    return status


def resume_from_remote(out_remote, store: TraceStore, cfg: AgopPathConfig, work: Path,
                       log: Callable[[str], None] = print) -> int:
    """Download trace parts and curves already uploaded by an earlier session."""
    rel = f"signals/{cfg.name}/{cfg.hash()}"
    got = 0
    files = out_remote.list_files(rel + "/parts")
    files.update(out_remote.list_files("curves"))
    for path in files:
        if path.startswith(rel + "/parts/"):
            dest = store.dir(cfg) / "parts" / Path(path).name
        elif path.startswith("curves/"):
            dest = Path(work) / "curves" / Path(path).name
        else:
            continue
        if not dest.exists():
            out_remote.download(path, dest)
            got += 1
    cfg_path = store.dir(cfg) / "config.json"
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True))
    log(f"resumed {got} file(s) from the remote")
    return got


def analyze_all(selection: list[dict], store: TraceStore, cfg: AgopPathConfig, work: Path,
                out_dir: Path, plots: bool = True, log: Callable[[str], None] = print) -> dict:
    """Analyze every selected run that has a trace and curves; write settings_runs.csv,
    settings_summary.csv, questions.csv, selection.csv, plots/ and summary.md."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, qrows, plot_paths = [], [], []
    for r in selection:
        rid = r["run_id"]
        cpath = _curves_path(work, rid)
        if not (store.has(cfg, rid) and cpath.exists()):
            continue
        try:
            curves = json.loads(cpath.read_text())
            rows, q, s = analyze_run(store.read(cfg, rid), curves, r)
            all_rows += rows + baseline_rows(curves, r)
            qrows.append({**q, "role": r.get("role")})
            if plots:
                plot_paths.append(plot_run(curves, s, r, pd.DataFrame(rows), out_dir / "plots" / r["app"] / f"{rid}.png"))
        except Exception as e:
            log(f"analysis failed for {rid}: {type(e).__name__}: {e}")
    settings = pd.DataFrame(all_rows)
    q = pd.DataFrame(qrows)
    files = {}
    if settings.empty:
        return {"settings": settings, "summary": pd.DataFrame(), "questions": q, "plots": [], "files": files,
                "markdown": "(no analyzed runs)"}
    summary = summarize_settings(settings)
    sel = pd.DataFrame([{k: r.get(k) for k in ("run_id", "app", "variant", "optim", "lr", "gamma", "alpha", "size",
                                               "seed", "category", "role", "max_test", "t_eps", "train99_step")}
                        for r in selection])
    for name, df in (("settings_runs", settings), ("settings_summary", summary), ("questions", q), ("selection", sel)):
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        files[f"{name}.csv"] = p
    qcols = ["app", "category", "optim", "gamma", "role", "path_plateau_frac", "kt_plateau_median", "minR_plateau",
             "minR_pre_teps", "tail_cv_kappa", "tail_cv_kt", "floor_ratio_late", "dchord_frac_below_0.05"]
    md = ("# F2 experiment: remaining-angle bound on corpus runs\n\n"
          f"Analyzed runs: {len(q)} of {len(selection)} selected. Settings grid: objects {OBJECTS}, lag fractions "
          f"{LAG_FRACS}, windows {WINDOWS}, kappa estimators {ESTIMATORS}, thresholds {EPS_R}, gates {GATES} (norm gate: remaining log-norm < "
          f"{EPS_NORM}); kappa fitted over [t/{FIT_FACTOR}, t]; patience {PATIENCE} eval rows. Default (plots): {DEFAULT}.\n\n"
          "## Headline answers\n\n" + headline(q, summary)
          + "\n\n## Per run (default setting)\n\n"
          + rows_to_markdown(q[qcols].round(3).to_dict("records")))
    (out_dir / "summary.md").write_text(md)
    files["summary.md"] = out_dir / "summary.md"
    for p in plot_paths:
        files[p.relative_to(out_dir).as_posix()] = p
    return {"settings": settings, "summary": summary, "questions": q, "plots": plot_paths, "files": files,
            "markdown": md}

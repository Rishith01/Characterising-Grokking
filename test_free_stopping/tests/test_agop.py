"""F2 study: the agop_path signal and the offline remaining-angle analysis."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from stopping import agop_study as st
from stopping.compute import compute_run
from stopping.signals import AgopPathConfig
from stopping.signals._linalg import input_agop
from stopping.store import TraceStore
from stopping.tfs_source import DirRunSource
from tfs.storage.hub import LocalRemote


def _unit_rows(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X, [[float(((X[i] - X[j]) ** 2).sum()) for j in range(i)] for i in range(len(X))]


# ---- math -------------------------------------------------------------------------------

def test_agop_of_a_linear_model_is_AtA():
    torch.manual_seed(0)
    lin = torch.nn.Linear(6, 3)
    G = input_agop(lin, torch.randn(50, 6), chunk=7)
    A = lin.weight.detach().double()
    assert torch.allclose(G, A.T @ A, atol=1e-6)


def test_mean_chord_matches_direct_computation():
    rng = np.random.default_rng(0)
    X, rows = _unit_rows(rng.normal(size=(9, 20)) + 3)
    D = st.full_d2(rows)
    A, B = np.array([0, 1, 2]), np.array([5, 6, 7, 8])
    a, b = X[A].mean(0), X[B].mean(0)
    direct = ((a / np.linalg.norm(a) - b / np.linalg.norm(b)) ** 2).sum()
    assert st.mean_chord2(D, A, B) == pytest.approx(direct, rel=1e-9)
    assert st.angle(D[0, 1]) == pytest.approx(math.acos(X[0] @ X[1]), rel=1e-9)


def _circle(steps, phi):
    _, rows = _unit_rows(np.c_[np.cos(phi), np.sin(phi)])
    return st.full_d2(rows)


@pytest.mark.parametrize("lag", st.LAG_FRACS)
def test_ratio_bound_recovers_the_remaining_angle_on_an_exponential_path(lag):
    # a point on a great circle at angle phi(t) = Phi (1 - exp(-k t)): remaining = Phi - phi
    k, Phi = 2e-3, 1.0
    steps = np.unique(np.r_[np.geomspace(1, 5000, 150).astype(int), np.linspace(0, 5000, 150).astype(int)])
    phi = Phi * (1 - np.exp(-k * steps))
    th1, dt1, th2, dt2, _ = st.path_lags(steps, _circle(steps, phi), lag, 1, False)
    _, R = st.bound(steps, th1, dt1, th2, dt2, "ratio")
    true = Phi - phi
    ok = np.isfinite(R) & (steps > 500) & (true > 1e-4)
    assert ok.sum() > 50
    # exact for an exponential path, up to the grid snapping of the lags
    assert np.median(R[ok] / true[ok]) == pytest.approx(1.0, abs=0.05)
    assert np.all(np.abs(R[ok] / true[ok] - 1) < 0.2)


def test_omega_over_kappa_overstates_with_lags_proportional_to_t():
    # why the bound is written as theta q / (1 - q) and not omega / kappa (THEORY.md §4.1)
    k = 2e-3
    steps = np.arange(0, 5001, 10)
    phi = 1 - np.exp(-k * steps)
    th1, dt1, th2, dt2, _ = st.path_lags(steps, _circle(steps, phi), 0.25, 1, False)
    ka = st.kappa_ratio(th1, dt1, th2, dt2)
    i = int(np.searchsorted(steps, 3000))
    naive = (th1[i] / dt1[i]) / ka[i]
    assert ka[i] == pytest.approx(k, rel=0.02)
    assert naive / (1 - phi[i]) > 1.5  # kappa * lag = 1.5 here


def test_bound_is_infinite_on_a_steady_drift():
    steps = np.arange(0, 5000, 25)
    th1, dt1, th2, dt2, _ = st.path_lags(steps, _circle(steps, 1e-4 * steps), 0.25, 1, False)
    for est in st.ESTIMATORS:
        _, R = st.bound(steps, th1, dt1, th2, dt2, est)
        fin = R[np.isfinite(th1) & (steps > 1000)]
        assert np.all(fin > 10), est  # constant speed: kappa ~ 0, so R is huge or infinite


def test_floor_correction_removes_pure_jitter():
    rng = np.random.default_rng(1)
    steps = np.arange(0, 3000, 10)
    X = np.c_[np.ones(len(steps)), 1e-3 * rng.normal(size=(len(steps), 5))]  # jitter around a fixed point
    _, rows = _unit_rows(X)
    D = st.full_d2(rows)
    raw = st.path_lags(steps, D, 0.25, 3, False)[0]
    cor = st.path_lags(steps, D, 0.25, 3, True)[0]
    late = steps > 1000
    assert np.nanmedian(cor[late]) < 0.5 * np.nanmedian(raw[late])


def test_first_stop_needs_patience():
    steps = np.arange(6) * 10
    assert st.first_stop(steps, np.array([1, 0, 1, 1, 1, 0], bool), 3) == 40
    assert st.first_stop(steps, np.array([1, 0, 1, 1, 0, 0], bool), 3) is None


# ---- the signal on a real (dummy) run ----------------------------------------------------

@pytest.fixture(scope="module")
def agop_trace(dummy_run):
    src = DirRunSource(dummy_run.parents[1])
    with src.open(dummy_run.name, set()) as h:
        return compute_run(h, [AgopPathConfig()], eval_kw={"n_log": 10, "n_uniform": 10})[AgopPathConfig().key]


def test_signal_rows_are_causal_and_consistent(agop_trace, dummy_run):
    df = agop_trace
    k = len(df)
    assert k >= 8
    for i in range(k):
        assert len(df["d2_M"].iloc[i]) == i and len(df["d2_G"].iloc[i]) == i and len(df["d2_w0"].iloc[i]) == i
    D = st.full_d2(df["d2_M"])
    assert np.all(D >= 0) and np.all(D <= 4)
    from tfs.storage.reader import RunReader
    tel = RunReader(dummy_run).telemetry().set_index("step")
    w = np.sqrt(sum(tel[f"l{i}_w_norm"] ** 2 for i in range(3)))
    assert np.allclose(df["w_norm"], w.loc[df["step"]].to_numpy(), rtol=1e-5)
    src = DirRunSource(dummy_run.parents[1])
    with src.open(dummy_run.name, set()) as h:
        cut = compute_run(h, [AgopPathConfig()], max_step=120, eval_steps=df["step"].tolist())[AgopPathConfig().key]
    pd.testing.assert_frame_equal(df[df["step"] <= 120].reset_index(drop=True), cut)


def test_trace_survives_the_store_and_analysis_runs(agop_trace, dummy_run, tmp_path):
    store = TraceStore(tmp_path / "traces")
    cfg = AgopPathConfig()
    store.write(cfg, dummy_run.name, agop_trace)
    back = store.read(cfg, dummy_run.name)
    assert list(back["d2_M"].iloc[3]) == pytest.approx(list(agop_trace["d2_M"].iloc[3]))
    curves = {"step": back["step"].tolist(), "train_acc": [1.0] * len(back), "test_acc": [0.5] * len(back)}
    info = {"run_id": dummy_run.name, "app": "dummy", "category": "fast", "optim": "adamw",
            "t_eps": int(back["step"].iloc[2]), "train99_step": int(back["step"].iloc[1]), "max_test": 0.5}
    rows, q, s = st.analyze_run(back, curves, info)
    n_settings = len(st.OBJECTS) * len(st.LAG_FRACS) * len(st.WINDOWS) * len(st.ESTIMATORS) * len(st.EPS_R) * len(st.GATES)
    assert len(rows) == n_settings
    assert q["eval_rows"] == len(back)
    summary = st.summarize_settings(pd.DataFrame(rows))
    assert set(summary["app"]) == {"dummy", "all"}


def test_run_study_end_to_end_with_resume(dummy_run, tmp_path):
    src = DirRunSource(dummy_run.parents[1])
    cfg = AgopPathConfig()
    sel = [{"run_id": dummy_run.name, "app": "dummy", "category": "fast", "optim": "adamw", "role": "base",
            "variant": "dummy", "alpha": 1.0, "gamma": 0.1, "t_eps": 50, "train99_step": 20, "max_test": 0.5}]
    remote = LocalRemote(tmp_path / "remote")
    store = TraceStore(tmp_path / "w1" / "traces")
    status = st.run_study({"dummy": src}, sel, store, cfg, tmp_path / "w1", out_remote=remote,
                          eval_kw={"n_log": 10, "n_uniform": 10}, log=lambda m: None)
    assert status[0]["ok"], status
    res = st.analyze_all(sel, store, cfg, tmp_path / "w1", tmp_path / "out", log=print)
    assert len(res["questions"]) == 1 and res["plots"][0].exists()
    assert "Headline answers" in res["markdown"]
    # a fresh session picks the uploaded parts back up
    store2 = TraceStore(tmp_path / "w2" / "traces")
    assert st.resume_from_remote(remote, store2, cfg, tmp_path / "w2", log=lambda m: None) == 2
    assert store2.has(cfg, dummy_run.name)
    assert json.loads((tmp_path / "w2" / "curves" / f"{dummy_run.name}.json").read_text())["step"]


def test_pick_study_runs_spreads_over_cells_and_adds_seeds():
    rows = []
    for o in ("adamw", "adam_l2", "sgd"):
        for g in (0.0, 0.1):
            for seed in (0, 1, 2):
                for cat, a in (("grokker", 8), ("fast", 1)):
                    rows.append({"run_id": f"{o}_{g}_{a}_{seed}", "variant": "mnist d3", "optim": o, "lr": 1e-3,
                                 "gamma": g, "alpha": a, "size": 1000, "seed": seed,
                                 "category": cat if g > 0 else "never", "status": "complete"})
    sel = st.pick_study_runs(rows, "mnist", seed_grokkers=2)
    roles = pd.Series([r["role"] for r in sel]).value_counts().to_dict()
    assert roles["base"] == 6  # 3 optimizers x (grokker, fast)
    assert roles["gamma0"] == 2  # adamw and sgd; adam_l2 at gamma 0 duplicates adamw
    assert roles["seed1"] == 2 and roles["seed2"] == 2
    assert len({r["run_id"] for r in sel}) == len(sel)

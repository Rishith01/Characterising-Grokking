"""Final F2 study: selection, constant-delta angle, calibration statistics, the parallel
replay/analysis workers and the report, end to end on the dummy run."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stopping import agop_final as af
from stopping import agop_study as st
from stopping.signals import AgopPathConfig
from stopping.store import TraceStore
from tfs.storage.hub import LocalRemote

PROJECT = Path(__file__).resolve().parents[1]


def _circle(phi):
    X = np.c_[np.cos(phi), np.sin(phi)]
    rows = [[float(((X[i] - X[j]) ** 2).sum()) for j in range(i)] for i in range(len(X))]
    return st.full_d2(rows)


def test_const_lags_measures_the_angle_over_a_fixed_lag():
    steps = np.arange(0, 2001, 10)
    phi = 1.0 - np.exp(-steps / 500)
    th = af.const_lags(steps, _circle(phi), 200, 1, False)
    i = np.arange(len(steps))
    ok = steps >= 200
    assert np.isnan(th[~ok]).all()
    assert th[ok] == pytest.approx(phi[ok] - phi[i[ok] - 20], abs=1e-7)


def _rows():
    out = []
    cats = ["grokker", "fast", "never", "grokker", "fast", "late_collapse"]
    for v in ("modular x+y", "modular x*y"):
        for o in ("adamw", "sgd"):
            for a, cat in enumerate(cats):
                for s in range(4):
                    if v == "modular x*y" and a == 5 and s == 3:
                        continue  # an incomplete cell is never picked
                    out.append({"run_id": f"{v}{o}{a}s{s}", "variant": v, "optim": o, "lr": 1e-3, "gamma": 0.1,
                                "alpha": float(a), "size": None, "seed": s, "status": "complete",
                                "category": cat if s == 0 else "fast"})
    return out


def test_pick_final_runs_takes_whole_cells_round_robin_over_categories():
    sel = af.pick_final_runs(_rows(), "modular", {"modular x+y": 3, "modular x*y": 6})
    cells = pd.DataFrame(sel).groupby("cell")
    assert (cells.size() == 4).all()
    assert (cells["seed"].apply(sorted).apply(tuple) == (0, 1, 2, 3)).all()
    xy = pd.DataFrame(sel)
    first = xy[(xy.variant == "modular x+y") & (xy.optim == "adamw") & (xy.seed == 0)]
    assert sorted(first["cell_category"]) == ["fast", "grokker", "never"]
    times = xy[(xy.variant == "modular x*y") & (xy.optim == "adamw") & (xy.seed == 0)]
    assert len(times) == 5  # 6 asked, the late_collapse cell lacks seed 3


def test_cluster_ci_and_sign_test():
    v = np.r_[np.zeros(10), np.ones(10)]
    cl = np.repeat(np.arange(10), 2)
    lo, hi = af._cluster_ci(v, cl)
    assert lo <= 0.5 <= hi
    w, l, p = af._sign_test(-np.ones(20), cl)
    assert (w, l) == (10, 0) and p == pytest.approx(2 / 2 ** 10)


def test_calibrate_picks_on_calibration_runs_and_scores_on_test_runs():
    meta = pd.DataFrame({"run_id": [f"r{i}" for i in range(8)], "cell": [f"c{i % 4}" for i in range(8)],
                         "category": "grokker", "app": "modular", "seed": [0, 0, 0, 0, 2, 2, 2, 2]})
    rows = []
    for i in range(8):
        cal = i < 4
        for fam, setting, cost in (("horizon", "horizon", 0.1), ("fixed_frac", "fixed_frac c=0.5", 0.05),
                                   ("const_delta", "A", 0.01 if cal else 0.2), ("const_delta", "B", 0.02)):
            rows.append({"run_id": f"r{i}", "family": fam, "setting": setting, "cost_0.1": cost, "premature": False,
                         "never": fam == "horizon", "regret": 0.0, "late": cost})
    t = af.calibrate(pd.DataFrame(rows), meta, {f"r{i}" for i in range(4)}, {f"r{i}" for i in range(4, 8)}, 0.1)
    t = t.set_index("family")
    assert t.loc["const_delta", "setting"] == "A"  # chosen on calibration
    assert t.loc["const_delta", "test_cost"] == pytest.approx(0.2)  # honest test score
    assert t.loc["const_delta", "hindsight_cost"] == pytest.approx(0.02)
    assert t.loc["F2 best", "setting"] == "A"
    assert t.loc["fixed_frac", "d_horizon"] == pytest.approx(-0.05)


def test_parallel_replay_analysis_and_report_end_to_end(dummy_run, tmp_path):
    cfg = AgopPathConfig()
    work, remote = tmp_path / "work", LocalRemote(tmp_path / "remote")
    store = TraceStore(work / "traces")
    info = {"run_id": dummy_run.name, "app": "modular", "variant": "modular x+y", "optim": "adamw", "lr": 1e-3,
            "gamma": 0.1, "alpha": 1.0, "size": None, "seed": 0, "category": "grokker", "cell": "c",
            "cell_category": "grokker", "max_test": 0.5, "t_eps": 50, "train99_step": 20, "steps": 200}
    status = af.replay_parallel([info], store, cfg, work, PROJECT, workers=2,
                                source={"kind": "dir", "root": str(dummy_run.parents[1]), "verify": True},
                                eval_kw={"n_log": 10, "n_uniform": 20}, device="cpu", out_remote=remote, poll=0.5,
                                log=lambda m: None)
    assert [s["ok"] for s in status] == [True], status
    assert af._trace_done(store, cfg, work, dummy_run.name)
    assert any(p.endswith(f"{dummy_run.name}.parquet") for p in remote.list_files("signals"))
    # resuming in a fresh work dir downloads the trace and curves
    store2 = TraceStore(tmp_path / "w2" / "traces")
    assert af.resume_from(remote, "x", store2, cfg, tmp_path / "w2", {dummy_run.name}, log=lambda m: None) == 2

    out = tmp_path / "analysis"
    st_an = af.analyze_parallel([info], store, cfg, work, out, PROJECT, workers=1, poll=0.5, log=lambda m: None)
    assert [s["ok"] for s in st_an] == [True], st_an
    rows, series, diag = af.load_analysis(out, [dummy_run.name])
    assert set(af.FAMILY_ORDER) - {"F2 best"} <= set(rows["family"])
    assert np.isfinite(rows["cost_0.1"]).all()

    # fake four seeds of one cell (and an x*y twin) to exercise calibration and the report
    sel = []
    for v in ("modular x+y", "modular x*y"):
        for s in range(4):
            rid = f"{v[-3:].replace('*', 'x').replace('+', 'p')}_s{s}"
            sel.append({**info, "run_id": rid, "seed": s, "variant": v, "cell": f"cell{v}"})
    meta = af.build_meta(sel)
    R = pd.concat([rows.assign(run_id=r) for r in meta.run_id], ignore_index=True)
    S = pd.concat([series.assign(run_id=r) for r in meta.run_id], ignore_index=True)
    D = pd.concat([diag.assign(run_id=r) for r in meta.run_id], ignore_index=True)
    files = af.build_report(R, S, D, meta, tmp_path / "report_AGOP")
    assert (tmp_path / "report_AGOP" / "REPORT_auto.md").exists()
    assert "figures/cost_seed_split.png" in files and files["figures/cost_seed_split.png"].exists()
    calib = pd.read_csv(tmp_path / "report_AGOP" / "tables" / "calibration.csv")
    assert set(calib["split"]) == {"seeds", "operation"}

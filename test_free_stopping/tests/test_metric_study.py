"""N1 / R1 / B5 study: selection (teacher-student split by size), lag changes, the rule
grid, calibration, and the trace -> analysis -> per-metric reports path on the dummy run."""
import numpy as np
import pandas as pd
import pytest

from stopping import metric_study as ms
from stopping.tfs_source import DirRunSource
from tfs.storage.hub import LocalRemote


def test_prepare_rows_splits_teacher_student_by_size():
    rows = [{"run_id": "a", "variant": "teacher_student", "size": 50.0},
            {"run_id": "b", "variant": "teacher_student", "size": 400}]
    out = ms.prepare_rows(rows, "teacher_student")
    assert [r["variant"] for r in out] == ["teacher_student n50", "teacher_student n400"]
    assert rows[0]["variant"] == "teacher_student"  # input untouched
    assert ms.prepare_rows(rows, "mnist")[0]["variant"] == "teacher_student"


def test_select_runs_takes_whole_teacher_student_cells_per_size():
    rows = []
    for n in (50, 100, 400):
        for o in ("adamw", "sgd"):
            for a, cat in enumerate(("grokker", "fast", "never")):
                for s in range(4):
                    rows.append({"run_id": f"n{n}{o}{a}s{s}", "variant": "teacher_student", "optim": o, "lr": 1e-3,
                                 "gamma": 0.1, "alpha": float(a), "size": n, "seed": s, "status": "complete",
                                 "category": cat if s == 0 else "fast"})
    sel = ms.select_runs({"teacher_student": rows})
    df = pd.DataFrame(sel)
    assert len(df) == 3 * 2 * 2 * 4  # 2 cells per (size, optimizer), 4 seeds each
    assert set(df["variant"]) == {"teacher_student n50", "teacher_student n100", "teacher_student n400"}
    cal, test = ms.split_ids(ms.build_meta(sel), "teacher_student", "size")
    assert len(cal) == 32 and len(test) == 16


def test_lag_change_growing_and_fixed():
    src = np.arange(0, 1001, 10, dtype=float)
    logy = src / 1000.0
    ev = np.array([0.0, 100.0, 500.0, 1000.0])
    d, gap = ms.lag_change(src, logy, ev, "growing", 0.5, 1000.0)
    assert np.isnan(d[0])
    assert d[1:] == pytest.approx([0.05, 0.25, 0.5])
    assert gap[1:] == pytest.approx([50, 250, 500])
    d, _ = ms.lag_change(src, logy, ev, "fixed", 0.1, 1000.0)
    assert np.isnan(d[0]) and d[1:] == pytest.approx([0.1, 0.1, 0.1])  # t=100 reaches back exactly to 0


def _synthetic(T=20000, gamma=0.1):
    steps = np.arange(0, T + 1, 50)
    w = 10 * np.exp(-steps / 3000) + 1  # shrinks, then settles
    tr = pd.DataFrame({"step": steps, "w_norm": w, "summary": steps % 500 == 0})
    s = tr["summary"]
    tr.loc[s, "erank_l1"] = 50 + 20 * np.exp(-steps[s] / 2000)
    tr.loc[s, "erank_l2"] = 30 + 10 * np.exp(-steps[s] / 2000)
    tr.loc[s, "alpha_l0"] = 2 + 4 * np.exp(-steps[s] / 4000)
    test = np.where(steps >= 8000, 1.0, 0.01)
    curves = {"step": steps.tolist(), "train_acc": [1.0] * len(steps), "test_acc": test.tolist()}
    info = {"run_id": "r", "optim": "adamw", "lr": 1e-3, "gamma": gamma, "max_test": 1.0, "t_eps": 8000,
            "category": "grokker"}
    return tr, curves, info


def test_analyze_run_scores_the_full_grid():
    rows, series = ms.analyze_run(*_synthetic())
    n = rows.groupby("metric")["setting"].nunique()
    assert n["N1"] == 7 * 2 * 5 * 2  # lag x normalization x eps x rule (abs, shrink)
    assert n["R1"] == 3 * 7 * 5 * 2
    assert n["B5"] == 2 * 5 + 2 * 3 * 3
    assert n["ref"] == len(ms.FIXED_FRACS) + 2
    assert rows.groupby(["metric", "family", "setting"]).size().max() == 1
    assert np.isfinite(rows["cost_0.1"]).all()
    assert len(series) == 41 and {"w_norm", "erank_mean", "alpha_mean"} <= set(series.columns)
    # the norm settles late (exp(-t/3000)); a strict settle rule must not stop before it has
    # change over 200 steps ~ 0.67 exp(-t/3000) / w <= 0.003 from t ~ 16,200 (+ patience)
    s = rows[rows["setting"] == "N1 abs fixed c=0.01 norm=none eps=0.003"]["stop"].iloc[0]
    assert 15500 <= s <= 18500
    # over a lag of 25% of t the change never falls below 0.001 within the budget
    assert np.isnan(rows[rows["setting"] == "N1 abs growing f=0.25 norm=none eps=0.001"]["stop"].iloc[0])
    # B5 level rules: alpha <= 4 is crossed once 4 exp(-t/4000) <= 2, i.e. t >= 2773, + patience
    lvl = rows[rows["setting"] == "B5 mean alpha<=4.0"]["stop"].iloc[0]
    assert 2500 <= lvl <= 5000


def test_analyze_run_at_gamma_zero_falls_back_to_unnormalized():
    r0, _ = ms.analyze_run(*_synthetic(gamma=0.0))
    for rule in ("abs", "shrink"):  # decay eps=0.1 is the 3rd threshold; at gamma = 0 it uses none's 3rd, 0.01
        a = r0[r0["setting"] == f"N1 {rule} fixed c=0.05 norm=decay eps=0.1"]
        b = r0[r0["setting"] == f"N1 {rule} fixed c=0.05 norm=none eps=0.01"]
        assert len(a) == 1 and list(a["stop"]) == list(b["stop"])


def test_calibrate_tunes_on_calibration_runs():
    meta = pd.DataFrame({"run_id": [f"r{i}" for i in range(8)], "cell": [f"c{i % 4}" for i in range(8)],
                         "app": "modular", "seed": [0, 0, 0, 0, 2, 2, 2, 2]})
    rows = []
    for i in range(8):
        cal = i < 4
        for fam, setting, cost in (("horizon", "horizon", 0.1), ("fixed_frac", "fixed_frac c=0.5", 0.05),
                                   ("oracle_teps", "oracle_teps", 0.0),
                                   ("n1_abs", "A", 0.01 if cal else 0.2), ("n1_shrink", "B", 0.02)):
            rows.append({"run_id": f"r{i}", "family": fam, "setting": setting, "cost_0.1": cost,
                         "premature": False, "never": False, "regret": cost, "late": 0.0})
    t = ms.calibrate(pd.DataFrame(rows), meta, {"r0", "r1", "r2", "r3"}, {"r4", "r5", "r6", "r7"}, 0.1, "N1")
    t = t.set_index("family")
    assert t.loc["n1_abs", "setting"] == "A" and t.loc["n1_abs", "test_cost"] == pytest.approx(0.2)
    assert t.loc["N1 best", "setting"] == "A"  # chosen on calibration, even though B wins on test
    assert t.loc["N1 best", "hindsight_setting"] == "B"
    assert t.loc["n1_shrink", "d_fixed"] == pytest.approx(-0.03)


def test_trace_analysis_and_reports_end_to_end(dummy_run, tmp_path):
    work, remote = tmp_path / "work", LocalRemote(tmp_path / "remote")
    info = {"run_id": dummy_run.name, "app": "modular", "variant": "modular x+y", "optim": "adamw", "lr": 1e-3,
            "gamma": 0.1, "alpha": 1.0, "size": None, "seed": 0, "category": "grokker", "cell": "c",
            "cell_category": "grokker", "max_test": 0.5, "t_eps": 50, "train99_step": 20, "steps": 200}
    src = DirRunSource(dummy_run.parents[1])
    status = ms.run_all([info], work, source=src, out_remote=remote, min_eigs=4, log=lambda m: None)
    assert [s["ok"] for s in status] == [True], status
    assert status[0]["did"] == "trace+analysis"
    trace = pd.read_parquet(work / "traces" / f"{dummy_run.name}.parquet")
    assert {"w_norm", "summary"} <= set(trace.columns)
    assert any(c.startswith("erank_l") for c in trace.columns) and any(c.startswith("alpha_l") for c in trace.columns)
    assert f"traces/{dummy_run.name}.parquet" in remote.list_files("traces")
    # a second call does nothing; a fresh work dir resumes from the remote
    assert ms.run_all([info], work, source=src, log=lambda m: None)[0]["did"] == "cached"
    assert ms.resume_from(remote, tmp_path / "w2", {dummy_run.name}, log=lambda m: None) == 2

    rows, series = ms.load_analysis(work, [dummy_run.name])
    assert set(rows["metric"]) == {"N1", "R1", "B5", "ref"}
    sel = []
    for v in ("modular x+y", "modular x*y"):
        for s in range(4):
            sel.append({**info, "run_id": f"{v[-3:]}_s{s}", "seed": s, "variant": v, "cell": f"cell{v}"})
    meta = ms.build_meta(sel)
    R = pd.concat([rows.assign(run_id=r) for r in meta.run_id], ignore_index=True)
    S = pd.concat([series.assign(run_id=r) for r in meta.run_id], ignore_index=True)
    files = ms.build_reports(R, S, meta, tmp_path / "out")
    for m in ms.METRICS:
        assert f"report_{m}/REPORT_auto.md" in files
        assert files[f"report_{m}/figures/cost_heldout.png"].exists()
        calib = pd.read_csv(tmp_path / "out" / f"report_{m}" / "tables" / "calibration.csv")
        assert set(calib["family"]) == set(ms.family_order(m))
        assert set(calib["split"]) == {"seeds", "operation"}

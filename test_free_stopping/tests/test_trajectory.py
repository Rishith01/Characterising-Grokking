"""Phase 5 pipeline (trajectory/): known-answer tests for the quantities, causality of the
online AGOP angles, and an end-to-end run on tiny trained runs (CPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_cfg, train
from trajectory import layout
from trajectory.corpus import corpus_table
from trajectory.grid import common_grid, early_points, nearest_index
from trajectory.quantities import agop as qa
from trajectory.quantities import compute_curves
from trajectory.quantities.norm import rho_bar
from trajectory.quantities.rank import participation_ratio
from trajectory.rawfile import load_raw, save_raw


# ---- grid ----------------------------------------------------------------------------------

def test_grid_corpus_shape():
    g = common_grid(100_000, 50, 500)
    assert g[0] == 0 and g[-1] == 100_000
    assert set(range(0, 100_001, 500)) <= set(g.tolist())
    assert np.all(g % 50 == 0) and np.all(np.diff(g) > 0)
    early = early_points(g, 500, 100_000)
    assert 15 <= len(early) <= 30 and early.max() < 5000
    assert 215 <= len(g) <= 235


def test_nearest_index_ties_to_earlier():
    g = np.array([0, 10, 20])
    assert nearest_index(g, np.array([5, 14, 16, 100])).tolist() == [0, 1, 2, 2]


# ---- N3 ------------------------------------------------------------------------------------

def _decay_run(lr=1e-3, gamma=0.01, w0=400.0, every=50, T=20_000, drift=0.0):
    steps = np.arange(0, T + 1, every)
    sq = w0 * (1 - lr * gamma) ** (2 * steps) * (1 + drift * steps / T)
    return steps, sq, lr * gamma * sq


def test_rho_bar_is_one_under_pure_decay():
    steps, sq, D = _decay_run()
    grid = np.arange(0, 20_001, 500)
    r = rho_bar(steps, sq, D, grid)
    assert np.isnan(r[0])
    assert np.allclose(r[1:], 1.0, atol=1e-3)


def test_rho_bar_zero_when_stationary_and_negative_when_growing():
    steps = np.arange(0, 10_001, 50)
    grid = np.arange(0, 10_001, 500)
    D = np.full(len(steps), 4e-3)
    assert np.allclose(rho_bar(steps, np.full(len(steps), 400.0), D, grid)[1:], 0.0)
    growing = 400.0 + 1e-3 * steps
    assert np.all(rho_bar(steps, growing, D, grid)[1:] < 0)


# ---- AGOP angles ---------------------------------------------------------------------------

def _unit(a):
    return a / np.linalg.norm(a)


def test_angle_between_disjoint_psd_matrices_is_right_angle():
    a, b = _unit(np.diag([1.0, 0.0])), _unit(np.diag([0.0, 1.0]))
    d2 = float(((a - b) ** 2).sum())
    assert qa.angle(np.array(d2)) == pytest.approx(np.pi / 2)
    assert qa.angle(np.array(0.0)) == 0.0


def _random_path(n=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    xs = []
    a = rng.normal(size=(d, d))
    for _ in range(n):
        a = a + 0.1 * rng.normal(size=(d, d))
        m = a @ a.T
        xs.append(_unit(m).ravel())
    xs = np.array(xs)
    return ((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1)


def test_alignment_to_final_is_one_at_the_end_and_matches_cosine():
    d2 = _random_path()
    al = qa.alignment_to_final(d2)
    assert al[-1] == pytest.approx(1.0)
    assert np.all(al <= 1.0 + 1e-12) and np.all(al >= -1e-12)


def test_online_angles_are_causal():
    """AO1 / AO2 at grid index i must not change when the run is cut after i."""
    d2 = _random_path(n=50)
    steps = np.arange(0, 50 * 500, 500)
    T = float(steps[-1])
    full_half = qa.angle_to_lag(d2, steps, np.where(steps > 0, steps / 2, -1.0))
    full_c = qa.angle_to_lag(d2, steps, np.full(len(steps), 0.05 * T))
    for k in (10, 25, 40):
        cut = d2[:k, :k]
        s = steps[:k]
        np.testing.assert_array_equal(qa.angle_to_lag(cut, s, np.where(s > 0, s / 2, -1.0)), full_half[:k])
        np.testing.assert_array_equal(qa.angle_to_lag(cut, s, np.full(k, 0.05 * T)), full_c[:k])


def test_participation_ratio():
    assert participation_ratio(np.ones(5)) == pytest.approx(5.0)
    assert participation_ratio(np.array([1.0, 0, 0])) == pytest.approx(1.0)


# ---- corpus --------------------------------------------------------------------------------

def test_corpus_marks_gamma_zero_adam_duplicates():
    base = dict(variant="modular x+y", lr=1e-3, alpha=1.0, size=0.5, seed=0, steps=100_000, wall_s=1.0)
    rows = [dict(base, run_id="modular__a", optim="adamw", gamma=0.0),
            dict(base, run_id="modular__b", optim="adam_l2", gamma=0.0),
            dict(base, run_id="modular__c", optim="adam_l2", gamma=1e-4),
            dict(base, run_id="modular__d", optim="adamw", gamma=0.0, variant="modular x*y")]
    t = corpus_table(rows)
    assert list(t["run_id"]) == ["modular__a", "modular__b", "modular__c"]
    assert t.set_index("run_id")["duplicate_of"].to_dict() == {"modular__a": None, "modular__b": "modular__a",
                                                              "modular__c": None}


def test_rawfile_roundtrip(tmp_path):
    p = save_raw(tmp_path / "x" / "r.npz", {"a": np.arange(3), "b": np.eye(2)}, {"k": 1})
    arrays, meta = load_raw(p)
    assert meta == {"k": 1} and arrays["a"].tolist() == [0, 1, 2]


# ---- end to end on tiny trained runs -------------------------------------------------------

def _extract(run_dir):
    from stopping.tfs_source import TfsRunHandle
    from trajectory.extract.recorded import extract_recorded
    from trajectory.extract.replay import extract_replay
    rec = extract_recorded(TfsRunHandle(run_dir))
    rep = extract_replay(run_dir, device="cpu", verify=True)
    return rec, rep


def test_end_to_end_adamw(dummy_run):
    rec, rep = _extract(dummy_run)
    assert rep[1]["replay_level"] == "L0"
    curves = compute_curves(rec, rep)
    got = set(curves["metric"])
    assert {"N1", "N2", "N3", "R1", "R2", "R3", "AP1", "AO1", "AO2"} <= got
    grid = rep[0]["grid"]
    n1 = curves[(curves.metric == "N1")].set_index("step")["value"]
    assert list(n1.index) == grid.tolist() and n1.notna().all()
    # recorded and replayed spectra together cover every grid step
    r1 = curves[(curves.metric == "R1") & (curves.variant == "main") & (curves.layer == 0)]
    assert r1["value"].notna().all()
    ap = curves[(curves.metric == "AP1") & (curves.variant == "main")].set_index("step")["value"]
    assert ap.iloc[-1] == pytest.approx(1.0)
    ao2 = curves[(curves.metric == "AO2") & (curves.variant == "main")]
    assert ao2["value"].notna().any()


def test_early_spectra_match_recorded_summaries(dummy_run):
    """The replay probe uses the Recorder's own code, so at a summary step it must give the
    stored summaries exactly (bit-identical replay)."""
    from tfs.storage.reader import RunReader
    from trajectory.extract.probes import EarlySpectraProbe
    from trajectory.extract.replay import extract_replay

    class AtSummaries(EarlySpectraProbe):
        def steps(self, plan):
            return {50, 100}

    arrays, _ = extract_replay(dummy_run, probes=[AtSummaries()], device="cpu", verify=True)
    reader = RunReader(dummy_run)
    for j, s in enumerate(arrays["early_steps"]):
        stored = reader.summaries(int(s))
        np.testing.assert_array_equal(arrays["early_sv_l0"][j], stored["sv/l0"].astype(np.float64))
        np.testing.assert_array_equal(arrays["early_actspec_l1"][j], stored["actspec/l1"].astype(np.float64))


def test_adam_l2_gets_n3_only_with_replay(dummy_cfg, tmp_path):
    run = train(make_cfg(dummy_cfg, "adam_l2"), tmp_path)
    rec, rep = _extract(run)
    without = compute_curves(rec, None)
    assert "N3" not in set(without["metric"])
    with_rep = compute_curves(rec, rep)
    n3 = with_rep[(with_rep.metric == "N3") & (with_rep.layer == -1)]["value"]
    assert n3.iloc[2:].notna().all()  # the first interval contains step 0, before Adam has state


def test_runner_launch_and_curves(dummy_run, tmp_path):
    """Two worker processes over a local run directory, then Stage Q."""
    from tests.conftest import PROJECT
    from trajectory.runner import build_curves, launch

    rid = dummy_run.name
    root = dummy_run.parent.parent
    runs = [{"run_id": rid, "app": dummy_run.parent.name}]
    for stage in ("recorded", "replay"):
        status = launch(stage, runs, tmp_path, PROJECT, workers=2, source={"kind": "dir", "root": str(root)},
                        poll=0.5, log=lambda m: None)
        assert [s["ok"] for s in status] == [True], status
        assert layout.raw_path(tmp_path, stage, rid).exists()
    curves, problems = build_curves(pd.DataFrame({"run_id": [rid]}), tmp_path, log=lambda m: None)
    assert not problems and set(curves["run_id"]) == {rid}
    assert (tmp_path / layout.CURVES).exists()


def test_upload_then_resume_in_a_new_session(dummy_run, tmp_path):
    """Raw files are uploaded as they finish; a new session (empty work dir) resumes them
    from the remote and launches nothing."""
    from tests.conftest import PROJECT
    from tfs.storage.hub import LocalRemote
    from trajectory.runner import launch, resume

    rid = dummy_run.name
    runs = [{"run_id": rid, "app": dummy_run.parent.name}]
    src = {"kind": "dir", "root": str(dummy_run.parent.parent)}
    remote = LocalRemote(tmp_path / "remote")
    first = tmp_path / "session1"
    launch("recorded", runs, first, PROJECT, workers=1, source=src, out_remote=remote, upload_every=1,
           poll=0.5, log=lambda m: None)
    assert layout.raw_rel("recorded", rid) in remote.list_files("raw/recorded")
    second = tmp_path / "session2"
    assert resume(remote, second, "recorded", {rid}, log=lambda m: None) == 1
    assert layout.raw_path(second, "recorded", rid).exists()
    assert launch("recorded", runs, second, PROJECT, workers=1, source=src, out_remote=remote, poll=0.5,
                  log=lambda m: None) == []


def test_pilot_covers_every_variant_and_optimizer():
    from trajectory.corpus import pilot_runs
    rows = []
    for v in ("modular x+y", "mnist d3"):
        for o in ("adamw", "adam_l2", "sgd", "sgd_momentum"):
            for g, s in ((0.0, 0), (1e-3, 1)):
                rows.append(dict(run_id=f"{v[:3]}__{o}_{g}_{s}", variant=v, optim=o, gamma=g, lr=1e-3, alpha=1.0,
                                 size=0.5, seed=s, wall_s=1.0))
    t = corpus_table(rows)
    pick = pilot_runs(t, candidates={"mod__sgd_0.0_0"})
    assert len(pick) == 8
    by = {(r["variant"], r["optim"]): r for r in pick}
    assert by[("modular x+y", "sgd")]["run_id"] == "mod__sgd_0.0_0"  # candidate preferred
    assert by[("mnist d3", "adam_l2")]["gamma"] > 0  # gamma > 0 preferred: exercises AdamDecayProbe


def test_compare_with_f2_matches_identical_values():
    from trajectory.validate import compare_with_f2
    steps = [0, 500, 1000]
    curves = pd.concat([
        pd.DataFrame({"run_id": "r", "metric": "AP1", "variant": "main", "layer": -1, "step": steps,
                      "value": [0.5, 0.8, 1.0]}),
        pd.DataFrame({"run_id": "r", "metric": "N1", "variant": "main", "layer": -1, "step": steps,
                      "value": np.log([10.0, 9.0, 8.0])}),
    ])
    f2 = pd.DataFrame({"run_id": "r", "step": [0, 500, 750, 1000], "align_M": [0.5, 0.8, 0.9, 1.0],
                       "align_G": np.nan, "erank_G": np.nan, "w_norm": [10.0, 9.0, 8.5, 8.0]})
    c = compare_with_f2(curves, f2).set_index("quantity")
    assert c.loc["AP1 main vs align_M", "shared_steps"] == 3
    assert c.loc["AP1 main vs align_M", "max_rel_diff"] == pytest.approx(0.0)
    assert c.loc["N1 main vs w_norm", "max_rel_diff"] < 1e-12

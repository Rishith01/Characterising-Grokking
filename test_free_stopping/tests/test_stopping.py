"""Phase 4 package `stopping`: rules, methods, store, math helpers, the runner and the
tfs adapter (DESIGN.md §13)."""
import numpy as np
import pandas as pd
import pytest
import torch

from stopping.compute import compute, compute_run, eval_grid
from stopping.heldout import ValLossConfig, b6_method
from stopping.method import Method
from stopping.methods import default_methods, signal_configs
from stopping.rule import Rule, RuleConfig
from stopping.signal import config_from_dict
from stopping.signals import (DisparityConfig, ERankConfig, EvidenceConfig, HtsrAlphaConfig, HtsrTrapsConfig,
                              LabelWaveConfig, StepConfig, TrainLossConfig, WeightNormConfig)
from stopping.signals._linalg import (batch_grad, correlation_traps, effective_rank, evidence,
                                      per_example_moments, powerlaw_fit)
from stopping.store import TraceStore
from stopping.tfs_source import DirRunSource, HfRunSource, _loss_parts, parts_for
from tfs.losses import compute_loss
from tfs.storage.hub import LocalRemote, push_runs

ALL = [StepConfig(), TrainLossConfig(), WeightNormConfig(window=3), ERankConfig(window=2),
       HtsrAlphaConfig(min_eigs=8), HtsrTrapsConfig(min_eigs=8), LabelWaveConfig(), DisparityConfig(),
       DisparityConfig(form="replay"), EvidenceConfig(), ValLossConfig()]


def _df(values, steps=None):
    steps = list(range(len(values))) if steps is None else steps
    return pd.DataFrame({"step": steps, "value": values})


# ---- rules and methods ------------------------------------------------------------------

def test_threshold_rule_needs_patience_consecutive_rows():
    df = _df([5, 1, 5, 1, 1, 1, 5])
    r = RuleConfig(direction="below", threshold=2, patience=3, smoother="none")
    assert Rule(r).decide(df).step == 5
    assert Rule(RuleConfig(direction="below", threshold=2, patience=4, smoother="none")).decide(df).step is None


def test_min_step_delays_the_stop():
    df = _df([1, 1, 1, 1, 1])
    r = RuleConfig(direction="below", threshold=2, patience=1, smoother="none", min_step=3)
    assert Rule(r).decide(df).step == 3


def test_plateau_rule_counts_rows_without_improvement():
    df = _df([10, 9, 8, 8, 8.1, 7.99, 7.995, 7.99])
    # improvement must beat the best by > 1% of |best|; 7.99 vs 8 is not one
    r = RuleConfig(mode="plateau", direction="below", threshold=0.01, patience=4, smoother="none")
    assert Rule(r).decide(df).step == 6
    up = RuleConfig(mode="plateau", direction="above", threshold=0.0, patience=2, smoother="none")
    assert Rule(up).decide(_df([1, 2, 3, 2, 2, 4])).step == 4


def test_ema_smoothing_and_nan_rows():
    t = Rule(RuleConfig(threshold=0, smoother="ema", smooth_param=0.5)).trace(_df([2.0, np.nan, 0.0]))
    assert t["smooth"].tolist() == [2.0, 2.0, 1.0]
    assert not t["cond"].iloc[1]


def test_rule_is_causal():
    rng = np.random.default_rng(0)
    df = _df(list(rng.normal(size=60)))
    r = Rule(RuleConfig(direction="below", threshold=-0.5, patience=2, smoother="ema", smooth_param=0.3))
    full = r.trace(df)
    part = r.trace(df.iloc[:30])
    pd.testing.assert_frame_equal(full.iloc[:30], part)


def test_method_stops_at_the_earliest_part():
    a, b = StepConfig(), TrainLossConfig()
    m = Method("m", ((a, RuleConfig(direction="above", threshold=5, smoother="none")),
                     (b, RuleConfig(direction="below", threshold=0.5, smoother="none"))))
    traces = {a.key: _df(list(range(10))), b.key: _df([1, 1, 0.1] + [1] * 7)}
    d = m.decide(traces)
    assert (d.step, d.part) == (2, 1)


# ---- configs and store ------------------------------------------------------------------

def test_config_hash_is_stable_and_round_trips():
    c = WeightNormConfig(window=7)
    assert c.hash() == WeightNormConfig(window=7).hash() != WeightNormConfig().hash()
    assert config_from_dict(c.to_dict()) == c
    assert len({s.key for s in ALL}) == len(ALL)


def test_store_round_trip(tmp_path):
    st, c = TraceStore(tmp_path), ERankConfig()
    st.write(c, "r1", _df([1.0, 2.0]))
    st.write(c, "r2", _df([3.0]))
    assert st.has(c, "r1") and not st.has(c, "r3")
    assert st.load(c)["run_id"].tolist() == ["r1", "r1", "r2"]
    files = st.upload_files([c])
    assert set(files) == {f"signals/erank/{c.hash()}/traces.parquet", f"signals/erank/{c.hash()}/config.json"}


def test_eval_grid_is_on_log_steps():
    log = np.arange(0, 100_001, 50)
    g = eval_grid(log)
    assert g[0] == 0 and g[-1] == 100_000 and set(g) <= set(log)
    assert 400 <= len(g) <= 600 and (np.diff(g) > 0).all()


# ---- math helpers -----------------------------------------------------------------------

def test_effective_rank():
    assert effective_rank(np.ones(7)) == pytest.approx(7)
    assert effective_rank(np.array([3.0, 0, 0])) == pytest.approx(1)
    assert np.isnan(effective_rank(np.zeros(3)))


def test_powerlaw_fit_recovers_alpha():
    rng = np.random.default_rng(1)
    x = (1 - rng.random(5000)) ** (-1 / 1.5)  # Pareto, p(x) ~ x^-2.5 on x >= 1
    a, xmin, ks, n = powerlaw_fit(x)
    assert a == pytest.approx(2.5, abs=0.1) and ks < 0.05


def test_no_traps_in_a_gaussian_matrix_and_traps_with_a_huge_element():
    rng = np.random.default_rng(0)
    w = torch.tensor(rng.normal(size=(400, 200)))
    assert correlation_traps(w, np.random.default_rng(0)) == 0
    w[3, 5] = 400.0
    assert correlation_traps(w, np.random.default_rng(0)) >= 1


@pytest.mark.parametrize("loss,kind", [("ce", "classification"), ("mse", "classification"), ("mse", "regression")])
def test_per_example_gradients_average_to_the_batch_gradient(loss, kind):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)).double()
    x = torch.randn(20, 6, dtype=torch.float64)
    y = torch.randint(0, 4, (20,)) if kind == "classification" else torch.randn(20, 4, dtype=torch.float64)
    target, loss_fn = _loss_parts(loss, kind, y, 4)
    target = target.double() if target.is_floating_point() else target
    s1, s2, n = per_example_moments(model, x, target, loss_fn, chunk=7)
    ref = compute_loss(loss, model(x), y, kind)
    gs = torch.autograd.grad(ref, list(model.parameters()))
    for (name, _), g in zip(model.named_parameters(), gs):
        assert torch.allclose(s1[name] / n, g.reshape(-1), atol=1e-10)
    bg = batch_grad(model, x, target, loss_fn)
    assert all(torch.allclose(bg[k], s1[k] / n, atol=1e-10) for k in bg)


def test_evidence_formula():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(200, 3))  # zero-mean noise: EB near 0
    eb, p = evidence(torch.tensor(g.sum(0)), torch.tensor((g ** 2).sum(0)), 200)
    mean, var = g.mean(0), g.var(0, ddof=1)
    assert p == 3 and eb == pytest.approx(1 - 200 / 3 * (mean ** 2 / var).sum())
    big = g + 5.0  # a real gradient: EB strongly negative
    eb2, _ = evidence(torch.tensor(big.sum(0)), torch.tensor((big ** 2).sum(0)), 200)
    assert eb2 < -100
    dead = np.zeros((10, 2))
    assert np.isnan(evidence(torch.tensor(dead.sum(0)), torch.tensor(dead.sum(0)), 10)[0])


# ---- the runner on a real (dummy) run ---------------------------------------------------

@pytest.fixture(scope="module")
def traces(dummy_run):
    src = DirRunSource(dummy_run.parents[1])
    with src.open(dummy_run.name, set()) as h:
        return compute_run(h, ALL), h.context


def test_every_signal_emits_on_its_cadence(traces):
    tr, ctx = traces
    log, summ = set(ctx.log_steps.tolist()), set(ctx.summary_steps.tolist())
    for c in ALL:
        df = tr[c.key]
        assert len(df) > 0, c.key
        steps = set(df["step"])
        cad = c.build().cadence
        assert steps <= (summ if cad == "summary" else log), c.key
        values = df["value"].to_numpy(dtype=float)
        if c == DisparityConfig():  # dummy n_train = 256 = probe batch: degenerate by design
            assert np.isnan(values).all()
        else:
            assert np.isfinite(values).any(), c.key
    assert len(tr[LabelWaveConfig().key]) == len(log) - 1  # no change at the first log step


def test_values_match_the_stored_data(traces, dummy_run):
    from tfs.storage.reader import RunReader
    tr, _ = traces
    tel = RunReader(dummy_run).telemetry()
    w = np.sqrt(sum(tel[f"l{i}_w_norm"] ** 2 for i in range(3)))
    assert np.allclose(tr[WeightNormConfig(window=3).key]["value"], w)
    assert np.allclose(tr[TrainLossConfig().key]["value"], tel["train_loss"])
    assert tr[EvidenceConfig().key]["batch"].eq(64).all()  # the dummy run's minibatch size


def test_runner_is_causal(dummy_run, traces):
    full, _ = traces
    src = DirRunSource(dummy_run.parents[1])
    with src.open(dummy_run.name, set()) as h:
        cut = compute_run(h, ALL, max_step=120)
    for c in ALL:
        a = full[c.key][full[c.key]["step"] <= 120].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, cut[c.key], obj=c.key)


def test_only_heldout_signals_see_heldout_data(dummy_run, monkeypatch):
    from stopping.signals.basic import TrainLoss
    seen = []
    orig = TrainLoss.update
    monkeypatch.setattr(TrainLoss, "update", lambda self, obs: (seen.append(obs.heldout), orig(self, obs))[1])
    src = DirRunSource(dummy_run.parents[1])
    with src.open(dummy_run.name, set()) as h:
        out = compute_run(h, [ValLossConfig(), TrainLossConfig()])
    assert out[ValLossConfig().key]["value"].notna().all()
    assert seen and all(x is None for x in seen)


def test_compute_caches_and_resumes(dummy_run, tmp_path):
    src = DirRunSource(dummy_run.parents[1])
    st = TraceStore(tmp_path / "traces")
    cfgs = [TrainLossConfig(), ERankConfig()]
    s1 = compute(src, src.run_ids(), cfgs, st, log=lambda m: None)
    assert s1[0]["ok"] and s1[0]["signals"] == 2
    s2 = compute(src, src.run_ids(), cfgs, st, log=lambda m: None)
    assert s2[0]["signals"] == 0


def test_hf_source_downloads_only_the_needed_parts(dummy_run, tmp_path):
    import shutil
    from tfs.storage import layout
    copy = tmp_path / "local" / dummy_run.parent.name / dummy_run.name
    shutil.copytree(dummy_run, copy)
    remote = LocalRemote(tmp_path / "remote")
    push_runs(remote, [copy], log=lambda m: None)
    src = HfRunSource(prefix="", work=tmp_path / "work", remote=remote)
    assert src.run_ids() == [dummy_run.name]
    assert parts_for({"models"}) == ["keyframes", "telemetry"]
    with src.open(dummy_run.name, {"telemetry"}) as h:
        present = {p.name for p in h.run_dir.iterdir()}
        assert layout.TELEMETRY in present and layout.KEYFRAMES not in present
        assert layout.ORACLE not in present
        run_dir = h.run_dir
    assert not run_dir.exists()  # deleted on close


def test_default_methods_decide_on_the_dummy_run(traces):
    tr, _ = traces
    ms = default_methods(budget=100) + [b6_method()]
    for m in ms:
        assert all(s.key in tr or s not in ALL for s in m.signals())
    m0 = ms[0]
    assert m0.decide({StepConfig().key: tr[StepConfig().key]}).step == 100
    assert len(signal_configs(ms)) == len({s.key for m in ms for s in m.signals()})

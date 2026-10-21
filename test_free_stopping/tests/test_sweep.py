import pytest

from tfs.storage.reader import RunReader
from tfs.sweep.grid import expand
from tfs.sweep.launcher import run_parallel

BASE = {
    "task": {"name": "dummy", "params": {"n_train": 64, "n_test": 64}},
    "model": {"depth": 2, "width": 8, "activation": "relu", "init_scale": 1.0},
    "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.1},
    "loss": "ce", "steps": 20, "seed": 0,
    "record": {"log_every": 5, "summary_every": 10, "keyframe_every": 10},
}


def test_variants_times_product():
    spec = {"blocks": [{"base": BASE,
                        "variants": [{"model.init_scale": 2.0}, {"optim": {"name": "sgd", "lr": 0.1}}],
                        "product": {"seed": [0, 1, 2]}}]}
    cfgs = expand(spec)
    assert len(cfgs) == 6
    assert {c.model.init_scale for c in cfgs[:3]} == {2.0} and {c.seed for c in cfgs[:3]} == {0, 1, 2}
    # a dict override replaces the whole sub-tree: weight_decay falls back to its default
    assert cfgs[3].optim.name == "sgd" and cfgs[3].optim.weight_decay == 0.0
    assert BASE["model"]["init_scale"] == 1.0  # base not mutated


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        expand({"blocks": [{"base": BASE}, {"base": BASE}]})


def test_phase1_sweeps_expand():
    assert len(expand("configs/sweeps/phase1_repro.yaml")) == 10
    assert len(expand("configs/sweeps/phase1_det.yaml")) == 16
    assert len(expand("configs/sweeps/phase1b_repro.yaml")) == 11
    assert len(expand("configs/sweeps/phase1b_lr_scan.yaml")) == 16


def test_launcher_runs_and_resumes(tmp_path):
    cfgs = expand({"blocks": [{"base": BASE, "product": {"seed": [0, 1]}}]})
    res = run_parallel(cfgs, tmp_path, "cpu", workers=2, log=lambda *_: None)
    assert sorted(r.status for r in res) == ["complete", "complete"]
    assert all(RunReader(r.run_dir).status == "complete" for r in res)
    again = run_parallel(cfgs, tmp_path, "cpu", workers=2, log=lambda *_: None)
    assert {r.status for r in again} == {"skipped"}


def test_markdown_report():
    from tfs.evaluation.report import rows_to_markdown, summary_markdown

    md = rows_to_markdown([{"a": 1, "b": 0.123456}, {"a": "x|y", "c": [1, 2]}])
    assert md.splitlines()[0] == "| a | b | c |" and "x/y" in md and "0.1235" in md
    assert "## t" in summary_markdown("T", {"t": [{"a": 1}]})

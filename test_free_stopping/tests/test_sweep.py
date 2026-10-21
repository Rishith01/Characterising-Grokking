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


def test_launcher_chains_replay_check(tmp_path):
    cfgs = expand({"blocks": [{"base": BASE, "product": {"seed": [0, 1]}}]})
    seen = []
    res = run_parallel(cfgs, tmp_path, "cpu", workers=2, log=lambda *_: None, replay_check=True,
                       on_result=seen.append)
    assert [r.replay_level for r in res] == ["L0", "L0"] and len(seen) == 2
    assert all((r.run_dir / "replay_report.json").exists() for r in res)


def test_sweep_uploads_and_resumes(tmp_path):
    from scripts.sweep import sweep
    from tfs.storage.hub import LocalRemote, PrefixRemote, read_manifest

    cfgs = expand({"blocks": [{"base": BASE, "product": {"seed": [0, 1, 2]}}]})
    remote = PrefixRemote(LocalRemote(tmp_path / "remote"), "phase2")
    rows, curves, info = sweep(cfgs, tmp_path / "runs", "cpu", 2, remote, batch_size=2, log=lambda *_: None)
    assert info["pushed"] == 3 and info["upload_failed"] == []
    assert {r["replay_level"] for r in rows} == {"L0"} and set(curves) == {c.run_id for c in cfgs}
    assert set(read_manifest(remote)) == {c.run_id for c in cfgs}
    assert not any((tmp_path / "runs").rglob("DONE"))  # local copies deleted after verification
    rows2, _, info2 = sweep(cfgs, tmp_path / "runs", "cpu", 2, remote, log=lambda *_: None)
    assert rows2 == [] and info2["skipped_on_remote"] == 3


def test_run_summary_and_category(dummy_run):
    from tfs.evaluation.run_summary import provisional_category, run_curves, run_summary

    row = run_summary(dummy_run)
    assert row["status"] == "complete" and 0 <= row["max_test"] <= 1 and row["t_eps"] is not None
    c = run_curves(dummy_run, n_points=10)
    assert len(c["step"]) == len(c["test_acc"]) <= 10 and c["step"] == sorted(c["step"])
    base = {"status": "complete", "max_test": 0.98, "final_test": 0.98, "train99_step": 500}
    assert provisional_category({**base, "t_eps": 20000}) == "grokker"
    assert provisional_category({**base, "t_eps": 1500}) == "fast"
    assert provisional_category({**base, "max_test": 0.1, "final_test": 0.1, "t_eps": 0}) == "never"
    assert provisional_category({**base, "final_test": 0.6, "t_eps": 150}) == "late_collapse"
    assert provisional_category({**base, "train99_step": 3000, "t_eps": 3500}) == "joint_plateau"


def test_env_lock_round_trip():
    from scripts.env_lock import lock_text, mismatches, parse_key_fields

    text = lock_text("cpu")
    fields = parse_key_fields(text)
    assert fields["torch"] and "torch==" in text and mismatches(text, text) == []
    assert mismatches(text, text.replace(f"torch: {fields['torch']}", "torch: 0.0")) == [
        f"torch: {fields['torch']} -> 0.0"]

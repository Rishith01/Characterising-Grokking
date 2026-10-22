from collections import Counter
from pathlib import Path

from tfs.storage.hub import LocalRemote, PrefixRemote, read_manifest
from tfs.sweep import corpus
from tfs.sweep.grid import expand
from tfs.sweep.lr_select import variant_label

PROJECT = Path(__file__).resolve().parents[1]
BASE = {
    "task": {"name": "dummy", "params": {"n_train": 64, "n_test": 64}},
    "model": {"depth": 2, "width": 8, "activation": "relu", "init_scale": 1.0},
    "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.1},
    "loss": "ce", "steps": 20, "seed": 0,
    "record": {"log_every": 5, "summary_every": 10, "keyframe_every": 10},
}


def test_phase3_sweeps_match_the_frozen_grid():
    """PLAN.md §3, "Phase 3 grid: locked values"."""
    cfgs = {d: expand(corpus.sweep_file(PROJECT, d)) for d in corpus.DATASETS}
    counts = {d: Counter(variant_label(c) for c in cs) for d, cs in cfgs.items()}
    assert counts == {
        "mnist": {"mnist d3": 128, "mnist d8": 128},
        "modular": {"modular x+y": 576, "modular x*y": 128, "modular x/y": 128},
        "teacher_student": {"teacher_student": 576},
    }
    every = [c for cs in cfgs.values() for c in cs]
    assert len({c.run_id for c in every}) == 1664
    assert all(isinstance(v, float) for c in every for v in (c.optim.lr, c.optim.weight_decay, c.model.init_scale))
    assert all(Counter(c.seed for c in cs) == {s: len(cs) // 4 for s in range(4)} for cs in cfgs.values())
    assert {c.steps for c in cfgs["teacher_student"]} == {300_000}
    assert {c.task.params["n_test"] for c in cfgs["teacher_student"]} == {1000}
    d8 = [c for c in cfgs["mnist"] if c.model.depth == 8]
    assert {c.task.params["n_train"] for c in d8} == {4000}
    # depth 8, SGD family: lr and gamma per alpha
    assert {(c.optim.name, c.model.init_scale, c.optim.lr) for c in d8 if c.optim.name.startswith("sgd")} == {
        ("sgd", 2.0, 0.1), ("sgd", 4.0, 1e-4), ("sgd_momentum", 2.0, 0.1), ("sgd_momentum", 4.0, 1e-4)}
    assert sorted({c.optim.weight_decay for c in d8 if c.optim.name == "sgd" and c.model.init_scale == 4.0}) == \
        [0.0, 0.01, 0.03, 0.1]
    assert not any(c.task.params.get("op") == "x^2+xy+y^2" for c in cfgs["modular"])


def test_corpus_order_puts_calibration_seeds_first():
    order = corpus.corpus_order(expand(corpus.sweep_file(PROJECT, "mnist")))
    seeds = [c.seed for c in order]
    assert set(seeds[:128]) == {0, 1} and set(seeds[128:]) == {2, 3}
    assert order[0].model.depth == 8  # the slow depth-8 runs start first


def test_train_status_audit_and_drop_failed(tmp_path):
    from scripts.sweep import sweep

    cfgs = expand({"blocks": [{"base": BASE, "product": {"seed": [0, 1, 2, 3]}}]})
    remote = PrefixRemote(LocalRemote(tmp_path / "remote"), corpus.remote_prefix("mnist"))
    st = corpus.manifest_status(remote, cfgs)
    assert st["expected"] == 4 and st["uploaded"] == 0 and st["missing"] == 4

    sweep(cfgs[:3], tmp_path / "runs", "cpu", 2, remote, log=lambda *_: None)
    st = corpus.manifest_status(remote, cfgs)
    assert st["uploaded"] == 3 and st["missing"] == 1 and st["replay_level"] == {"L0": 3}

    rows = corpus.audit(remote, cfgs, tmp_path / "audit", "cpu", 2, frac=0.5, log=lambda *_: None)
    assert len(rows) == 2 and {r["download"] for r in rows} == {"ok"} and {r["replay_level"] for r in rows} == {"L0"}
    assert all(r["keyframes_checked"] >= 2 for r in rows)  # a full replay, not one interval
    assert not any((tmp_path / "audit").rglob("DONE"))  # local copies removed
    again = corpus.audit(remote, cfgs, tmp_path / "audit", "cpu", 2, frac=0.5, log=lambda *_: None)
    assert [r["run_id"] for r in again] == [r["run_id"] for r in rows]  # same sample every time

    bad = rows[0]["run_id"]
    manifest = read_manifest(remote)
    manifest[bad]["replay_level"] = "failed"
    import json
    p = tmp_path / "m.json"
    p.write_text(json.dumps(manifest))
    remote.commit({"manifest.json": p}, "mark failed")
    assert corpus.manifest_status(remote, cfgs)["replay_failed"] == [bad]
    corpus.drop_failed(remote, [bad], log=lambda *_: None)
    st = corpus.manifest_status(remote, cfgs)
    assert st["uploaded"] == 2 and st["replay_failed"] == []


def test_plot_file_names_are_safe():
    from tfs.evaluation.plots import slug

    assert slug("modular x/y") == "modular_xdivy"
    assert slug("modular x^2+xy+y^2") == "modular_x2plusxyplusy2"
    assert slug("mnist d8") == "mnist_d8"

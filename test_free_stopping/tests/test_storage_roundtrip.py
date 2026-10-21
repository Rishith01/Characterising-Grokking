import json

import numpy as np
import pytest

from tfs.storage import layout
from tfs.storage.reader import IncompleteRun, RunReader
from tfs.storage.writer import RunWriter


def test_run_directory_contents(dummy_run, dummy_cfg):
    r = RunReader(dummy_run)
    assert r.status == "complete"
    assert r.config == dummy_cfg
    assert r.verify_checksums() == []
    for name in layout.IMMUTABLE_FILES + (layout.CHECKSUMS, layout.DONE):
        assert (dummy_run / name).exists(), name


def test_streams_shapes(dummy_run, dummy_cfg):
    r = RunReader(dummy_run)
    steps = r.log_steps
    assert steps[0] == 0 and steps[-1] == dummy_cfg.steps
    assert np.all(np.diff(steps) == dummy_cfg.record.log_every)
    preds = r.train_preds()
    assert preds.shape == (len(steps), dummy_cfg.task.params["n_train"]) and preds.dtype == np.int16
    assert list(r.fingerprints()["step"]) == list(steps)
    assert r.keyframe_steps.tolist() == [0, 50, 100, 150, 200]
    assert r.summary_steps() == [0, 50, 100, 150, 200]
    s = r.summaries(100)
    assert {"sv/l0", "sv/l2", "actspec/l0", "actspec/l1", "gnoise/l0"} <= set(s)
    assert not any(k.startswith("gram/") for k in s)  # O(width^2) matrices come from replay
    assert all(v.dtype == np.float32 for v in s.values())


def test_update_columns_nan_only_at_step0(dummy_run):
    tel = RunReader(dummy_run).telemetry()
    assert tel.loc[0, "l0_update_norm"] != tel.loc[0, "l0_update_norm"]  # NaN
    assert tel["l0_update_norm"].iloc[1:].notna().all()


def test_keyframe_hashes_match_contents(dummy_run):
    r = RunReader(dummy_run)
    hashes = r.keyframe_hashes()
    for k in r.keyframe_steps:
        assert r.keyframe(int(k)).hashes() == hashes[int(k)]


def test_checksum_detects_corruption(dummy_run, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(dummy_run, copy)
    with open(copy / layout.TELEMETRY, "ab") as f:
        f.write(b"x")
    assert RunReader(copy).verify_checksums() == [layout.TELEMETRY]


def test_incomplete_run_rejected(dummy_cfg, tmp_path):
    RunWriter(tmp_path / "r", dummy_cfg)  # never finalized
    with pytest.raises(IncompleteRun):
        RunReader(tmp_path / "r")


def test_meta_records_environment(dummy_run):
    meta = json.loads((dummy_run / layout.META).read_text())
    assert meta["env"]["device"] == "cpu"
    assert meta["env"]["flags"]["deterministic_algorithms"] is True
    assert meta["n_params"] > 0 and "code" in meta


def test_divergence_detected_at_a_log_step(dummy_cfg, tmp_path):
    from dataclasses import replace

    from tfs.config import OptimConfig

    from .conftest import train

    cfg = replace(dummy_cfg, optim=OptimConfig("sgd", lr=1e6))
    r = RunReader(train(cfg, tmp_path))
    assert r.status == "diverged"
    # detection happens on the update that leads into a log step
    assert (r.meta["diverged_at"] + 1) % cfg.record.log_every == 0

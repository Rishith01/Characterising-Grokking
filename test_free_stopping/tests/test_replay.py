import json
import shutil
from dataclasses import replace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tfs.storage import layout
from tfs.training.determinism import EnvRecord
from tfs.training.loop import TrainLoop
from tfs.causal.view import TrainContext
from tfs.training.replay import CorruptRun, EnvironmentMismatch, ReplayDivergence, ReplayEngine
from tfs.training.state import tensor_sha256

from .conftest import OPTIM, make_cfg, train


def _rewrite_keyframe(src, dst, key, fn):
    """Copy a run and change one keyframe tensor *consistently*: the tensor, its entry in
    keyframe_hashes.json and checksums.json all agree, as if the run had been recorded
    that way. Replay must then catch the mismatch with what training actually does."""
    shutil.copytree(src, dst)
    kf = load_file(dst / layout.KEYFRAMES)
    kf[key] = fn(kf[key])
    save_file(kf, dst / layout.KEYFRAMES)
    hashes = json.loads((dst / layout.KEYFRAME_HASHES).read_text())
    step = key.split("/", 1)[0][1:]
    hashes[step][key] = tensor_sha256(kf[key])
    (dst / layout.KEYFRAME_HASHES).write_text(json.dumps(hashes, indent=2, sort_keys=True))
    sums = json.loads((dst / layout.CHECKSUMS).read_text())
    for name in (layout.KEYFRAMES, layout.KEYFRAME_HASHES):
        sums[name] = layout.file_sha256(dst / name)
    (dst / layout.CHECKSUMS).write_text(json.dumps(sums, indent=2, sort_keys=True))
    return dst


@pytest.mark.parametrize("optim", list(OPTIM))
def test_full_replay_is_L0(dummy_cfg, tmp_path, optim):
    run = train(make_cfg(dummy_cfg, optim), tmp_path)
    report = ReplayEngine(run).verify("all")
    assert report.level == "L0", report.first_failure
    assert report.keyframes_checked == [50, 100, 150, 200]
    assert len(report.steps_checked) == 21


def test_every_keyframe_interval_is_L0(dummy_run):
    eng = ReplayEngine(dummy_run)
    keys = [int(k) for k in eng.reader.keyframe_steps]
    report = eng.verify(list(zip(keys[:-1], keys[1:])))
    assert report.level == "L0", report.first_failure
    assert report.keyframes_checked == keys[1:]


def test_iterate_yields_requested_steps_and_matches_keyframes(dummy_run):
    eng = ReplayEngine(dummy_run)
    got = []
    for step, model in eng.iterate([73, 120, 150]):
        got.append(step)
        if step == 150:
            stored = eng.reader.keyframe(150).params
            for name, p in model.param_list():
                assert torch.equal(p.detach().cpu(), stored[name])
    assert got == [73, 120, 150]


def test_replay_never_opens_a_later_keyframe(dummy_run):
    eng = ReplayEngine(dummy_run)
    for _ in eng.iterate([60, 90], verify=False):
        pass
    assert eng.reader.opened_keyframes == [50]


def test_recording_is_side_effect_free(dummy_cfg):
    """Training with all the per-step extras on gives the same trajectory as with none."""
    ctx = TrainContext.from_config(dummy_cfg)
    finals = []
    for want in (lambda t: True, lambda t: False):
        loop = TrainLoop(dummy_cfg, ctx.x_train, ctx.y_train, ctx.kind, ctx.d_in, ctx.d_out, "cpu")
        loop.advance(0, 120, want_info=want)
        finals.append(loop.capture(120).hashes())
    assert finals[0] == finals[1]


def test_perturbed_keyframe_fails(dummy_run, tmp_path):
    run = _rewrite_keyframe(dummy_run, tmp_path / "run", "k100/p/l0.weight", lambda w: w + 1e-3)
    report = ReplayEngine(run).verify([(50, 100)])
    assert report.level == "failed"
    assert "keyframe 100" in report.first_failure


def test_small_perturbation_is_L1(dummy_run, tmp_path):
    """A keyframe off by far less than the tolerance degrades L0 -> L1, not to failed."""
    run = _rewrite_keyframe(dummy_run, tmp_path / "run", "k100/p/l0.weight",
                            lambda w: w + 1e-9 * w.norm() / w.numel() ** 0.5)
    report = ReplayEngine(run).verify([(50, 100)])
    assert report.level == "L1", report.first_failure


def test_divergence_raises_during_iterate(dummy_run, tmp_path):
    run = _rewrite_keyframe(dummy_run, tmp_path / "run", "k50/p/l1.weight", lambda w: w * 1.01)
    with pytest.raises(ReplayDivergence):
        for _ in ReplayEngine(run).iterate([60]):
            pass


def test_corrupted_files_refused(dummy_run, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(dummy_run, run)
    kf = load_file(run / layout.KEYFRAMES)
    kf["k100/p/l0.weight"] = kf["k100/p/l0.weight"] + 1e-3
    save_file(kf, run / layout.KEYFRAMES)  # bytes no longer match checksums.json
    with pytest.raises(CorruptRun):
        ReplayEngine(run)


def test_environment_mismatch_refused(dummy_run, monkeypatch):
    real = EnvRecord.capture

    def fake(device):
        return replace(real(device), torch="0.0.0-other")

    monkeypatch.setattr(EnvRecord, "capture", staticmethod(fake))
    with pytest.raises(EnvironmentMismatch):
        ReplayEngine(dummy_run)
    assert ReplayEngine(dummy_run, allow_env_mismatch=True).env_mismatches == ["torch"]

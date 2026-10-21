import json
import shutil

import pytest

from tfs.storage import layout
from tfs.storage.hub import (LocalRemote, PrefixRemote, delete_runs, pull_run, push_runs, read_manifest,
                             update_replay_level, verify_remote)
from tfs.storage.reader import RunReader
from tfs.training.replay import ReplayEngine


def _copy(run, tmp_path):
    reader = RunReader(run)
    dest = layout.run_dir(tmp_path / "runs", reader.meta["app"], reader.meta["run_id"])
    shutil.copytree(run, dest)
    return dest


def test_push_delete_pull_cycle(dummy_run, tmp_path):
    """G0 cycle against a local remote: push, verify, delete, pull, checksums, replay."""
    local = _copy(dummy_run, tmp_path)
    meta = RunReader(local).meta
    remote = LocalRemote(tmp_path / "remote")

    res = push_runs(remote, [local], delete_local=True, log=lambda *_: None)
    assert res["pushed"] == [meta["run_id"]] and not res["verify_failed"]
    assert not local.exists()
    entry = read_manifest(remote)[meta["run_id"]]
    assert entry["status"] == "complete" and entry["replay_level"] == "unverified"

    back = pull_run(remote, meta["app"], meta["run_id"], tmp_path / "pulled")
    assert RunReader(back).verify_checksums() == []
    assert ReplayEngine(back).verify("all").level == "L0"


def test_partial_pull(dummy_run, tmp_path):
    local = _copy(dummy_run, tmp_path)
    meta = RunReader(local).meta
    remote = LocalRemote(tmp_path / "remote")
    push_runs(remote, [local], log=lambda *_: None)
    back = pull_run(remote, meta["app"], meta["run_id"], tmp_path / "pulled", parts=["telemetry"])
    assert (back / layout.TELEMETRY).exists() and not (back / layout.KEYFRAMES).exists()
    assert RunReader(back).verify_checksums() == []


def test_verify_remote_detects_bad_copy(dummy_run, tmp_path):
    local = _copy(dummy_run, tmp_path)
    meta = RunReader(local).meta
    remote = LocalRemote(tmp_path / "remote")
    push_runs(remote, [local], delete_local=False, log=lambda *_: None)
    target = tmp_path / "remote" / layout.remote_prefix(meta["app"], meta["run_id"]) / layout.TELEMETRY
    target.write_bytes(target.read_bytes() + b"x")
    assert verify_remote(remote, local) == [f"{layout.remote_prefix(meta['app'], meta['run_id'])}/{layout.TELEMETRY}"]


def test_replay_level_reaches_manifest(dummy_run, tmp_path):
    local = _copy(dummy_run, tmp_path)
    remote = LocalRemote(tmp_path / "remote")
    eng = ReplayEngine(local)
    eng.write_report(eng.verify("all"))
    update_replay_level(remote, local)
    run_id = RunReader(local).meta["run_id"]
    assert read_manifest(remote)[run_id]["replay_level"] == "L0"
    # the replay report is mutable and stays outside the checksummed set
    assert layout.REPLAY_REPORT not in json.loads((local / layout.CHECKSUMS).read_text())


class _NoHashRemote(LocalRemote):
    """Like an HF repo whose Xet-stored files expose no content hash."""

    def list_files(self, prefix):
        from dataclasses import replace

        return {k: replace(v, sha256=None, git_sha1=None) for k, v in super().list_files(prefix).items()}


def test_verification_falls_back_to_download(dummy_run, tmp_path):
    local = _copy(dummy_run, tmp_path)
    meta = RunReader(local).meta
    remote = _NoHashRemote(tmp_path / "remote")
    assert push_runs(remote, [local], delete_local=False, log=lambda *_: None)["verify_failed"] == []
    target = tmp_path / "remote" / layout.remote_prefix(meta["app"], meta["run_id"]) / layout.META
    data = bytearray(target.read_bytes())
    data[-2] ^= 1  # same size, different content
    target.write_bytes(bytes(data))
    assert verify_remote(remote, local) == [f"{layout.remote_prefix(meta['app'], meta['run_id'])}/{layout.META}"]


def test_delete_runs_removes_folder_and_manifest_entry(dummy_run, tmp_path):
    local = _copy(dummy_run, tmp_path)
    meta = RunReader(local).meta
    remote = LocalRemote(tmp_path / "remote")
    push_runs(remote, [local], delete_local=False, log=lambda *_: None)
    (tmp_path / "remote" / "phase1").mkdir()
    (tmp_path / "remote" / "phase1" / "report.json").write_text("{}")  # reports must survive

    with pytest.raises(KeyError):  # unknown ids fail before anything is deleted
        delete_runs(remote, [meta["run_id"], "nope"], log=lambda *_: None)
    assert remote.list_files(layout.remote_prefix(meta["app"], meta["run_id"]))

    delete_runs(remote, [meta["run_id"]], log=lambda *_: None)
    assert remote.list_files(layout.remote_prefix(meta["app"], meta["run_id"])) == {}
    assert meta["run_id"] not in read_manifest(remote)
    assert (tmp_path / "remote" / "phase1" / "report.json").exists()


def test_prefix_remote_keeps_phases_apart(dummy_run, tmp_path):
    """One repo, one sub-folder per phase: each has its own runs/ and manifest."""
    local = _copy(dummy_run, tmp_path)
    meta = RunReader(local).meta
    base = LocalRemote(tmp_path / "remote")
    p2 = PrefixRemote(base, "phase2/")
    res = push_runs(p2, [local], delete_local=True, log=lambda *_: None)
    assert res["pushed"] == [meta["run_id"]] and not res["verify_failed"]
    assert (tmp_path / "remote" / "phase2" / layout.MANIFEST).exists()
    assert read_manifest(base) == {} and read_manifest(PrefixRemote(base, "phase3")) == {}

    back = pull_run(p2, meta["app"], meta["run_id"], tmp_path / "pulled")
    assert RunReader(back).verify_checksums() == []
    delete_runs(p2, [meta["run_id"]], log=lambda *_: None)
    assert not (tmp_path / "remote" / "phase2" / "runs" / meta["app"] / meta["run_id"]).exists()

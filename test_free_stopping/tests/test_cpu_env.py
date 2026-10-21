import os
from dataclasses import replace

from tfs import cpu_env
from tfs.sweep.launcher import child_env
from tfs.training.determinism import EnvRecord


def test_child_env_pins_cpu_numerics(monkeypatch):
    monkeypatch.delenv("TFS_CPU_UNPINNED", raising=False)
    env = child_env("cpu", workers=4)
    for k, v in cpu_env.PINNED.items():
        assert env[k] == v
    assert int(env["TFS_NUM_THREADS"]) == max(1, cpu_env.available_cpus() // 4)
    # GPU workers are not given a CPU thread share
    assert child_env("cuda", workers=4).get("TFS_NUM_THREADS") == os.environ.get("TFS_NUM_THREADS")


def test_unpinned_switch_restores_phase1_env(monkeypatch):
    monkeypatch.setenv("TFS_CPU_UNPINNED", "1")
    for k in cpu_env.PINNED:
        monkeypatch.delenv(k, raising=False)
    env = child_env("cpu", workers=4)
    assert all(k not in env for k in cpu_env.PINNED)


def test_cpu_replay_checks_cpu_model_and_pinning():
    here = EnvRecord.capture("cpu")
    other_cpu = replace(here, flags={**here.flags, "cpu": {**here.flags["cpu"], "cpu_model": "some other CPU"}})
    assert "flags.cpu.cpu_model" in here.mismatches(other_cpu)
    unpinned = replace(here, flags={**here.flags, "cpu": {**here.flags["cpu"], "MKL_DYNAMIC": None}})
    assert "flags.cpu.MKL_DYNAMIC" in here.mismatches(unpinned)
    assert here.mismatches(here) == []

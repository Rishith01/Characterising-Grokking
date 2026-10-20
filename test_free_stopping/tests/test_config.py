from dataclasses import replace

import pytest
import torch

from tfs.config import RecordConfig, RunConfig
from tfs.seeding import SeedStreams, batch_indices
from tfs.training.schedule import Schedule


def test_dict_roundtrip(dummy_cfg, tmp_path):
    assert RunConfig.from_dict(dummy_cfg.to_dict()) == dummy_cfg
    dummy_cfg.to_yaml(tmp_path / "c.yaml")
    assert RunConfig.from_yaml(tmp_path / "c.yaml") == dummy_cfg


def test_ids(dummy_cfg):
    other = dummy_cfg.with_seed(1)
    assert dummy_cfg.run_id == RunConfig.from_dict(dummy_cfg.to_dict()).run_id
    assert dummy_cfg.run_id != other.run_id
    assert dummy_cfg.group_id == other.group_id


def test_schedules_must_nest(dummy_cfg):
    with pytest.raises(ValueError):
        replace(dummy_cfg, record=RecordConfig(log_every=10, summary_every=25, keyframe_every=50))


def test_schedule_contains_endpoints(dummy_cfg):
    s = Schedule.build(203, dummy_cfg.record)
    for steps in (s.log_steps, s.summary_steps, s.keyframe_steps):
        assert steps[0] == 0 and steps[-1] == 203
    assert set(s.keyframe_steps) <= set(s.summary_steps) <= set(s.log_steps)


def test_seed_streams_independent():
    a, b = SeedStreams(0), SeedStreams(0)
    assert a.int_seed("init") == b.int_seed("init")
    assert a.int_seed("init") != a.int_seed("order")
    assert a.int_seed("init") != SeedStreams(1).int_seed("init")


def test_batch_indices_cover_each_epoch():
    n, bs = 100, 32
    epoch0 = torch.cat([batch_indices(7, t, n, bs) for t in range(4)])
    assert sorted(epoch0.tolist()) == list(range(n))
    assert torch.equal(batch_indices(7, 5, n, bs), batch_indices(7, 5, n, bs))
    assert not torch.equal(batch_indices(7, 0, n, bs), batch_indices(7, 4, n, bs))

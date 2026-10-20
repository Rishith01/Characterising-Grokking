import json

import pandas as pd
import pytest

from tfs.storage import layout

from .conftest import OPTIM, make_cfg, train


@pytest.mark.parametrize("optim", list(OPTIM))
def test_two_trainings_are_bit_identical(dummy_cfg, tmp_path, optim):
    cfg = make_cfg(dummy_cfg, optim)
    a = train(cfg, tmp_path / "a")
    b = train(cfg, tmp_path / "b")
    assert json.loads((a / layout.KEYFRAME_HASHES).read_text()) == json.loads((b / layout.KEYFRAME_HASHES).read_text())
    pd.testing.assert_frame_equal(pd.read_parquet(a / layout.TELEMETRY), pd.read_parquet(b / layout.TELEMETRY))
    # every data file except meta.json (timestamps, wall time) is byte-identical
    ca = json.loads((a / layout.CHECKSUMS).read_text())
    cb = json.loads((b / layout.CHECKSUMS).read_text())
    ca.pop(layout.META), cb.pop(layout.META)
    assert ca == cb


def test_full_batch_path(dummy_cfg, tmp_path):
    cfg = make_cfg(dummy_cfg, "adamw", batch_size=None)
    a = train(cfg, tmp_path / "a")
    b = train(cfg, tmp_path / "b")
    assert (a / layout.KEYFRAME_HASHES).read_text() == (b / layout.KEYFRAME_HASHES).read_text()

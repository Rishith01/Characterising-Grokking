import gzip
import struct
from dataclasses import replace

import numpy as np
import pytest
import torch

from tfs.config import RecordConfig, RunConfig
from tfs.seeding import SeedStreams
from tfs.tasks import get_task
from tfs.tasks.modular import modular_pairs
from tfs.storage.reader import RunReader

from .conftest import train


@pytest.mark.parametrize("op,check", [
    ("x+y", lambda a, b, p: (a + b) % p),
    ("x-y", lambda a, b, p: (a - b) % p),
    ("x*y", lambda a, b, p: (a * b) % p),
    ("x^2+xy+y^2", lambda a, b, p: (a * a + a * b + b * b) % p),
])
def test_modular_labels(op, check):
    a, b, y = modular_pairs(op, 7)
    assert len(y) == 49 and np.array_equal(y, check(a, b, 7))


def test_modular_division_inverts_multiplication():
    p = 11
    a, b, y = modular_pairs("x/y", p)
    assert len(y) == p * (p - 1) and b.min() == 1
    assert np.array_equal((y * b) % p, a)


def test_modular_split_and_encoding():
    task = get_task("modular")
    d = task.build({"op": "x+y", "p": 7, "train_frac": 0.5}, SeedStreams(0))
    assert d.x_train.shape == (24, 14) and d.x_test.shape == (25, 14)
    assert torch.all(d.x_train.sum(1) == 2)
    a, b = d.x_train[:, :7].argmax(1), d.x_train[:, 7:].argmax(1)
    assert torch.equal((a + b) % 7, d.y_train)
    d2 = task.build({"op": "x+y", "p": 7, "train_frac": 0.5}, SeedStreams(0))
    assert torch.equal(d.x_train, d2.x_train)


def test_teacher_student_is_seeded_and_accuracy_thresholded():
    task = get_task("teacher_student")
    d = task.build({"n_train": 50, "n_test": 40}, SeedStreams(3))
    d2 = get_task("teacher_student").build({"n_train": 50, "n_test": 40}, SeedStreams(3))
    assert torch.equal(d.y_train, d2.y_train) and d.kind == "regression"
    assert task.accuracy(d.y_test, d.y_test) == 1.0
    assert task.accuracy(d.y_test + 1.0, d.y_test) == 0.0
    rec = task.train_pred_record(d.y_train + 0.05, d.y_train)
    assert rec.dtype == np.float16 and np.allclose(rec, 0.0025, rtol=1e-2)


def _write_idx(path, arr):
    header = struct.pack(">BBBB", 0, 0, 8, arr.ndim) + b"".join(struct.pack(">I", s) for s in arr.shape)
    with gzip.open(path, "wb") as f:
        f.write(header + arr.astype(np.uint8).tobytes())


def test_mnist_reader_on_synthetic_idx(tmp_path, monkeypatch):
    """The IDX reader and subset logic, on fake files (no download in tests)."""
    import tfs.tasks.mnist as m

    rng = np.random.default_rng(0)
    imgs = {"train": rng.integers(0, 256, (60, 28, 28)), "t10k": rng.integers(0, 256, (20, 28, 28))}
    labels = {"train": rng.integers(0, 10, 60), "t10k": rng.integers(0, 10, 20)}
    for split in ("train", "t10k"):
        _write_idx(tmp_path / f"{split}-images-idx3-ubyte.gz", imgs[split])
        _write_idx(tmp_path / f"{split}-labels-idx1-ubyte.gz", labels[split])
    monkeypatch.setattr(m, "ensure_mnist", lambda root=None, download=True: tmp_path)
    m.load_mnist.cache_clear()
    d = get_task("mnist").build({"n_train": 16}, SeedStreams(0))
    assert d.x_train.shape == (16, 784) and d.x_test.shape == (20, 784)
    assert float(d.x_train.max()) <= 1.0
    flat = imgs["train"].reshape(60, -1) / 255.0
    rows = [int(np.where(np.all(np.isclose(flat, x.numpy()), axis=1))[0][0]) for x in d.x_train]
    assert np.array_equal(labels["train"][rows], d.y_train.numpy())
    m.load_mnist.cache_clear()


def test_recording_cadence_never_changes_trajectory(dummy_cfg, tmp_path):
    """Summaries (incl. gradient-noise probes) and fingerprints at a different cadence
    give bit-identical keyframes."""
    a = train(dummy_cfg, tmp_path / "a")
    dense = replace(dummy_cfg, record=RecordConfig(log_every=5, summary_every=10, keyframe_every=50))
    b = train(dense, tmp_path / "b")
    ha, hb = RunReader(a).keyframe_hashes(), RunReader(b).keyframe_hashes()
    assert ha == hb


def test_gradient_noise_summaries(dummy_run):
    s = RunReader(dummy_run).summaries(100)
    g = s["gnoise/l0"]
    assert g.shape == (3,) and g[1] >= -1e-12 and np.isclose(g[0] + g[1], g[2])


@pytest.mark.parametrize("loss,task,params", [
    ("ce", "modular", {"op": "x+y", "p": 7, "train_frac": 0.6}),
    ("mse", "teacher_student", {"n_train": 30, "n_test": 20}),
])
def test_tasks_train_and_replay_L0(dummy_cfg, tmp_path, loss, task, params):
    from tfs.config import TaskConfig
    from tfs.training.replay import ReplayEngine

    cfg = replace(dummy_cfg, task=TaskConfig(task, params), loss=loss, batch_size=None, steps=100)
    run = train(cfg, tmp_path)
    assert ReplayEngine(run).verify("all").level == "L0"

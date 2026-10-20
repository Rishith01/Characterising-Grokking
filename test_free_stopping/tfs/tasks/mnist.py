"""MNIST subset (Omnigrok §4, Deep Grokking; PLAN.md §2).

A seeded subset of `n_train` images from the 60k training set; the test set is the full
10k. Pixels are scaled to [0, 1] and flattened to 784 dims. 10 classes.

Data comes from the raw IDX files, downloaded once into $TFS_DATA_DIR/mnist (default
~/.cache/tfs/mnist) from torchvision's official mirror and verified by MD5. No
torchvision dependency.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..seeding import SeedStreams
from .base import TASKS, ClassificationTask, TaskData

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {  # name -> md5 of the .gz file (same values torchvision checks)
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def data_dir() -> Path:
    return Path(os.environ.get("TFS_DATA_DIR", Path.home() / ".cache" / "tfs")) / "mnist"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def ensure_mnist(root: Path | None = None, download: bool = True) -> Path:
    root = Path(root) if root else data_dir()
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        path = root / name
        if path.exists() and _md5(path) == md5:
            continue
        if not download:
            raise FileNotFoundError(f"{path} missing or corrupt and download=False")
        urllib.request.urlretrieve(MIRROR + name, path)
        if _md5(path) != md5:
            path.unlink()
            raise IOError(f"MD5 mismatch for downloaded {name}")
    return root


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


@lru_cache(maxsize=2)
def load_mnist(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = Path(root)
    return (_read_idx(r / "train-images-idx3-ubyte.gz"), _read_idx(r / "train-labels-idx1-ubyte.gz"),
            _read_idx(r / "t10k-images-idx3-ubyte.gz"), _read_idx(r / "t10k-labels-idx1-ubyte.gz"))


@TASKS.register("mnist")
class MNISTSubset(ClassificationTask):
    name = "mnist"

    def build(self, params: dict, streams: SeedStreams) -> TaskData:
        root = ensure_mnist(params.get("root"), download=params.get("download", True))
        x_tr, y_tr, x_te, y_te = load_mnist(str(root))
        n_train = int(params["n_train"])
        idx = np.sort(streams.numpy("data").choice(len(y_tr), size=n_train, replace=False))

        def prep(x, y):
            x = torch.from_numpy(x.reshape(len(x), -1).astype(np.float32) / 255.0)
            return x, torch.from_numpy(y.astype(np.int64))

        xa, ya = prep(x_tr[idx], y_tr[idx])
        xb, yb = prep(x_te, y_te)
        return TaskData(xa, ya, xb, yb, 784, 10, "classification", 10)

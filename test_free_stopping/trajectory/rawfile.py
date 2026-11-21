"""One raw file per run and stage: named numpy arrays plus a JSON metadata block, in a
single .npz written atomically (a crash never leaves a half-written file)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

_META = "__meta__"


def save_raw(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _META in arrays:
        raise ValueError(f"array name {_META!r} is reserved")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **{k: np.asarray(v) for k, v in arrays.items()},
                            **{_META: np.array(json.dumps(meta, default=str))})
    tmp.replace(path)
    return path


def load_raw(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z[_META])) if _META in z.files else {}
        arrays = {k: z[k] for k in z.files if k != _META}
    return arrays, meta

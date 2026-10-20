"""RunWriter: buffers a run's records in memory and writes the run directory at the end
(DESIGN.md §5). Used only by the Trainer.

Runs are small enough (a few hundred MB at most, dominated by ~11 keyframes) that
buffering is cheap, and writing once means a crashed run never leaves a half-written
directory that looks complete: DONE is the last file written.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from safetensors.numpy import save_file as save_numpy
from safetensors.torch import save_file as save_torch

from . import layout
from ..config import RunConfig
from ..training.state import TrainState


class RunWriter:
    def __init__(self, run_dir: Path, cfg: RunConfig, overwrite: bool = False):
        self.run_dir = Path(run_dir)
        if (self.run_dir / layout.DONE).exists() and not overwrite:
            raise FileExistsError(f"{self.run_dir} is already a complete run")
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)  # an incomplete earlier attempt
        self.run_dir.mkdir(parents=True)
        self.cfg = cfg
        self._telemetry: list[dict] = []
        self._preds: list[np.ndarray] = []
        self._fingerprints: list[dict] = []
        self._oracle: list[dict] = []
        self._summaries: dict[str, np.ndarray] = {}
        self._keyframes: dict[str, torch.Tensor] = {}
        self._keyframe_hashes: dict[str, dict[str, str]] = {}

    def log(self, telemetry: dict, train_preds: np.ndarray, fingerprint: dict, oracle: dict) -> None:
        self._telemetry.append(telemetry)
        self._preds.append(train_preds)
        self._fingerprints.append(fingerprint)
        self._oracle.append(oracle)

    def summary(self, step: int, summaries: dict[str, np.ndarray]) -> None:
        for name, arr in summaries.items():
            self._summaries[f"s{step}/{name}"] = np.ascontiguousarray(arr)

    def keyframe(self, state: TrainState) -> None:
        self._keyframes.update(state.to_flat())
        self._keyframe_hashes[str(state.step)] = state.hashes()

    def finalize(self, status: str, meta_extra: Optional[dict] = None) -> Path:
        d = self.run_dir
        pd.DataFrame(self._telemetry).to_parquet(d / layout.TELEMETRY, index=False)
        np.save(d / layout.TRAIN_PREDS, np.stack(self._preds))
        pd.DataFrame(self._fingerprints).to_parquet(d / layout.FINGERPRINTS, index=False)
        pd.DataFrame(self._oracle).to_parquet(d / layout.ORACLE, index=False)
        save_numpy(self._summaries, str(d / layout.SUMMARIES))
        save_torch(self._keyframes, str(d / layout.KEYFRAMES))
        _write_json(d / layout.KEYFRAME_HASHES, self._keyframe_hashes)

        meta = {
            "run_id": self.cfg.run_id,
            "group_id": self.cfg.group_id,
            "app": self.cfg.app,
            "status": status,
            "config": self.cfg.to_dict(),
            "keyframe_steps": sorted(int(k) for k in self._keyframe_hashes),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **(meta_extra or {}),
        }
        _write_json(d / layout.META, meta)

        checksums = {name: layout.file_sha256(d / name) for name in layout.IMMUTABLE_FILES}
        _write_json(d / layout.CHECKSUMS, checksums)
        (d / layout.DONE).write_text(status + "\n")
        return d


def _write_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

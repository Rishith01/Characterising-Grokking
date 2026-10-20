"""RunReader: lazy, read-only access to a run directory (DESIGN.md §5).

Deliberately has no accessor for the oracle (test) file. See tfs/evaluation/oracle.py
and DESIGN.md §10.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

from . import layout
from ..config import RunConfig
from ..training.determinism import EnvRecord
from ..training.state import TrainState


class IncompleteRun(RuntimeError):
    pass


class RunReader:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        if not (self.run_dir / layout.DONE).exists():
            raise IncompleteRun(f"{self.run_dir} has no {layout.DONE} marker")
        # Every keyframe step opened through this reader, in order. Lets tests assert
        # that replay to step t never touches a keyframe after t.
        self.opened_keyframes: list[int] = []

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.run_dir / layout.META).read_text())

    @property
    def config(self) -> RunConfig:
        return RunConfig.from_dict(self.meta["config"])

    @property
    def status(self) -> str:
        return self.meta["status"]

    @property
    def env(self) -> EnvRecord:
        return EnvRecord.from_dict(self.meta["env"])

    @property
    def keyframe_steps(self) -> np.ndarray:
        return np.array(self.meta["keyframe_steps"], dtype=np.int64)

    @property
    def log_steps(self) -> np.ndarray:
        return self.telemetry()["step"].to_numpy()

    # ---- streams -------------------------------------------------------------------
    def telemetry(self) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir / layout.TELEMETRY)

    def train_preds(self) -> np.ndarray:
        return np.load(self.run_dir / layout.TRAIN_PREDS)

    def fingerprints(self) -> pd.DataFrame:
        """For replay verification only; never exposed to signals."""
        return pd.read_parquet(self.run_dir / layout.FINGERPRINTS)

    def summary_steps(self) -> list[int]:
        with safe_open(self.run_dir / layout.SUMMARIES, framework="np") as f:
            return sorted({int(k.split("/", 1)[0][1:]) for k in f.keys()})

    def summaries(self, step: int) -> dict[str, np.ndarray]:
        prefix = f"s{step}/"
        with safe_open(self.run_dir / layout.SUMMARIES, framework="np") as f:
            return {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}

    def keyframe(self, step: int) -> TrainState:
        if step not in set(self.keyframe_steps.tolist()):
            raise KeyError(f"no keyframe at step {step}; have {self.keyframe_steps.tolist()}")
        self.opened_keyframes.append(int(step))
        prefix = f"k{step}/"
        with safe_open(self.run_dir / layout.KEYFRAMES, framework="pt") as f:
            flat = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        return TrainState.from_flat(step, flat)

    def keyframe_hashes(self) -> dict[int, dict[str, str]]:
        raw = json.loads((self.run_dir / layout.KEYFRAME_HASHES).read_text())
        return {int(k): v for k, v in raw.items()}

    # ---- integrity -----------------------------------------------------------------
    def verify_checksums(self) -> list[str]:
        """Names of files whose sha256 does not match checksums.json (empty = intact).
        Files that are absent (partial download) are skipped."""
        expected = json.loads((self.run_dir / layout.CHECKSUMS).read_text())
        bad = []
        for name, sha in expected.items():
            path = self.run_dir / name
            if path.exists() and layout.file_sha256(path) != sha:
                bad.append(name)
        return bad

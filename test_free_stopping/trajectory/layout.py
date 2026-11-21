"""Where Phase 5 outputs live: one source of truth for local and HF paths.

HF (repo advay333/tfs), under OUT_PREFIX:
    raw/recorded/<run_id>.npz    Stage E, recorded data (no replay)
    raw/replay/<run_id>.npz      Stage E, replay pass
    curves.parquet               Stage Q, every metric on the common grid, all runs (long format)
    runs.parquet                 run metadata and outcome labels (trajectory.corpus)
The local work directory mirrors the same relative paths.
"""
from __future__ import annotations

from pathlib import Path

OUT_PREFIX = "phase5"
STAGES = ("recorded", "replay")


def raw_rel(stage: str, run_id: str) -> str:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    return f"raw/{stage}/{run_id}.npz"


def raw_path(work: Path, stage: str, run_id: str) -> Path:
    return Path(work) / raw_rel(stage, run_id)


CURVES = "curves.parquet"
RUNS = "runs.parquet"

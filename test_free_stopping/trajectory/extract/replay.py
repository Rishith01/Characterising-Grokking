"""Stage E from one replay pass per run (PHASE5_PLAN.md §2).

The run is retrained step by step from its first keyframe with tfs's ReplayEngine,
which (with verify=True) checks every log step and keyframe against the recording and
raises if the replay drifts out of tolerance. Every probe is called at the steps it
asked for, in one pass, so a run is replayed exactly once whatever the probes measure.
On Molab, with the pinned environment, a replay is bit-identical to the original (L0).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from tfs.training.replay import ReplayEngine, ReplayReport

from ..grid import common_grid
from .probes import Plan, Probe, default_probes

VERSION = 1


def _plan(eng: ReplayEngine) -> Plan:
    cfg = eng.reader.config
    rec = cfg.record
    return Plan(grid=common_grid(cfg.steps, rec.log_every, rec.summary_every),
                log_steps=np.asarray(eng.reader.log_steps, dtype=np.int64),
                summary_every=rec.summary_every, steps=cfg.steps, optimizer=cfg.optim.name,
                gamma=float(cfg.optim.weight_decay))


def extract_replay(run_dir: Path, probes: Optional[Sequence[Probe]] = None, device: str = "cpu",
                   verify: bool = True, allow_env_mismatch: bool = False) -> tuple[dict[str, np.ndarray], dict]:
    """(arrays, meta) for one run already on disk with its keyframes and telemetry (plus
    train predictions and fingerprints when verify is on)."""
    probes = list(probes) if probes is not None else default_probes()
    t0 = time.time()
    eng = ReplayEngine(run_dir, device, allow_env_mismatch=allow_env_mismatch)
    plan = _plan(eng)
    wanted: dict[int, list[Probe]] = {}
    for p in probes:
        p.start(eng, plan)
        for s in p.steps(plan):
            wanted.setdefault(int(s), []).append(p)
    report = ReplayReport(env_mismatches=eng.env_mismatches)
    for t, _model in eng.iterate(sorted(wanted), verify=verify, report=report):
        for p in wanted[int(t)]:
            p.at(int(t), eng)
    arrays: dict[str, np.ndarray] = {"grid": plan.grid}
    for p in probes:
        arrays.update(p.finish())
    meta = {
        "stage": "replay", "version": VERSION, "run_id": eng.reader.config.run_id,
        "probes": {p.name: p.version for p in probes},
        "replay_level": report.level if verify else "unverified", "replay_max_err": report.max_err,
        "env_mismatches": list(eng.env_mismatches), "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else None,
        "seconds": round(time.time() - t0, 1),
    }
    return arrays, meta

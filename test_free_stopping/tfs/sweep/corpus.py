"""Phase 3 corpus: one dataset per notebook, run order, manifest status and the G3 audit
(PLAN.md §6, Phase 3).

Each dataset lives in its own sub-folder of the HF repo, with its own runs/ and
manifest.json: phase3/mnist/, phase3/modular/, phase3/teacher_student/. Notebooks for
different datasets can then run in separate Molab sessions, even at the same time,
without ever writing the same manifest.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from ..config import RunConfig
from ..storage import layout
from ..storage.hub import Remote, delete_runs, pull_run, read_manifest, update_replay_level
from .launcher import replay_parallel

DATASETS = ("mnist", "modular", "teacher_student")
PHASE = "phase3"


def sweep_file(project: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; have {DATASETS}")
    return Path(project) / "configs" / "sweeps" / f"{PHASE}_{dataset}.yaml"


def remote_prefix(dataset: str) -> str:
    return f"{PHASE}/{dataset}"


def corpus_order(configs: Iterable[RunConfig]) -> list[RunConfig]:
    """Calibration seeds {0, 1} first, so a session that ends early still leaves complete
    calibration data; within that, longest runs first, so the tail of a session is not
    one slow run."""
    return sorted(configs, key=lambda c: (c.seed >= 2, -c.steps, -c.model.depth, -c.model.width, c.run_id))


def manifest_status(remote: Remote, configs: list[RunConfig]) -> dict:
    """How far the corpus is: expected vs. uploaded runs, status and replay levels."""
    manifest = read_manifest(remote)
    expected = {c.run_id for c in configs}
    present = {k: v for k, v in manifest.items() if k in expected}
    levels = Counter(v.get("replay_level") for v in present.values())
    return {
        "expected": len(expected),
        "uploaded": len(present),
        "missing": len(expected - set(present)),
        "unexpected": len(set(manifest) - expected),
        "status": dict(Counter(v.get("status") for v in present.values())),
        "replay_level": dict(levels),
        "replay_failed": sorted(k for k, v in present.items() if v.get("replay_level") in ("failed", "error")),
        "size_gb": round(sum(v.get("size_bytes", 0) for v in present.values()) / 1e9, 2),
    }


def audit(remote: Remote, configs: list[RunConfig], work: Path, device: str, workers: int,
          frac: float = 0.05, seed: int = 0, log: Callable[[str], None] = print) -> list[dict]:
    """G3 audit: re-download a random `frac` of the uploaded runs (at least one), check
    every file against checksums.json (pull_run raises on a mismatch), replay each one end
    to end, and record the full-replay level in the remote manifest.

    The sample is drawn from a fixed seed over the sorted run ids, so re-running the audit
    checks the same runs."""
    manifest = read_manifest(remote)
    ids = sorted(c.run_id for c in configs if c.run_id in manifest)
    if not ids:
        return []
    k = max(1, round(frac * len(ids)))
    sample = sorted(random.Random(seed).sample(ids, k))
    work = Path(work)
    rows, dirs = [], {}
    for run_id in sample:
        app = manifest[run_id]["app"]
        try:
            dirs[run_id] = pull_run(remote, app, run_id, work)
            rows.append({"run_id": run_id, "download": "ok"})
        except Exception as e:  # checksum mismatch or a missing file
            rows.append({"run_id": run_id, "download": f"FAILED: {e!r}"[:300], "replay_level": "n/a"})
    log(f"audit: replaying {len(dirs)} of {len(ids)} runs end to end")
    replay_parallel(list(dirs.values()), device, workers, full=True, log=log)
    for row in rows:
        d = dirs.get(row["run_id"])
        if d is None:
            continue
        rep_path = d / layout.REPLAY_REPORT
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
        row.update(replay_level=rep.get("level", "error"), keyframes_checked=len(rep.get("keyframes_checked", [])),
                   log_steps_checked=len(rep.get("steps_checked", [])), max_err=rep.get("max_err"))
        if rep:
            update_replay_level(remote, d)
        shutil.rmtree(d, ignore_errors=True)
    return rows


def drop_failed(remote: Remote, run_ids: list[str], log: Callable[[str], None] = print) -> list[str]:
    """Delete runs whose replay failed, so the next sweep retrains them (PLAN.md §4.1:
    failed runs are excluded and rerun)."""
    return delete_runs(remote, run_ids, log=log) if run_ids else []

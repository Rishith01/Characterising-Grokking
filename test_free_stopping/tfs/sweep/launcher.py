"""Run many configs concurrently, one subprocess per run (DESIGN.md §6, "Launcher").

One process per run keeps runs isolated (determinism flags, CUDA context, memory) and
is enough to keep a GPU busy at these model sizes. A run whose directory is already
complete (DONE marker) is skipped, so re-launching a sweep resumes it.

With `replay_check=True`, each finished run is followed, in the same worker slot, by a
one-keyframe-interval replay check in a fresh process (scripts.replay_check), which
writes the run's replay_report.json before anything uploads it.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .. import cpu_env
from ..config import RunConfig
from ..storage import layout

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class LaunchResult:
    run_id: str
    run_dir: Path
    status: str  # complete | diverged | failed | skipped
    wall_s: float
    log: str = ""
    replay_level: Optional[str] = None  # set when replay_check ran: L0 | L1 | failed | error


def child_env(device: str, workers: int) -> dict:
    """Environment for child runs: CPU numerics pinned for reproducibility (tfs.cpu_env),
    and every worker, CPU or GPU, gets an equal share of the CPUs actually available.

    GPU workers need a share too: the training math runs on the GPU, but each process
    still does CPU work (Python, summaries, data indexing), and by default torch starts
    one thread per core in every process. Phase 1b measured what that does at 20 CPUs:
    aggregate throughput fell from 488 steps/s at 8 GPU workers to 109 at 16 and 24 at
    32, with the GPU mostly idle."""
    cpus = (os.cpu_count() or 1) if cpu_env.unpinned() else cpu_env.available_cpus()
    return cpu_env.pin(dict(os.environ), max(1, cpus // workers))


def run_parallel(configs: list[RunConfig], runs_root: Path, device: str = "cpu", workers: int = 4,
                 log: Callable[[str], None] = print, poll_s: float = 0.5, replay_check: bool = False,
                 on_result: Optional[Callable[[LaunchResult], None]] = None) -> list[LaunchResult]:
    """Train every config, `workers` at a time. `on_result` is called (in this thread)
    as each run finishes, after its replay check if one was requested."""
    runs_root = Path(runs_root).resolve()
    env = child_env(device, workers)
    pending = list(configs)
    running: dict = {}
    results: list[LaunchResult] = []
    total = len(pending)

    def finish(res: LaunchResult):
        results.append(res)
        extra = f", replay {res.replay_level}" if res.replay_level else ""
        log(f"[{len(results)}/{total}] {res.run_id}: {res.status} in {res.wall_s:.0f}s{extra}")
        if on_result is not None:
            on_result(res)

    with tempfile.TemporaryDirectory() as tmp:
        while pending or running:
            while pending and len(running) < workers:
                cfg = pending.pop(0)
                run_dir = layout.run_dir(runs_root, cfg.app, cfg.run_id)
                if (run_dir / layout.DONE).exists():
                    finish(LaunchResult(cfg.run_id, run_dir, "skipped", 0.0))
                    continue
                yaml_path = Path(tmp) / f"{cfg.run_id}.yaml"
                cfg.to_yaml(yaml_path)
                log_path = Path(tmp) / f"{cfg.run_id}.log"
                cmd = [sys.executable, "-m", "scripts.train", str(yaml_path), "--device", device,
                       "--runs-root", str(runs_root), "--overwrite"]
                running[_spawn(cmd, log_path, env)] = ("train", cfg, run_dir, time.time(), log_path, None)
            for proc in [p for p in running if p.poll() is not None]:
                kind, cfg, run_dir, t0, log_path, res = running.pop(proc)
                out = log_path.read_text(errors="replace")
                if kind == "train":
                    status = _status(run_dir) if proc.returncode == 0 else "failed"
                    res = LaunchResult(cfg.run_id, run_dir, status, time.time() - t0, out[-4000:])
                    if status == "failed":
                        log(out[-2000:])
                    elif replay_check:
                        # The slot stays busy: the replay check runs next, in a fresh process.
                        cmd = [sys.executable, "-m", "scripts.replay_check", str(run_dir), "--device", device,
                               "--seed", str(_interval_seed(cfg.run_id))]
                        rlog = log_path.with_suffix(".replay.log")
                        running[_spawn(cmd, rlog, env)] = ("replay", cfg, run_dir, t0, rlog, res)
                        continue
                    finish(res)
                else:
                    res.replay_level = _replay_level(run_dir, proc.returncode)
                    if res.replay_level == "error":
                        log(out[-2000:])
                    finish(res)
            if running:
                time.sleep(poll_s)
    return results


def _spawn(cmd: list[str], log_path: Path, env: dict) -> subprocess.Popen:
    # Output goes to a file, never a pipe: a full pipe buffer would block the child.
    with open(log_path, "w") as fh:
        return subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)


def _interval_seed(run_id: str) -> int:
    """Which keyframe interval a run's replay check covers: random across runs, fixed per run."""
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)


def _replay_level(run_dir: Path, returncode: int) -> str:
    report = run_dir / layout.REPLAY_REPORT
    if returncode in (0, 1) and report.exists():
        return json.loads(report.read_text())["level"]
    return "error"


def _status(run_dir: Path) -> str:
    done = run_dir / layout.DONE
    return done.read_text().strip() if done.exists() else "failed"


def replay_parallel(run_dirs: list[Path], device: str = "cpu", workers: int = 4, full: bool = True,
                    log: Callable[[str], None] = print, poll_s: float = 0.5) -> dict[str, int]:
    """Run scripts.replay_check on each run dir (writes replay_report.json). Returns
    {run_dir: exit code}."""
    pending = [Path(d) for d in run_dirs]
    running, codes = {}, {}
    env = child_env(device, workers)
    tmp = tempfile.TemporaryDirectory()
    while pending or running:
        while pending and len(running) < workers:
            d = pending.pop(0)
            cmd = [sys.executable, "-m", "scripts.replay_check", str(d), "--device", device]
            if full:
                cmd.append("--full")
            log_path = Path(tmp.name) / f"{len(codes) + len(running)}_{d.name}.log"
            with open(log_path, "w") as fh:
                proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[proc] = (d, log_path)
        for proc in [p for p in running if p.poll() is not None]:
            d, log_path = running.pop(proc)
            codes[str(d)] = proc.returncode
            out = log_path.read_text(errors="replace")
            log(out.strip().splitlines()[-1] if out.strip() else f"{d}: exit {proc.returncode}")
            if proc.returncode not in (0, 1):
                log(out[-2000:])
        if running:
            time.sleep(poll_s)
    tmp.cleanup()
    return codes

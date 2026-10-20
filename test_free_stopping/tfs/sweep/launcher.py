"""Run many configs concurrently, one subprocess per run (DESIGN.md §6, "Launcher").

One process per run keeps runs isolated (determinism flags, CUDA context, memory) and
is enough to keep a GPU busy at these model sizes. A run whose directory is already
complete (DONE marker) is skipped, so re-launching a sweep resumes it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

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


def child_env(device: str, workers: int) -> dict:
    """Give each CPU worker an equal share of cores; N workers that each grab every core
    oversubscribe the machine and run far slower than one process would."""
    env = dict(os.environ)
    if device == "cpu":
        n = str(max(1, (os.cpu_count() or 1) // workers))
        env.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, TFS_NUM_THREADS=n)
    return env


def run_parallel(configs: list[RunConfig], runs_root: Path, device: str = "cpu", workers: int = 4,
                 log: Callable[[str], None] = print, poll_s: float = 0.5) -> list[LaunchResult]:
    runs_root = Path(runs_root).resolve()
    env = child_env(device, workers)
    pending = list(configs)
    running: dict = {}
    results: list[LaunchResult] = []
    total = len(pending)
    with tempfile.TemporaryDirectory() as tmp:
        while pending or running:
            while pending and len(running) < workers:
                cfg = pending.pop(0)
                run_dir = layout.run_dir(runs_root, cfg.app, cfg.run_id)
                if (run_dir / layout.DONE).exists():
                    results.append(LaunchResult(cfg.run_id, run_dir, "skipped", 0.0))
                    log(f"[{len(results)}/{total}] {cfg.run_id}: already complete, skipped")
                    continue
                yaml_path = Path(tmp) / f"{cfg.run_id}.yaml"
                cfg.to_yaml(yaml_path)
                log_path = Path(tmp) / f"{cfg.run_id}.log"
                with open(log_path, "w") as fh:
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "scripts.train", str(yaml_path), "--device", device,
                         "--runs-root", str(runs_root), "--overwrite"],
                        cwd=PROJECT_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env,
                    )
                running[proc] = (cfg, run_dir, time.time(), log_path)
            for proc in [p for p in running if p.poll() is not None]:
                cfg, run_dir, t0, log_path = running.pop(proc)
                out = log_path.read_text(errors="replace")
                status = _status(run_dir) if proc.returncode == 0 else "failed"
                results.append(LaunchResult(cfg.run_id, run_dir, status, time.time() - t0, out[-4000:]))
                log(f"[{len(results)}/{total}] {cfg.run_id}: {status} in {time.time() - t0:.0f}s")
                if status == "failed":
                    log(out[-2000:])
            if running:
                time.sleep(poll_s)
    return results


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

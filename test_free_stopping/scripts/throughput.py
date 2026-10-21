"""GPU throughput benchmark (informs Phase 2/3 sizing).

Runs one fixed set of short runs (8 seeds x {teacher-student, modular, MNIST d3, MNIST d8},
3000 steps) at several worker counts, and optionally once more under NVIDIA MPS
(Multi-Process Service, which lets kernels from different processes run concurrently
instead of time-slicing). For each setting it reports aggregate optimizer steps per
second, and checks that every run's keyframe hashes are identical to the first setting:
concurrency level and MPS must not change any result.

Usage: python -m scripts.throughput --workers 8 16 32 [--mps-workers 16] --runs-root /tmp/tp --out tp.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from tfs.storage import layout
from tfs.sweep.grid import expand
from tfs.sweep.launcher import run_parallel

STEPS = 3000
REC = {"log_every": 50, "summary_every": 500, "keyframe_every": 1000}


def bench_configs():
    seeds = list(range(8))
    base = {"steps": STEPS, "record": REC, "seed": 0}
    return expand({"blocks": [
        {"base": {**base, "task": {"name": "teacher_student", "params": {"n_train": 100, "n_test": 100}},
                  "model": {"depth": 3, "width": 100, "activation": "tanh", "init_scale": 2.0},
                  "optim": {"name": "adamw", "lr": 3e-4, "weight_decay": 0.03}, "loss": "mse"},
         "product": {"seed": seeds}},
        {"base": {**base, "task": {"name": "modular", "params": {"op": "x+y", "p": 97, "train_frac": 0.3}},
                  "model": {"depth": 3, "width": 512, "activation": "relu", "init_scale": 1.0},
                  "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1.0}, "loss": "ce"},
         "product": {"seed": seeds}},
        {"base": {**base, "task": {"name": "mnist", "params": {"n_train": 1000}}, "batch_size": 200,
                  "model": {"depth": 3, "width": 200, "activation": "relu", "init_scale": 8.0},
                  "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01}, "loss": "mse"},
         "product": {"seed": seeds}},
        {"base": {**base, "task": {"name": "mnist", "params": {"n_train": 2000}}, "batch_size": 200,
                  "model": {"depth": 8, "width": 400, "activation": "relu", "init_scale": 8.0},
                  "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01}, "loss": "mse"},
         "product": {"seed": seeds}},
    ]})


def _hashes(results) -> dict:
    return {r.run_id: (r.run_dir / layout.KEYFRAME_HASHES).read_text() if r.status == "complete" else None
            for r in results}


def _one(configs, root: Path, workers: int, label: str, log) -> tuple[dict, dict]:
    shutil.rmtree(root, ignore_errors=True)
    t0 = time.time()
    res = run_parallel(configs, root, "cuda", workers, log=lambda *_: None)
    wall = time.time() - t0
    n_ok = sum(r.status == "complete" for r in res)
    row = {"setting": label, "workers": workers, "runs_ok": n_ok, "runs": len(res), "wall_s": round(wall, 1),
           "agg_steps_per_s": round(n_ok * STEPS / wall), "mean_run_s": round(sum(r.wall_s for r in res) / len(res), 1)}
    log(f"{label}: {row}")
    hashes = _hashes(res)
    shutil.rmtree(root, ignore_errors=True)
    return row, hashes


class Mps:
    """Start/stop an MPS control daemon in a private directory, if the system allows."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="tfs_mps_"))
        self.env = {"CUDA_MPS_PIPE_DIRECTORY": str(self.dir / "pipe"), "CUDA_MPS_LOG_DIRECTORY": str(self.dir / "log")}
        self.started = False
        self.error = None

    def __enter__(self):
        if shutil.which("nvidia-cuda-mps-control") is None:
            self.error = "nvidia-cuda-mps-control not found"
            return self
        for d in self.env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **self.env},
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            self.error = f"start failed: {r.stderr.strip() or r.stdout.strip()}"
            return self
        time.sleep(2)
        os.environ.update(self.env)  # children inherit it via the launcher's environment
        self.started = True
        return self

    def __exit__(self, *exc):
        if self.started:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env={**os.environ, **self.env},
                           capture_output=True, text=True, timeout=30)
            for k in self.env:
                os.environ.pop(k, None)
        shutil.rmtree(self.dir, ignore_errors=True)


def throughput(workers_list: list[int], runs_root: Path, mps_workers: int | None = None, log=print) -> list[dict]:
    configs = bench_configs()
    rows, reference = [], None
    for w in workers_list:
        row, hashes = _one(configs, runs_root / f"w{w}", w, f"time-slicing, {w} workers", log)
        reference = reference or hashes
        row["identical_to_first"] = hashes == reference
        rows.append(row)
    if mps_workers:
        with Mps() as mps:
            if mps.started:
                row, hashes = _one(configs, runs_root / "mps", mps_workers, f"MPS, {mps_workers} workers", log)
                row["identical_to_first"] = hashes == reference
            else:
                row = {"setting": f"MPS, {mps_workers} workers", "error": mps.error}
                log(row)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--mps-workers", type=int, default=None)
    ap.add_argument("--runs-root", default="tp_runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = throughput(args.workers, Path(args.runs_root), args.mps_workers)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

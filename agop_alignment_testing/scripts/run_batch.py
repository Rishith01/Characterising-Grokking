"""Batch runner for Phase 2 corpus configs (project.md Phase 2: "freeze the
evaluation corpus"). Runs every config in sequence, printing progress for each one
so a long batch can be monitored, and keeps going past individual failures instead
of aborting the whole batch.

Usage: python -m scripts.run_batch configs/phase2/*.yaml
       python -m scripts.run_batch configs/phase2/*.yaml --runs-dir runs
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Union

import pandas as pd

from agopx.runner import run


def run_batch(config_paths: list, runs_dir: Union[str, Path] = "runs") -> dict:
    results = {"ok": [], "failed": []}
    total = len(config_paths)
    t_batch_start = time.time()

    for i, config_path in enumerate(config_paths, 1):
        name = Path(config_path).stem
        print(f"[{i}/{total}] {name} -- starting", flush=True)
        t0 = time.time()
        try:
            run_dir = run(config_path, runs_dir)
            elapsed = time.time() - t0
            summary = _final_metrics_summary(run_dir)
            print(f"[{i}/{total}] {name} -- done in {elapsed:.1f}s  {summary}", flush=True)
            results["ok"].append(str(run_dir))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i}/{total}] {name} -- FAILED after {elapsed:.1f}s: {e!r}", flush=True)
            results["failed"].append((str(config_path), str(e)))

    total_elapsed = time.time() - t_batch_start
    print(
        f"\nBatch done: {len(results['ok'])} ok, {len(results['failed'])} failed, "
        f"{total_elapsed / 60:.1f} min total",
        flush=True,
    )
    if results["failed"]:
        print("Failed configs:", flush=True)
        for config_path, err in results["failed"]:
            print(f"  {config_path}: {err}", flush=True)

    return results


def _final_metrics_summary(run_dir: Union[str, Path]) -> str:
    df = pd.read_json(Path(run_dir) / "metrics.jsonl", lines=True)
    last = df.iloc[-1]
    return f"final test_acc={last['test/accuracy']:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    results = run_batch(args.configs, args.runs_dir)
    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

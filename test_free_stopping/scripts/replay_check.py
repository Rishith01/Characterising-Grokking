"""Verify replay for one or more local runs and write replay_report.json for each.

Usage: python -m scripts.replay_check runs/dummy/<run_id> [...] [--full] [--device cpu]

Default checks one randomly chosen keyframe interval; --full replays the whole run.
Exit code 1 if any run fails.
"""
from __future__ import annotations

import argparse
import random
import sys

from tfs import cpu_env

cpu_env.pin()  # before torch loads: OpenMP reads its variables at load time

from tfs.training import determinism  # noqa: E402

determinism.apply()

from tfs.training.replay import ReplayEngine  # noqa: E402


def check(run_dir: str, full: bool, device: str, allow_env_mismatch: bool, rng: random.Random) -> str:
    eng = ReplayEngine(run_dir, device, allow_env_mismatch=allow_env_mismatch)
    keys = [int(k) for k in eng.reader.keyframe_steps]
    if full or len(keys) < 2:
        intervals = "all"
    else:
        i = rng.randrange(len(keys) - 1)
        intervals = [(keys[i], keys[i + 1])]
    report = eng.verify(intervals)
    eng.write_report(report)
    print(f"{run_dir}: {report.level}  (log steps {len(report.steps_checked)}, "
          f"keyframes {report.keyframes_checked}, max_err {report.max_err})")
    if report.first_failure:
        print(f"  first failure: {report.first_failure}")
    return report.level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="for choosing the random interval")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    levels = [check(d, args.full, args.device, args.allow_env_mismatch, rng) for d in args.run_dirs]
    sys.exit(1 if "failed" in levels else 0)


if __name__ == "__main__":
    main()

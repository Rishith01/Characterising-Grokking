"""Train one run from a YAML config (development; no upload).

Usage: python -m scripts.train configs/runs/dummy.yaml [--device cpu] [--runs-root runs] [--seed N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tfs import cpu_env

cpu_env.pin()  # before torch loads: OpenMP reads its variables at load time

from tfs.training import determinism  # noqa: E402

determinism.apply()  # before anything touches CUDA

from tfs.config import RunConfig  # noqa: E402
from tfs.training.trainer import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--seed", type=int, default=None, help="override the config's seed")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.with_seed(args.seed)
    run_dir = Trainer(cfg, Path(args.runs_root), args.device, overwrite=args.overwrite).run()
    print(run_dir)


if __name__ == "__main__":
    main()

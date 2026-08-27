"""config -> run directory (project.md Section 6).

runs/<run_id>/
    config.yaml       # copy of the input config, for provenance
    metrics.jsonl      # one line per snapshot: {"t": ..., **snap.metrics}
    features/M_t.npy   # snap.M, float32 (project.md "Practical notes")
    features/G_t.npy   # snap.agop, float32, only when the learner produces one
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from .learners.base import Snapshot
from .learners.neural import NeuralConfig, NNLearner
from .learners.rfm import RFMConfig, RFMLearner

LEARNERS = {"rfm": RFMLearner, "nn": NNLearner}
CONFIGS = {"rfm": RFMConfig, "nn": NeuralConfig}


def run(config_path: Union[str, Path], runs_dir: Union[str, Path] = "runs") -> Path:
    config_path = Path(config_path)
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    learner_type = raw.pop("learner", "rfm")
    run_id = raw.pop("run_id", config_path.stem)

    config = CONFIGS[learner_type](**raw)
    learner = LEARNERS[learner_type](config)

    run_dir = Path(runs_dir) / run_id
    features_dir = run_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")

    with open(run_dir / "metrics.jsonl", "w") as f:
        for snap in learner.steps():
            np.save(features_dir / f"M_{snap.t}.npy", snap.M.astype(np.float32))
            if snap.agop is not None:
                np.save(features_dir / f"G_{snap.t}.npy", snap.agop.astype(np.float32))
            f.write(json.dumps({"t": snap.t, **snap.metrics}) + "\n")

    return run_dir


def load_run(run_dir: Union[str, Path]) -> list:
    """The inverse of run(): reconstruct the list of Snapshots a run produced, by
    reading metrics.jsonl back against the saved features/M_t.npy files.
    """
    run_dir = Path(run_dir)
    features_dir = run_dir / "features"

    snapshots = []
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            record = json.loads(line)
            t = record.pop("t")
            M = np.load(features_dir / f"M_{t}.npy")
            agop_path = features_dir / f"G_{t}.npy"
            agop = np.load(agop_path) if agop_path.exists() else None
            snapshots.append(Snapshot(t=t, M=M, metrics=record, agop=agop))
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--runs-dir", type=str, default="runs")
    args = parser.parse_args()

    run_dir = run(args.config, args.runs_dir)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()

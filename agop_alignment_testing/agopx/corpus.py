"""Evaluation corpus of labelled runs (project.md Section 6 / Phase 2, "Freeze the
evaluation corpus"). Each run gets a grok step -- the first t where test accuracy
crosses 0.9 -- or None if it never does. Any probe is scored against exactly this
frozen set (agopx/evaluate.py, Phase 3).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

GROK_THRESHOLD = 0.9


@dataclass
class CorpusEntry:
    run_id: str
    run_dir: str
    category: str  # "grokker", "non_grokker", "fast_learner", "multi_task"
    learner: str  # "rfm" or "nn"
    operation: str
    kernel_type: Optional[str]
    seed: int
    grok_step: Optional[int]


def grok_step(run_dir: Union[str, Path], threshold: float = GROK_THRESHOLD) -> Optional[int]:
    df = pd.read_json(Path(run_dir) / "metrics.jsonl", lines=True)
    hits = df[df["test/accuracy"] >= threshold]
    if len(hits) == 0:
        return None
    return int(hits["t"].iloc[0])


def label_run(run_dir: Union[str, Path], category: str) -> CorpusEntry:
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    return CorpusEntry(
        run_id=run_dir.name,
        run_dir=str(run_dir),
        category=category,
        learner=config.get("learner", "rfm"),
        operation=config["operation"],
        kernel_type=config.get("kernel_type"),
        seed=config.get("seed", 0),
        grok_step=grok_step(run_dir),
    )


def build_corpus(entries: list, out_path: Union[str, Path] = "corpus.jsonl") -> list:
    """entries: list of (run_dir, category) pairs."""
    labeled = [label_run(run_dir, category) for run_dir, category in entries]
    with open(out_path, "w") as f:
        for e in labeled:
            f.write(json.dumps(asdict(e)) + "\n")
    return labeled


def load_corpus(path: Union[str, Path] = "corpus.jsonl") -> list:
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(CorpusEntry(**json.loads(line)))
    return entries

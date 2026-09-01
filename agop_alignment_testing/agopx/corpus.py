"""Evaluation corpus of labelled runs (project.md Section 6 / Phase 2, "Freeze the
evaluation corpus"). Each run gets a grok step -- the first t where test accuracy
crosses 0.9 -- or None if it never does. Any probe is scored against exactly this
frozen set (agopx/evaluate.py, Phase 3).

Categories are validated against the run's measured outcome, not taken on trust; see
CATEGORY_EXPECTS_GROK and _check_category.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

GROK_THRESHOLD = 0.9

# Whether a run in each category must reach GROK_THRESHOLD test accuracy.
#   True  -- must grok; None grok_step is a contradiction.
#   False -- must never grok; a non-None grok_step is a contradiction.
#   None  -- either outcome is admissible for the category.
#
# "partial_learner" is the sub-threshold training-fraction regime (roughly r in
# [0.20, 0.30] for quadratic at p=61): circulant deviation falls several-fold while
# test accuracy stays near zero. Those runs are structurally *not* non-grokkers --
# real feature learning happens -- but they never generalize, which makes them the
# sharpest available negative for any probe that merely detects emerging structure.
CATEGORY_EXPECTS_GROK = {
    "grokker": True,
    "fast_learner": True,
    "non_grokker": False,
    "partial_learner": False,
}


class CorpusLabelError(ValueError):
    """A run's hand-assigned category contradicts its measured grok_step.

    This exists because the category used to be taken on trust, and a batch of NN
    runs configured as a no-weight-decay "non-grokker" control in fact grokked at
    epoch 25 -- they sat in the corpus as non_grokker entries and would have
    contaminated the false-positive arm of every probe score. The paper's
    no-regularization control (Appendix Fig. 5, left) is vanilla SGD run for 200k
    epochs; translating it to AdamW with weight_decay=0 for 50 epochs produces a
    run that groks, just later. Categories are now checked against the data.
    """


@dataclass
class CorpusEntry:
    run_id: str
    run_dir: str
    category: str
    learner: str  # "rfm" or "nn"
    operation: str
    kernel_type: Optional[str]
    p: int
    training_fraction: float
    seed: int
    n_steps: int  # number of snapshots recorded, i.e. trajectory length
    # Which matrix probes should read for this run (features.select_matrix).
    # RFM's M_t is already [G]^{1/2}; the NN records the NFM in M and its own AGOP
    # alongside, and the paper's Fig. 5B measures are computed on sqrt(AGOP).
    probe_source: str
    grok_step: Optional[int]


def grok_step(run_dir: Union[str, Path], threshold: float = GROK_THRESHOLD) -> Optional[int]:
    df = pd.read_json(Path(run_dir) / "metrics.jsonl", lines=True)
    hits = df[df["test/accuracy"] >= threshold]
    if len(hits) == 0:
        return None
    return int(hits["t"].iloc[0])


def _check_category(run_id: str, category: str, step: Optional[int]) -> None:
    if category not in CATEGORY_EXPECTS_GROK:
        raise CorpusLabelError(
            f"{run_id}: unknown category {category!r}; have {sorted(CATEGORY_EXPECTS_GROK)}"
        )
    expects = CATEGORY_EXPECTS_GROK[category]
    if expects is True and step is None:
        raise CorpusLabelError(
            f"{run_id}: labelled {category!r} but never reached {GROK_THRESHOLD:.0%} test "
            f"accuracy. Either the run is misconfigured or the label is wrong."
        )
    if expects is False and step is not None:
        raise CorpusLabelError(
            f"{run_id}: labelled {category!r} but grokked at t={step}. A run that groks "
            f"cannot serve in the false-positive arm; fix the config or relabel it."
        )


def label_run(run_dir: Union[str, Path], category: str) -> CorpusEntry:
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    step = grok_step(run_dir)
    _check_category(run_dir.name, category, step)

    learner = config.get("learner", "rfm")
    n_steps = sum(1 for _ in open(run_dir / "metrics.jsonl"))

    return CorpusEntry(
        run_id=run_dir.name,
        run_dir=str(run_dir),
        category=category,
        learner=learner,
        operation=config["operation"],
        kernel_type=config.get("kernel_type"),
        p=config["p"],
        training_fraction=config["training_fraction"],
        seed=config.get("seed", 0),
        n_steps=n_steps,
        probe_source="sqrt_agop" if learner == "nn" else "M",
        grok_step=step,
    )


def build_corpus(entries: list, out_path: Union[str, Path] = "corpus.jsonl") -> list:
    """entries: list of (run_dir, category) pairs.

    Every entry is validated before anything is written, so a single mislabelled run
    fails the whole build rather than silently landing in the frozen corpus.
    """
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

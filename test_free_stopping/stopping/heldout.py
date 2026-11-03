"""The only module in `stopping` that reads test-side data (DESIGN.md §10, §13.5).

B6, validation-loss patience, is an oracle reference: it is reported separately and
never counted as a contender. tests/test_quarantine.py fails if any other signal, rule or
runner module imports this one (the adapter's RunHandle.heldout() is the one exception).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd

from tfs.evaluation.oracle import Oracle

from .method import Method
from .rule import RuleConfig
from .signal import Signal, SignalConfig, register
from .source import Observation


def read_heldout(run_dir: Path) -> pd.DataFrame:
    """step, test_loss, test_acc per log step."""
    return Oracle(run_dir).table()


@dataclass(frozen=True)
class ValLossConfig(SignalConfig):
    name: ClassVar[str] = "val_loss"


@register(ValLossConfig)
class ValLoss(Signal):
    uses_heldout = True
    requires = frozenset({"heldout"})
    cadence = "log"

    def update(self, obs: Observation) -> Optional[dict]:
        h = obs.heldout
        if h is None:
            return None
        return {"value": float(h["test_loss"]), "test_acc": float(h["test_acc"])}


def b6_method(threshold: float = 0.0, patience: int = 20) -> Method:
    """B6 with uncalibrated defaults: stop when the smoothed test loss has not improved
    for `patience` log rows."""
    return Method("B6 val-loss patience (oracle)", (
        (ValLossConfig(), RuleConfig(mode="plateau", direction="below", threshold=threshold, patience=patience)),
    ))

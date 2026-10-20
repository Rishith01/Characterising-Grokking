"""The only module allowed to read test-side data (DESIGN.md §10).

tests/test_quarantine.py fails if signal, rule, causal or reader code imports this
module or names the oracle file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..storage import layout


class Oracle:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def table(self) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir / layout.ORACLE)

    def test_acc(self) -> pd.Series:
        df = self.table()
        return pd.Series(df["test_acc"].to_numpy(), index=df["step"].to_numpy(), name="test_acc")

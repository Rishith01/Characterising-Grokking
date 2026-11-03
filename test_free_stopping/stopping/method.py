"""Method: a named stopping method = one or more (signal config, rule config) parts
(DESIGN.md §13.4). With several parts the method stops at the earliest part's stop
(any-of), e.g. a stationarity rule plus a train-accuracy guard."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .rule import Rule, RuleConfig, StopDecision
from .signal import SignalConfig


@dataclass(frozen=True)
class Method:
    name: str
    parts: tuple[tuple[SignalConfig, RuleConfig], ...]

    def signals(self) -> list[SignalConfig]:
        seen, out = set(), []
        for s, _ in self.parts:
            if s.key not in seen:
                seen.add(s.key)
                out.append(s)
        return out

    @property
    def uses_heldout(self) -> bool:
        return any(s.build().uses_heldout for s, _ in self.parts)

    def decide(self, traces: Mapping[str, pd.DataFrame]) -> StopDecision:
        """traces: signal config key -> that signal's trace for one run."""
        best = StopDecision(None, reason="no part stopped")
        for i, (s, r) in enumerate(self.parts):
            d = Rule(r).decide(traces[s.key])
            if d.step is not None and (best.step is None or d.step < best.step):
                best = StopDecision(d.step, d.value, i, f"{s.name}: {d.reason}")
        return best

"""Probe interface (project.md Section 6, "Probe: where the research lives").

causal=False probes (the paper's baselines, AGOPAlignment/CirculantDeviation) may see
the full trajectory including the endpoint M*. causal=True probes may only see
M_0..M_t at the point they are scored for step t -- evaluate.py's harness is
responsible for enforcing that boundary, not the probe itself (project.md: "the
harness must refuse to score a causal probe using anything past step t").
"""
from __future__ import annotations

from typing import Optional, Protocol

from ..learners.base import Snapshot


class Probe(Protocol):
    causal: bool

    def update(self, snap: Snapshot) -> Optional[dict]: ...

    def finalize(self, traj: list) -> dict: ...

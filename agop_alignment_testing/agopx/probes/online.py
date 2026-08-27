"""Causal probe candidates (project.md Section 3, Phase 3 "Online probes").
Not implemented in Phase 0 -- these are the actual research payoff, to be built
once Phase 1's gate passes and Phase 2's evaluation corpus is frozen.

Candidates from the plan:
    1. IncrementCoherence         -- cos(Delta_t, Delta_{t-1})
    2. PathDirectionPersistence   -- cos(Delta_t, M_hat_t - M_hat_0)
    3. SlidingWindowExtrapolation -- fit rho(M_t, M_{t+k}) over a trailing window,
                                      extrapolate to where it would reach 1
    4. SpectralVariants (stretch) -- AGOP spectrum entropy / participation ratio,
                                      top-eigenspace rotation angle between steps

Explicitly rejected: raw rho(M_t, M_{t+1}) -- confounded, high both when nothing is
happening and when training has converged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..features import frobenius_normalize, offdiag_block
from .base import Probe
from .registry import register


@register("increment_norm")
class IncrementNorm(Probe):
    """||Delta_t||_F, Delta_t = M_hat_t - M_hat_{t-1}, where M_hat is the
    off-diagonal block, Frobenius-normalised before differencing (project.md
    Phase 3 preprocessing note -- otherwise trace growth from the s=1/2 AGOP power
    gets measured instead of direction).

    Not one of the paper's four listed candidates -- a precursor diagnostic for
    the raw magnitude of drift, to look at before building IncrementCoherence
    (cosine of consecutive deltas) on top of it. Genuinely causal by construction:
    update() only ever touches the current snapshot plus one carried-over previous
    M_hat, never anything from finalize()'s full trajectory.
    """

    causal = True

    def __init__(self, p: int):
        self.p = p
        self._prev_m_hat: Optional[np.ndarray] = None

    def _m_hat(self, snap) -> np.ndarray:
        return frobenius_normalize(offdiag_block(snap.M.astype(np.float64), self.p))

    def update(self, snap) -> Optional[dict]:
        m_hat = self._m_hat(snap)
        result = None
        if self._prev_m_hat is not None:
            delta = m_hat - self._prev_m_hat
            result = {"t": snap.t, "increment_norm": float(np.linalg.norm(delta))}
        self._prev_m_hat = m_hat
        return result

    def finalize(self, traj: list) -> dict:
        self._prev_m_hat = None
        points = [r for snap in traj if (r := self.update(snap)) is not None]
        return {"trajectory": points}


@register("increment_coherence")
class IncrementCoherence(Probe):
    """cos(Delta_t, Delta_{t-1}), Delta_t = M_hat_t - M_hat_{t-1}, M_hat the
    off-diagonal block, Frobenius-normalised before differencing (same
    preprocessing as IncrementNorm). Coherent drift toward a fixed target gives
    ~1 (consecutive steps keep pointing the same way); aimless wandering ~0.
    Unlike ConsecutiveAGOPAlignment (which compares M_hat states themselves, and
    saturates at 1 the moment M stops moving much either way), this compares the
    *directions* of consecutive steps, so a converged/flat run should show
    coherence collapsing toward 0 (small residual deltas pointing every which way)
    rather than staying pinned at 1 -- project.md candidate #1.

    Genuinely causal: needs three consecutive snapshots (two deltas) before its
    first value, one step later than IncrementNorm/ConsecutiveAGOPAlignment.
    update() only ever carries forward the previous M_hat and the previous Delta.
    """

    causal = True

    def __init__(self, p: int):
        self.p = p
        self._prev_m_hat: Optional[np.ndarray] = None
        self._prev_delta: Optional[np.ndarray] = None

    def _m_hat(self, snap) -> np.ndarray:
        return frobenius_normalize(offdiag_block(snap.M.astype(np.float64), self.p))

    def update(self, snap) -> Optional[dict]:
        m_hat = self._m_hat(snap)
        result = None
        if self._prev_m_hat is not None:
            delta = m_hat - self._prev_m_hat
            if self._prev_delta is not None:
                prev_norm = np.linalg.norm(self._prev_delta)
                cur_norm = np.linalg.norm(delta)
                cos = 0.0
                if prev_norm > 0 and cur_norm > 0:
                    cos = float(np.dot(delta.ravel(), self._prev_delta.ravel()) / (prev_norm * cur_norm))
                result = {"t": snap.t, "increment_coherence": cos}
            self._prev_delta = delta
        self._prev_m_hat = m_hat
        return result

    def finalize(self, traj: list) -> dict:
        self._prev_m_hat = None
        self._prev_delta = None
        points = [r for snap in traj if (r := self.update(snap)) is not None]
        return {"trajectory": points}


@register("consecutive_agop_alignment")
class ConsecutiveAGOPAlignment(Probe):
    """cos(M_hat_t, M_hat_{t-1}) -- same off-diagonal-block preprocessing as
    AGOPAlignment (probes/offline.py), but against the previous step instead of
    the endpoint M*.

    This is exactly the quantity project.md explicitly rejects: "raw
    rho(M_t, M_{t+1})... is confounded -- consecutive similarity is high both when
    nothing is happening and when training has converged, so it cannot separate a
    dead run from a finished one." Implemented anyway (on request) to see that
    failure mode directly. Genuinely causal by construction, same pattern as
    IncrementNorm above: update() only ever touches the current snapshot plus one
    carried-over previous M_hat.
    """

    causal = True

    def __init__(self, p: int):
        self.p = p
        self._prev_m_hat: Optional[np.ndarray] = None

    def _m_hat(self, snap) -> np.ndarray:
        return offdiag_block(snap.M.astype(np.float64), self.p)

    def update(self, snap) -> Optional[dict]:
        m_hat = self._m_hat(snap)
        result = None
        if self._prev_m_hat is not None:
            prev_norm = np.linalg.norm(self._prev_m_hat)
            cur_norm = np.linalg.norm(m_hat)
            cos = 0.0
            if prev_norm > 0 and cur_norm > 0:
                cos = float(np.dot(m_hat.ravel(), self._prev_m_hat.ravel()) / (prev_norm * cur_norm))
            result = {"t": snap.t, "consecutive_agop_alignment": cos}
        self._prev_m_hat = m_hat
        return result

    def finalize(self, traj: list) -> dict:
        self._prev_m_hat = None
        points = [r for snap in traj if (r := self.update(snap)) is not None]
        return {"trajectory": points}

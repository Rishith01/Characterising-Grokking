"""Causal probe candidates (project.md Section 3, Phase 3 "Online probes").

Candidates from the plan:
    1. IncrementCoherence         -- cos(Delta_t, Delta_{t-1})
    2. PathDirectionPersistence   -- cos(Delta_t, M_hat_t - M_hat_0)
    3. SlidingWindowExtrapolation -- fit rho(M_t, M_{t+k}) over a trailing window,
                                      extrapolate to where it would reach 1
    4. SpectralVariants (stretch) -- AGOP spectrum entropy / participation ratio,
                                      top-eigenspace rotation angle between steps

Explicitly rejected: raw rho(M_t, M_{t+1}) -- confounded, high both when nothing is
happening and when training has converged. Implemented anyway as
ConsecutiveAGOPAlignment, to see that failure mode directly.

Shared preprocessing (project.md Phase 3): read the off-diagonal block only, and
Frobenius-normalise before differencing, else trace growth from the s=1/2 AGOP power
gets measured instead of direction. `source` selects which matrix to read from each
Snapshot -- "M" for RFM, "sqrt_agop" for NN runs being compared against the paper's
Fig. 5B (see features.select_matrix).

Every probe here defers its first output until it has a *defined* direction to
difference. M_0 = I has an exactly-zero off-diagonal block, so features.unit_direction
returns None for it; seeding the difference chain from that zero matrix used to make
IncrementNorm report a constant 1.0 at t=1 on every M_0=I run and made
IncrementCoherence's first delta a state rather than an increment.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..features import offdiag_block, select_matrix, unit_direction
from .base import Probe
from .registry import register


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (a_norm * b_norm))


class _BlockDirectionProbe(Probe):
    """Shared plumbing: pull the off-diagonal block out of the requested matrix and
    reduce it to a unit direction, or None when it has no defined direction.
    """

    causal = True

    def __init__(self, p: int, source: str = "M"):
        self.p = p
        self.source = source
        self._reset()

    def _reset(self) -> None:
        raise NotImplementedError

    def _m_hat(self, snap) -> Optional[np.ndarray]:
        return unit_direction(offdiag_block(select_matrix(snap, self.source), self.p))

    def finalize(self, traj: list) -> dict:
        self._reset()
        points = [r for snap in traj if (r := self.update(snap)) is not None]
        return {"trajectory": points}


@register("increment_norm")
class IncrementNorm(_BlockDirectionProbe):
    """||Delta_t||_F, Delta_t = M_hat_t - M_hat_{t-1}, where M_hat is the normalised
    off-diagonal block.

    Not one of project.md's four listed candidates -- a precursor diagnostic for the
    raw magnitude of drift, to look at before building IncrementCoherence (cosine of
    consecutive deltas) on top of it. Genuinely causal by construction: update() only
    ever touches the current snapshot plus one carried-over previous M_hat, never
    anything from finalize()'s full trajectory.

    First value lands at the second snapshot with a defined direction -- t=2 for an
    M_0=I run, since M_0's off-diagonal block is exactly zero.
    """

    def _reset(self) -> None:
        self._prev_m_hat: Optional[np.ndarray] = None

    def update(self, snap) -> Optional[dict]:
        m_hat = self._m_hat(snap)
        if m_hat is None:
            return None
        result = None
        if self._prev_m_hat is not None:
            delta = m_hat - self._prev_m_hat
            result = {"t": snap.t, "increment_norm": float(np.linalg.norm(delta))}
        self._prev_m_hat = m_hat
        return result


@register("increment_coherence")
class IncrementCoherence(_BlockDirectionProbe):
    """cos(Delta_t, Delta_{t-1}), Delta_t = M_hat_t - M_hat_{t-1} -- project.md
    candidate #1. Coherent drift toward a fixed target gives ~1 (consecutive steps
    keep pointing the same way); aimless wandering ~0.

    Unlike ConsecutiveAGOPAlignment (which compares M_hat states themselves, and
    saturates at 1 the moment M stops moving much either way), this compares the
    *directions* of consecutive steps.

    Needs three consecutive snapshots with defined directions (two deltas) before its
    first value -- t=3 for an M_0=I run. update() only ever carries forward the
    previous M_hat and the previous Delta.

    Empirical status as of the last corpus: this does NOT separate grokkers from
    non-grokkers. Measured on quadratic x+y at p=61 it sits at 0.98-0.99 for
    grokkers, low-training-fraction non-grokkers, and random-label controls alike.
    The premise in the original design note -- that a dead run's residual deltas
    would point every which way and collapse coherence toward 0 -- is false: RFM is a
    fixed-point iteration, so it converges coherently toward *some* fixed point
    whether or not that fixed point generalizes.
    """

    def _reset(self) -> None:
        self._prev_m_hat: Optional[np.ndarray] = None
        self._prev_delta: Optional[np.ndarray] = None

    def update(self, snap) -> Optional[dict]:
        m_hat = self._m_hat(snap)
        if m_hat is None:
            return None
        result = None
        if self._prev_m_hat is not None:
            delta = m_hat - self._prev_m_hat
            if self._prev_delta is not None:
                result = {"t": snap.t, "increment_coherence": _cos(delta, self._prev_delta)}
            self._prev_delta = delta
        self._prev_m_hat = m_hat
        return result


@register("consecutive_agop_alignment")
class ConsecutiveAGOPAlignment(_BlockDirectionProbe):
    """cos(M_hat_t, M_hat_{t-1}) -- the same off-diagonal-block preprocessing as
    AGOPAlignment, but against the previous step instead of the endpoint M*.

    This is exactly the quantity project.md explicitly rejects: "raw
    rho(M_t, M_{t+1})... is confounded -- consecutive similarity is high both when
    nothing is happening and when training has converged, so it cannot separate a
    dead run from a finished one." Implemented anyway (on request) to see that
    failure mode directly, and it does fail exactly that way: ~0.99+ everywhere from
    a few steps in, on grokkers and non-grokkers alike.

    Genuinely causal by construction, same pattern as IncrementNorm above.
    """

    def _reset(self) -> None:
        self._prev_m_hat: Optional[np.ndarray] = None

    def update(self, snap) -> Optional[dict]:
        m_hat = self._m_hat(snap)
        if m_hat is None:
            return None
        result = None
        if self._prev_m_hat is not None:
            result = {"t": snap.t, "consecutive_agop_alignment": _cos(m_hat, self._prev_m_hat)}
        self._prev_m_hat = m_hat
        return result

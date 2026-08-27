"""The paper's two a posteriori baselines (project.md Section 3, "The two baseline
measures"). Both are causal=False: AGOPAlignment needs M* (the endpoint), and
CirculantDeviation needs the discrete-log reordering, which needs to know Z*_p is
cyclic of order p-1 in advance.

Both probes are stateless during the run -- they only need the full trajectory, which
evaluate.py's harness (Phase 3) will supply to finalize(). update() is a no-op.
"""
from __future__ import annotations

import numpy as np

from ..features import apply_circulant_transform, circulant_transform_for_operation, offdiag_block
from .base import Probe
from .registry import register


@register("agop_alignment")
class AGOPAlignment(Probe):
    """rho(M_t, M*) = cosine similarity of vectorised M_t and the final M*, on the
    off-diagonal block only.

    project.md Section 3 just says "cosine similarity of vectorised matrices"
    without specifying block extraction, but the Phase 3 preprocessing note ("the
    diagonal dominates M and swamps the structural signal") turned out to apply
    here too -- confirmed empirically (not just by that general principle): M_0=I
    is purely diagonal, so its off-diagonal block is exactly zero, making this
    version start at exactly 0.0 and rise to 1.0, matching the paper's reported
    curve shape. The full-matrix version used through Phase 1 started around ~0.78
    purely from diagonal-on-diagonal overlap with M*'s Observation-1
    c1*I + c2*11^T blocks -- not real structural progress, just an artifact of
    including the dominant diagonal in the cosine similarity.
    """

    causal = False

    def __init__(self, p: int):
        self.p = p

    def update(self, snap):
        return None

    def finalize(self, traj: list) -> dict:
        m_star = offdiag_block(traj[-1].M.astype(np.float64), self.p).ravel()
        star_norm = np.linalg.norm(m_star)

        points = []
        for snap in traj:
            v = offdiag_block(snap.M.astype(np.float64), self.p).ravel()
            v_norm = np.linalg.norm(v)
            cos = float(np.dot(v, m_star) / (v_norm * star_norm)) if v_norm > 0 else 0.0
            points.append({"t": snap.t, "agop_alignment": cos})
        return {"trajectory": points}


@register("circulant_deviation")
class CirculantDeviation(Probe):
    """D(A) over the bottom-left p x p block A of M_t (project.md Section 3):

        D(A) = (1 / ||A||_F^2) * sum_j Var( S(A)[:, j] )

    where S shifts row l of A right by l positions. Zero iff A is exactly circulant.

    Which index relabeling reveals that circulant structure is operation-dependent
    (features.CIRCULANT_TRANSFORMS): none for x+y; a column reversal for x-y (since
    x-y=x+(-y) and negation mod p reverses index order); discrete-log reorder for
    x*y; both for x/y. Only "reorder" (mul/div) is documented in project.md Section
    3 -- the rest were found empirically in Phase 1.

    For mul/div, index 0 has no discrete log (0 has no multiplicative inverse), so
    after reordering it is a meaningless row/col -- real trained matrices still put
    small non-zero content there (~5.5% of the block's norm for a converged x*y
    run), which isn't circulant and shouldn't be compared as if it were. It's
    dropped here (A[1:, 1:] after the transform) for any "reorder"-involving
    operation. This was originally NOT dropped, and it mattered a lot: x*y/x/y
    appeared to plateau at a stable ~0.006 (vs ~0.0003 for x+y/x-y), which Phase 1
    reported as mult/div having a genuinely weaker circulant structure. That
    conclusion was wrong -- it was this measurement artifact. Once index 0 is
    excluded, x*y/x/y reach the same ~0.0003 floor as x+y/x-y.
    """

    causal = False

    def __init__(self, p: int, operation: str):
        self.p = p
        self.operation = operation
        self.transform = circulant_transform_for_operation(operation)

    def update(self, snap):
        return None

    def finalize(self, traj: list) -> dict:
        points = []
        for snap in traj:
            A = offdiag_block(snap.M.astype(np.float64), self.p)
            A = apply_circulant_transform(A, self.p, self.transform)
            if "reorder" in self.transform:
                A = A[1:, 1:]
            points.append({"t": snap.t, "circulant_deviation": _deviation(A)})
        return {"trajectory": points}


def _deviation(A: np.ndarray) -> float:
    p = A.shape[0]
    # np.roll(A[row], k) puts A[row, j] at position (j + k) mod p, i.e. it moves
    # entries towards higher indices for k > 0 -- shifting *row* l right by l in
    # the array-index sense that undoes scipy.linalg.circulant's construction
    # means rolling by -l here (verified: gives exactly 0 variance on a genuine
    # circulant matrix, vs. O(1) variance with +l).
    shifted = np.stack([np.roll(A[row], -row) for row in range(p)], axis=0)
    col_var = shifted.var(axis=0)
    norm_sq = np.linalg.norm(A) ** 2
    if norm_sq == 0.0:
        return 0.0
    return float(col_var.sum() / norm_sq)

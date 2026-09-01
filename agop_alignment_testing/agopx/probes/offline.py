"""The paper's two a posteriori baselines (project.md Section 3, "The two baseline
measures"). Both are causal=False: AGOPAlignment needs M* (the endpoint), and
CirculantDeviation needs the discrete-log reordering, which needs to know Z*_p is
cyclic of order p-1 in advance.

Both probes are stateless during the run -- they only need the full trajectory, which
evaluate.py's harness (Phase 3) will supply to finalize(). update() is a no-op.

Both take a `source` selecting which matrix to read out of each Snapshot
(features.select_matrix). RFM runs use the default "M". NN runs should use
"sqrt_agop": the paper computes the Fig. 5B progress measures on the square root of
the network's AGOP, while Snapshot.M is the NFM.
"""
from __future__ import annotations

import numpy as np

from ..features import (
    apply_circulant_transform,
    circulant_transform_for_operation,
    offdiag_block,
    select_matrix,
)
from .base import Probe
from .registry import register

# How much of the 2p x 2p feature matrix AGOPAlignment vectorises before taking the
# cosine similarity. See AGOPAlignment's docstring for why the default is not the
# paper's literal choice.
ALIGNMENT_REGIONS = ("offdiag_block", "no_diag", "full")


def _alignment_vector(M: np.ndarray, p: int, region: str) -> np.ndarray:
    if region == "offdiag_block":
        return offdiag_block(M, p).ravel()
    if region == "no_diag":
        M = M.copy()
        np.fill_diagonal(M, 0.0)
        return M.ravel()
    if region == "full":
        return M.ravel()
    raise ValueError(f"unknown alignment region {region!r}; have {list(ALIGNMENT_REGIONS)}")


@register("agop_alignment")
class AGOPAlignment(Probe):
    """rho(M_t, M*) = cosine similarity of vectorised M_t and the final M*.

    The paper's Eq. 8 is unambiguously the *full* vectorised matrix: "let A~, B~ in
    R^{d^2} denote the vectorization of A and B". This implementation defaults to
    the bottom-left p x p block instead (region="offdiag_block"), which is a
    deliberate deviation, not an oversight -- the `region` argument exists so the
    paper-literal version stays one keyword away and the gate check can plot all
    three side by side.

    Why deviate: the diagonal dominates M and swamps the structural signal
    (project.md's Phase 3 preprocessing note, which turned out to apply here too).
    Measured on a converged quadratic x+y run at p=61, the three regions give very
    different pictures of the same trajectory:

        t     full    no_diag   offdiag_block
        1    0.779     0.192        0.383
        10   0.922     0.852        0.863
        30   1.000     1.000        1.000

    M_0 = I is purely diagonal, so its off-diagonal block is exactly zero and this
    version starts at exactly 0.0 and rises to 1.0. The full-matrix version starts
    around 0.78 purely from diagonal-on-diagonal overlap with M*'s Observation-1
    c1*I + c2*11^T blocks -- not structural progress, and not the curve the paper
    plots, which starts near 0.2 in Fig. 2B. Note "no_diag" (full matrix with only
    the diagonal *entries* zeroed) lands closest to that 0.2; the reference repo
    zeroes diagonals the same way everywhere it visualises features (its
    `M_no_diag` images and `nfa_no_diag_corr` metric), so that may well be what
    produced the published figure. Left as an option rather than the default
    because the block version is the better-motivated choice for this project's own
    purposes.
    """

    causal = False

    def __init__(self, p: int, source: str = "M", region: str = "offdiag_block"):
        if region not in ALIGNMENT_REGIONS:
            raise ValueError(f"unknown alignment region {region!r}; have {list(ALIGNMENT_REGIONS)}")
        self.p = p
        self.source = source
        self.region = region

    def update(self, snap):
        return None

    def finalize(self, traj: list) -> dict:
        m_star = _alignment_vector(select_matrix(traj[-1], self.source), self.p, self.region)
        star_norm = np.linalg.norm(m_star)

        points = []
        for snap in traj:
            v = _alignment_vector(select_matrix(snap, self.source), self.p, self.region)
            v_norm = np.linalg.norm(v)
            cos = float(np.dot(v, m_star) / (v_norm * star_norm)) if v_norm > 0 else 0.0
            points.append({"t": snap.t, "agop_alignment": cos})
        return {"trajectory": points}


@register("circulant_deviation")
class CirculantDeviation(Probe):
    """D(A) over the bottom-left p x p block A of M_t (project.md Section 3):

        D(A) = (1 / ||A||_F^2) * sum_j Var( S(A)[:, j] )

    where S shifts row l of A right by l positions. Zero iff A is exactly circulant.

    One deliberate departure from the paper's written formula: the paper defines
    Var(v) = sum_j (v_j - Ev)^2, a *sum* of squared deviations, while _deviation
    below uses np.var, the *mean*. These differ by a factor of p. The mean is what
    reproduces the published figure -- on a converged quadratic x+y run at p=61 it
    peaks at 0.0102 and decays to 0.00032, matching Fig. 2B's ~0.01-to-~0 range,
    whereas the literal sum peaks at 0.62. So the paper's formula as written is off
    by p from the quantity it plots; this implementation follows the figure.

    Which index relabeling reveals that circulant structure is operation-dependent
    (features.CIRCULANT_TRANSFORMS): none for x+y; a column reversal for x-y;
    discrete-log reorder for x*y; both for x/y. Only "reorder" (mul/div) is
    documented in project.md Section 3. The "reverse" entries are not an empirical
    hack: the paper's footnote 4 says feature matrices "may also be block Hankel
    matrices which are constant on anti-diagonals" and that it uses "circulant" to
    cover both, and a column reversal is exactly the Hankel-to-circulant map. So
    x-y and x/y land in the paper's Hankel case. Measured on converged quadratic
    runs, the chosen transform beats every alternative by ~50x:

        op    identity  reverse  reorder  reorder+reverse
        x+y    0.00032  0.01639  0.01596      0.01634
        x-y    0.01639  0.00030  0.01658      0.01634
        x*y    0.01596  0.01638  0.00029      0.01636
        x/y    0.01621  0.01621  0.01641      0.00026

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

    def __init__(self, p: int, operation: str, source: str = "M"):
        self.p = p
        self.operation = operation
        self.source = source
        self.transform = circulant_transform_for_operation(operation)

    def update(self, snap):
        return None

    def finalize(self, traj: list) -> dict:
        points = []
        for snap in traj:
            A = offdiag_block(select_matrix(snap, self.source), self.p)
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
    # np.var is the MEAN squared deviation; the paper's Var() is the sum. See the
    # class docstring -- the mean is what reproduces the published figure's scale.
    col_var = shifted.var(axis=0)
    norm_sq = np.linalg.norm(A) ** 2
    if norm_sq == 0.0:
        return 0.0
    return float(col_var.sum() / norm_sq)

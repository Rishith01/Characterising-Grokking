"""Preprocessing shared by every probe (project.md Section 6, "Supporting modules").
Centralising this prevents the same subtle preprocessing bug from being reintroduced
in five places.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

from .kernels import matrix_power_psd


def offdiag_block(M: np.ndarray, p: int) -> np.ndarray:
    """The bottom-left p x p block of a 2p x 2p feature matrix (project.md Section 3,
    "Circulant deviation": "With A the bottom-left p x p block of M..."; also the
    block Phase 3 differences on, since the diagonal block dominates M and swamps
    the structural signal).
    """
    assert M.shape == (2 * p, 2 * p)
    return M[p:2 * p, 0:p]


def frobenius_normalize(M: np.ndarray) -> np.ndarray:
    """Normalise before differencing trajectories, else trace growth from the
    s = 1/2 AGOP power gets measured instead of direction.
    """
    norm = np.linalg.norm(M)
    if norm == 0.0:
        return M
    return M / norm


def _smallest_primitive_root(p: int) -> int:
    """Smallest generator of the cyclic group Z*_p (order p - 1)."""
    target = set(range(1, p))
    for g in range(2, p):
        seen = set()
        x = 1
        for _ in range(p - 1):
            x = (x * g) % p
            seen.add(x)
        if seen == target:
            return g
    raise ValueError(f"no primitive root found for p={p} -- is p prime?")


def _generator_powers(p: int, g: int) -> np.ndarray:
    """idx[i] = g^i mod p for i = 1..p-1, idx[0] = 0."""
    idx = np.zeros(p, dtype=np.int64)
    x = 1
    for i in range(1, p):
        x = (x * g) % p
        idx[i] = x
    return idx


def discrete_log_reorder(A: np.ndarray, p: int) -> np.ndarray:
    """Reorder a p x p matrix's rows/cols 1..p-1 by discrete log base the smallest
    generator g of Z*_p: new[i, j] = old[g^i mod p, g^j mod p] (project.md Section 3,
    "Reordering (Appendix C)"). Row/col 0 is left fixed -- it is identically zero in
    the matrices this is applied to, so its placement doesn't matter. Only meaningful
    for mul/div; Lemma G.1 guarantees the choice of generator is irrelevant, so the
    convention is the smallest one.

    Independent reimplementation from the paper's Appendix C description (not ported
    from nmallinar/rfm-grokking's utils.py), algebraically cross-checked to agree with
    that file's E.T @ M @ E construction: their lg_idx[i] is our discrete log of i
    (phi_g), and E's column c is a one-hot at row g^c, so (E.T M E)[a,b] = M[g^a, g^b]
    -- exactly the formula implemented here.
    """
    assert A.shape == (p, p)
    g = _smallest_primitive_root(p)
    idx = _generator_powers(p, g)
    return A[np.ix_(idx, idx)]


def _discrete_log_table(p: int) -> np.ndarray:
    """dlog[v] = i such that g^i = v (mod p), for the smallest generator g;
    dlog[0] = 0. The functional inverse of _generator_powers's idx (idx[i] = g^i):
    dlog[idx[i]] = i by construction, so discrete_log_reorder and unorder undo
    each other exactly.
    """
    g = _smallest_primitive_root(p)
    dlog = np.zeros(p, dtype=np.int64)
    x = 1
    for i in range(1, p):
        x = (x * g) % p
        dlog[x] = i
    return dlog


def unorder(A: np.ndarray, p: int) -> np.ndarray:
    """Inverse of discrete_log_reorder: places a matrix defined in discrete-log
    index space back into normal (a, b) index space. Used to plant an oracle
    mult/div-generalizing structure (random_circulant_seed_matrix, below).
    """
    assert A.shape == (p, p)
    dlog = _discrete_log_table(p)
    return A[np.ix_(dlog, dlog)]


# Index relabeling that reveals circulant structure in the off-diagonal block, per
# operation. project.md Section 3 documents discrete-log reordering for mul/div;
# the entries for "-" and the extra reversal for "/" were derived empirically during
# Phase 1 (see CirculantDeviation's docstring in probes/offline.py) but have a clean
# group-theoretic reading: x - y = x + (-y), and negation mod p reverses the cyclic
# order of the nonzero residues, so subtraction reduces to addition under a column
# reversal. Division's data encoding (x/y -> (xy mod p, y, x), see data.py) already
# folds it into a multiplication-shaped problem, and empirically needs that same
# reversal on top of the discrete-log reorder -- consistent with the reference repo's
# gen_random_div_circulant applying an extra rot90 on top of its circulant generator.
CIRCULANT_TRANSFORMS = {
    "x+y": "identity",
    "x-y": "reverse",
    "x*y": "reorder",
    "x/y": "reorder+reverse",
}


def circulant_transform_for_operation(operation: str) -> str:
    if operation not in CIRCULANT_TRANSFORMS:
        raise ValueError(
            f"no known circulant transform for operation {operation!r}; have {list(CIRCULANT_TRANSFORMS)}"
        )
    return CIRCULANT_TRANSFORMS[operation]


def apply_circulant_transform(A: np.ndarray, p: int, transform: str) -> np.ndarray:
    """For mul/div ("reorder" in transform), index 0 has no discrete log (0 has no
    multiplicative inverse), so it must be excluded from anything done in log-space
    -- reversal included. Reversing the *full* p x p matrix would shift whatever
    junk sits at index 0 to index p-1 instead (verified empirically: after
    discrete_log_reorder alone, row 0 of a converged x/y run is ~0, i.e. genuinely
    meaningless -- but reversing the whole matrix moves that near-zero content to
    row/col p-1, not row/col 0). So reversal, when combined with reordering, is
    applied only to the inner (p-1) x (p-1) submatrix (indices 1..p-1), keeping
    index 0 consistently the one to exclude afterward (CirculantDeviation does
    exactly that). This matches the reference repo's own construction:
    gen_random_div_circulant's rot90 is applied to the (p-1) x (p-1) circulant
    before embedding it at [1:, 1:], never touching row/col 0.
    """
    if "reorder" in transform:
        A = discrete_log_reorder(A, p)
        if "reverse" in transform:
            A = A.copy()
            A[1:, 1:] = A[1:, 1:][:, ::-1]
        return A
    if "reverse" in transform:
        return A[:, ::-1]
    return A


def apply_inverse_circulant_transform(C: np.ndarray, p: int, transform: str) -> np.ndarray:
    """Inverse of apply_circulant_transform: given an already-circulant C, builds a
    matrix A such that apply_circulant_transform(A, p, transform) recovers C. Used
    to plant an oracle generalizing off-diagonal block per operation
    (random_circulant_seed_matrix). Mirrors apply_circulant_transform's inner-
    submatrix-only reversal for reorder-involving transforms.
    """
    if "reorder" in transform:
        if "reverse" in transform:
            C = C.copy()
            C[1:, 1:] = C[1:, 1:][:, ::-1]
        return unorder(C, p)
    if "reverse" in transform:
        return C[:, ::-1]
    return C


def random_circulant_seed_matrix(p: int, operation: str, rng: np.random.Generator) -> np.ndarray:
    """Plants a random 'oracle' 2p x 2p feature matrix with the operation's
    generalizing structure already baked in (project.md Phase 2, "Fast learners":
    random-circulant-transformed inputs, Eq. 9). Used as M_0 in place of I so RFM
    starts from (near) the generalizing solution and should reach high test
    accuracy almost immediately -- separating "measures a real signal" from
    "counts iterations" for a probe.

    Independent reimplementation of nmallinar/rfm-grokking's
    train_random_circulant_kernel.py per-operation M construction (not ported): a
    random mean-centered circulant seeds the off-diagonal block, inverse-transformed
    per operation by whichever relabeling CirculantDeviation uses to *detect*
    circulant structure there (identity / reverse / reorder / reorder+reverse) --
    guaranteeing this plant is self-consistent with this codebase's own circulant
    convention, rather than assuming the reference repo's rot90-based construction
    uses the same index orientation. The diagonal blocks are set to the Observation-1
    projector I - (1/p) 11^T (empirically confirmed to match a converged x+y run's
    diagonal blocks in Phase 1, ~9% relative residual vs ~100% for a random matrix),
    and the whole thing is symmetrized and matrix-power-1/2'd, exactly like an
    ordinary AGOP update.

    For mul/div ("reorder" in transform), index 0 has no discrete log, so the
    circulant is built only over the (p-1) meaningful indices and embedded with
    row/col 0 held at zero -- matching the reference repo's own
    gen_random_mult_circulant/gen_random_div_circulant, which build a (p-1) x (p-1)
    circulant.circulant(row) and place it at C[1:, 1:], never touching row/col 0.
    Building a full p-length circulant instead (as an earlier version of this
    function did) plants spurious content at the meaningless index, which measurably
    degraded the fast-learner mechanism for x*y/x/y (partial, slower-than-normal
    generalization instead of the near-immediate jump seen for x+y/x-y).
    """
    transform = circulant_transform_for_operation(operation)
    if "reorder" in transform:
        row = rng.random(p - 1)
        row -= row.mean()
        C = np.zeros((p, p))
        C[1:, 1:] = scipy.linalg.circulant(row)
    else:
        col = rng.random(p)
        col -= col.mean()
        C = scipy.linalg.circulant(col)

    off_diag = apply_inverse_circulant_transform(C, p, transform)

    M = np.zeros((2 * p, 2 * p))
    M[p : 2 * p, 0:p] = off_diag
    M[0:p, p : 2 * p] = off_diag.T
    diag_block = np.eye(p) - np.ones((p, p)) / p
    M[0:p, 0:p] = diag_block
    M[p : 2 * p, p : 2 * p] = diag_block

    return matrix_power_psd(M, 0.5)


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    """Smooth NN-learner trajectories over epochs; single-epoch AdamW increments
    are minibatch noise (project.md Section 3, Phase 3 preprocessing note).
    """
    raise NotImplementedError

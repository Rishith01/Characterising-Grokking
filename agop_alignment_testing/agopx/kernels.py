"""Kernel functions and their AGOP updates (project.md Section 3, "RFM algorithm").

RFM loop:
    M_0 = I_d
    alpha   = k(X, X; M)^{-1} y                 # ridgeless kernel regression
    f(x)    = k(x, X; M) alpha
    M_next  = [G(f)]^s                          # AGOP, matrix power s = 1/2

AGOP: G(f) = (1/n) sum_j (J_f(x_j) - Jbar)(J_f(x_j) - Jbar)^T, where J_f(x_j) in
R^{d x p} has column c equal to the gradient of output component c of f wrt the
input, evaluated at training point x_j (j ranges over the n training points), and
Jbar = (1/n) sum_j J_f(x_j) is the mean Jacobian across training points.

The mean-centering is NOT in project.md Section 3's formula (which states the plain,
uncentered G(f) = (1/n) sum_j J_f(x_j) J_f(x_j)^T) -- it was found by reading
nmallinar/rfm-grokking's actual training loop (train_kernel.py calls
update(..., centering=True, ...) unconditionally, for every kernel type) after the
Gaussian kernel was found to plateau at ~50-60% test accuracy on x+y regardless of
bandwidth (a genuine converged fixed point, not under-training -- swept bandwidth
0.5 to 50, and ran 60 iterations to confirm M itself had stopped changing by t~30).
Adding centering fixed it: 98%+ by t=24 and still climbing, vs ~42% uncentered at the
same t. It made no visible difference to quadratic, which already grokked cleanly
either way, but the reference repo applies it uniformly, so this implementation now
does too. Centering doesn't change AGOPAlignment/CirculantDeviation's Phase 1
conclusions for quadratic (re-verified after this change), but the earlier
uncentered runs (Phase 1) are numerically superseded -- see run directories with a
'_centered' or later timestamp for the corrected versions.

Both kernels expand G(f) algebraically into O(n^3) matrix products (dominated by one
n x n matmul) instead of ever forming the (n, d, p) per-sample Jacobian tensor, which
costs O(n^2 d p) -- ~10x more expensive at the p=61 scale used here. Centering is
folded into the same closed form via the identity
(1/n) sum_j (J_j - Jbar)(J_j - Jbar)^T = (1/n) sum_j J_j J_j^T - Jbar Jbar^T, i.e. the
uncentered closed form minus the outer product of the mean Jacobian (itself computed
via a similar O(n^2)-style shortcut, not by forming per-sample Jacobians). Each
closed form -- uncentered and the Jbar correction -- was derived by hand and checked
against a direct, unoptimized transcription of the per-sample definition (explicit
double loop over i, j) on a random tiny (n=6, d=4, p=3) problem before use; see the
reasoning trail in the project history for those derivations.
"""
from __future__ import annotations

import numpy as np


def matrix_power_psd(G: np.ndarray, power: float) -> np.ndarray:
    """[G]^power for a symmetric PSD matrix, via eigendecomposition."""
    G = (G + G.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.clip(eigvals, 0.0, None)
    powered = np.power(eigvals, power)
    return (eigvecs * powered) @ eigvecs.T


class QuadraticKernel:
    """k(x, x'; M) = (x^T M x')^2."""

    name = "quadratic"

    def matrix(self, X1: np.ndarray, X2: np.ndarray, M: np.ndarray) -> np.ndarray:
        S = (X1 @ M) @ X2.T
        return S**2

    def agop(
        self, X: np.ndarray, M: np.ndarray, alpha: np.ndarray, K: np.ndarray, centering: bool = True
    ) -> np.ndarray:
        n = X.shape[0]
        XM = X @ M  # row i = (M x_i)^T, M assumed symmetric
        S = XM @ X.T  # S[j, i] = x_j^T M x_i
        A = alpha @ alpha.T
        G = (4.0 / n) * XM.T @ (A * (S.T @ S)) @ XM

        if centering:
            s_sum = S.sum(axis=1)  # (n,)
            jac_mean = (2.0 / n) * (XM * s_sum[:, None]).T @ alpha  # (d, p)
            G = G - jac_mean @ jac_mean.T
        return G


class GaussianKernel:
    """k(x, x'; M) = exp(-||x - x'||_M^2 / L), ||x - x'||_M^2 = (x-x')^T M (x-x')."""

    name = "gaussian"

    def __init__(self, bandwidth: float = 2.5):
        self.L = bandwidth

    def matrix(self, X1: np.ndarray, X2: np.ndarray, M: np.ndarray) -> np.ndarray:
        n1 = np.einsum("ij,jk,ik->i", X1, M, X1)
        n2 = np.einsum("ij,jk,ik->i", X2, M, X2)
        cross = (X1 @ M) @ X2.T
        D = n1[:, None] - 2.0 * cross + n2[None, :]
        np.clip(D, 0.0, None, out=D)
        return np.exp(-D / self.L)

    def agop(
        self, X: np.ndarray, M: np.ndarray, alpha: np.ndarray, K: np.ndarray, centering: bool = True
    ) -> np.ndarray:
        n = X.shape[0]
        XM = X @ M  # row i = M x_i

        B = alpha.T @ K  # (p, n), B[:, j] = sum_i alpha[i,:] K[i,j]
        v = np.sum(B**2, axis=0)  # (n,)
        A = alpha @ alpha.T  # (n, n)
        Q = alpha @ B  # (n, n)

        T1 = (X * v[:, None]).T @ X
        T4 = X.T @ (A * (K @ K)) @ X
        T2 = X.T @ (K * Q).T @ X

        H = (T1 - T2 - T2.T + T4) / n
        G = (4.0 / self.L**2) * M @ H @ M

        if centering:
            k_sum = K.sum(axis=1)  # (n,)
            V = k_sum[:, None] * XM - K @ XM  # (n, d)
            jac_mean = (2.0 / (self.L * n)) * V.T @ alpha  # (d, p)
            G = G - jac_mean @ jac_mean.T
        return G


KERNELS = {
    "quadratic": QuadraticKernel,
    "gaussian": GaussianKernel,
}


def build_kernel(kernel_type: str, **kwargs):
    if kernel_type not in KERNELS:
        raise ValueError(f"unknown kernel {kernel_type!r}; have {list(KERNELS)}")
    return KERNELS[kernel_type](**kwargs)

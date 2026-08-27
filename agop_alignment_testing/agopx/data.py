"""Modular-arithmetic task encoding (project.md Section 3, "Task encoding").

f*(a,b) = g(a,b) mod p. Input is e_a (+) e_b in R^{2p}, label is e_{f*(a,b)} in R^p.
"""
from __future__ import annotations

import numpy as np

# (x, y, p) -> (a, b, z) where z = g(a,b) is reduced mod p by the caller.
_DIVISION_OPS = {
    "x/y": lambda x, y, p: (x * y % p, y, x),
}

_OPS = {
    "x+y": lambda x, y, p: (x, y, x + y),
    "x-y": lambda x, y, p: (x, y, x - y),
    "x*y": lambda x, y, p: (x, y, x * y),
    **_DIVISION_OPS,
    "x^2+y": lambda x, y, p: (x, y, x**2 + y),
    "x^2+y^2": lambda x, y, p: (x, y, x**2 + y**2),
    "x^2+xy+y^2": lambda x, y, p: (x, y, x**2 + x * y + y**2),
}


def operation_mod_p_data(operation: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    """All (a, b) -> f*(a,b) triples for one operation.

    Returns:
        inputs: (N, 2) int array of (a, b) pairs.
        labels: (N,) int array of f*(a,b) in [0, p).
    """
    if operation not in _OPS:
        raise ValueError(f"unknown operation {operation!r}; have {list(_OPS)}")

    x = np.arange(0, p)
    y = np.arange(1, p) if operation in _DIVISION_OPS else np.arange(0, p)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xx, yy = xx.reshape(-1), yy.reshape(-1)

    a, b, z = _OPS[operation](xx, yy, p)
    labels = np.mod(z, p)
    inputs = np.stack([a, b], axis=1)
    return inputs, labels


def make_data_splits(
    inputs: np.ndarray, labels: np.ndarray, training_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = inputs.shape[0]
    train_size = int(training_fraction * n)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:train_size], perm[train_size:]
    return inputs[train_idx], labels[train_idx], inputs[test_idx], labels[test_idx]


def one_hot_pairs(pairs: np.ndarray, p: int) -> np.ndarray:
    """(N, 2) int pairs -> (N, 2p) float64, e_a concatenated with e_b."""
    n = pairs.shape[0]
    out = np.zeros((n, 2 * p), dtype=np.float64)
    out[np.arange(n), pairs[:, 0]] = 1.0
    out[np.arange(n), p + pairs[:, 1]] = 1.0
    return out


def one_hot_labels(labels: np.ndarray, p: int) -> np.ndarray:
    n = labels.shape[0]
    out = np.zeros((n, p), dtype=np.float64)
    out[np.arange(n), labels] = 1.0
    return out

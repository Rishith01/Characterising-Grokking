"""RFM learner (project.md Section 3, "RFM algorithm").

Yields a Snapshot(t, M_t, metrics_t) for t = 0..iters, where metrics_t is the
train/test performance of the ridgeless kernel regressor using M_t (so every
snapshot is internally consistent: the M and the metrics next to it came from
the same solve). M_{t+1} is obtained from M_t via one AGOP update.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ..data import make_data_splits, one_hot_labels, one_hot_pairs, operation_mod_p_data
from ..features import random_circulant_seed_matrix
from ..kernels import build_kernel, matrix_power_psd
from .base import Snapshot


@dataclass
class RFMConfig:
    operation: str = "x+y"
    p: int = 61
    training_fraction: float = 0.5
    iters: int = 30
    kernel_type: str = "quadratic"
    bandwidth: float = 2.5  # gaussian kernel only
    agop_power: float = 0.5
    # See kernels.py's module docstring: matches nmallinar/rfm-grokking's actual
    # training loop, which centers per-sample Jacobians before the AGOP outer-product
    # sum. Needed for the gaussian kernel to grok at all; makes no visible difference
    # to quadratic.
    centering: bool = True
    seed: int = 0
    # Non-grokker control (project.md Phase 2): independently reassign every train
    # and test label to a uniformly random value in [0, p), breaking any
    # relationship between input and label. A run that still "groks" this would
    # mean a probe is just counting iterations, not measuring real structure.
    random_labels: bool = False
    # Fast-learner control (project.md Phase 2): seed M_0 with an oracle
    # generalizing structure (features.random_circulant_seed_matrix) instead of I,
    # so the run should reach high test accuracy almost immediately.
    fast_learner: bool = False


class RFMLearner:
    """Recursive Feature Machine: ridgeless kernel regression + AGOP feature update."""

    def __init__(self, config: RFMConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)

        inputs, labels = operation_mod_p_data(config.operation, config.p)
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = make_data_splits(
            inputs, labels, config.training_fraction, rng
        )
        if config.random_labels:
            y_tr_raw = rng.integers(0, config.p, size=y_tr_raw.shape)
            y_te_raw = rng.integers(0, config.p, size=y_te_raw.shape)

        self.X_tr = one_hot_pairs(X_tr_raw, config.p)
        self.y_tr = one_hot_labels(y_tr_raw, config.p)
        self.X_te = one_hot_pairs(X_te_raw, config.p)
        self.y_te = one_hot_labels(y_te_raw, config.p)

        kernel_kwargs = {"bandwidth": config.bandwidth} if config.kernel_type == "gaussian" else {}
        self.kernel = build_kernel(config.kernel_type, **kernel_kwargs)
        self.d = self.X_tr.shape[1]

        if config.fast_learner:
            self.M0 = random_circulant_seed_matrix(config.p, config.operation, rng)
        else:
            self.M0 = np.eye(self.d, dtype=np.float64)

    def steps(self) -> Iterator[Snapshot]:
        M = self.M0.copy()

        for t in range(self.config.iters + 1):
            K_train = self.kernel.matrix(self.X_tr, self.X_tr, M)
            alpha = np.linalg.solve(K_train, self.y_tr)
            train_metrics = _eval(K_train, alpha, self.y_tr)

            K_test = self.kernel.matrix(self.X_te, self.X_tr, M)
            test_metrics = _eval(K_test, alpha, self.y_te)

            metrics = {
                "train/accuracy": train_metrics["accuracy"],
                "train/loss": train_metrics["loss"],
                "test/accuracy": test_metrics["accuracy"],
                "test/loss": test_metrics["loss"],
                "trace_M": float(np.trace(M)),
            }
            yield Snapshot(t=t, M=M.copy(), metrics=metrics)

            if t == self.config.iters:
                break

            G = self.kernel.agop(self.X_tr, M, alpha, K_train, centering=self.config.centering)
            M = matrix_power_psd(G, self.config.agop_power)


def _eval(K: np.ndarray, alpha: np.ndarray, y_onehot: np.ndarray) -> dict:
    preds = K @ alpha
    loss = float(np.mean((preds - y_onehot) ** 2))
    acc = float(np.mean(preds.argmax(-1) == y_onehot.argmax(-1)))
    return {"accuracy": acc, "loss": loss}

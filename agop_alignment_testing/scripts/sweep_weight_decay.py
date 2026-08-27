"""Sweep weight_decay for the NN learner (x+y, p=61) to find a value that both
groks per project.md Appendix B's intent and avoids the post-convergence circulant-
deviation drift found at weight_decay=1.0 (project history: wd=1.0 causes deviation
to dip then rise back up after grokking; wd=0 fixes it but deviates from spec and
never quite reaches 100% test accuracy).

Usage: python -m scripts.sweep_weight_decay
"""
from __future__ import annotations

import numpy as np

from agopx.features import offdiag_block
from agopx.learners.neural import NeuralConfig, NNLearner
from agopx.probes.offline import _deviation

WEIGHT_DECAYS = [0.003, 0.01, 0.03, 0.1, 0.3]

if __name__ == "__main__":
    for wd in WEIGHT_DECAYS:
        config = NeuralConfig(
            operation="x+y", p=61, training_fraction=0.5, width=1024,
            epochs=50, batch_size=32, lr=1e-3, weight_decay=wd,
            snapshot_every=5, seed=0,
        )
        learner = NNLearner(config)

        devs = []
        for snap in learner.steps():
            A = offdiag_block(snap.M.astype(np.float64), 61)
            dev = _deviation(A)
            devs.append((snap.t, dev, snap.metrics["test/accuracy"], snap.metrics["trace_M"]))

        # summarize: min deviation, where it occurs, and final deviation (rise = bad)
        ts, ds, accs, traces = zip(*devs)
        min_idx = int(np.argmin(ds))
        print(f"wd={wd}")
        for t, d, acc, tr in devs:
            print(f"  t={t:3d}  dev={d:.6f}  test_acc={acc:.4f}  trace={tr:.1f}")
        print(f"  -> min dev {ds[min_idx]:.6f} at t={ts[min_idx]}, final dev {ds[-1]:.6f}, "
              f"final test_acc {accs[-1]:.4f}, rise-from-min {ds[-1]-ds[min_idx]:.6f}")
        print()

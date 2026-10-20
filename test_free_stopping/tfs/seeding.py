"""Named, independent random streams derived from one integer seed (DESIGN.md §4).

Every source of randomness in the project draws from one of these streams, never from a
global RNG. Changing how one stream is consumed therefore never shifts the others, and
the whole run is regenerable from (config, seed).
"""
from __future__ import annotations

import numpy as np
import torch

# Fixed stream indices: append new streams at the end, never reorder.
STREAMS = ("data", "teacher", "init", "order", "probe", "fingerprint")


class SeedStreams:
    def __init__(self, seed: int):
        self.seed = seed
        children = np.random.SeedSequence(seed).spawn(len(STREAMS))
        self._seqs = dict(zip(STREAMS, children))

    def numpy(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self._seqs[name])

    def int_seed(self, name: str) -> int:
        """A 63-bit integer derived from the stream, for torch.Generator.manual_seed."""
        return _to_int(self._seqs[name])

    def torch(self, name: str) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.int_seed(name))
        return g


def batch_indices(order_seed: int, step: int, n: int, batch_size: int) -> torch.Tensor:
    """Minibatch indices for a given optimizer step, as a pure function of the step.

    Each epoch's permutation is derived from (order_seed, epoch), so the batch order
    carries no mutable RNG state: restoring a keyframe needs only the step number, and
    replay cannot drift through RNG bookkeeping.
    """
    steps_per_epoch = -(-n // batch_size)
    epoch, pos = divmod(step, steps_per_epoch)
    g = torch.Generator(device="cpu")
    g.manual_seed(_to_int(np.random.SeedSequence([order_seed, epoch])))
    perm = torch.randperm(n, generator=g)
    return perm[pos * batch_size : (pos + 1) * batch_size]


def _to_int(seq: np.random.SeedSequence) -> int:
    return int(seq.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))

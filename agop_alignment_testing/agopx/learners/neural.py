"""NN learner (project.md Section 3, "Neural network config" / Section 6).

One hidden layer, no biases, quadratic activation sigma(z) = z^2, width 1024,
AdamW, batch size 32, lr 1e-3, MSE loss, standard PyTorch init, p = 61, training
fraction 50%, 50 epochs.

Snapshot.M is the NFM = W_1^T W_1 -- the NN's structural analogue of the kernel-RFM's
M_t (project.md: "RFM iterations and NN epochs are both just a stream of feature
matrices"), which is what lets AGOPAlignment/CirculantDeviation run on NN snapshots
unmodified. Snapshot.agop is the network's own AGOP, kept alongside so the NFM-vs-
sqrt(AGOP) sanity check (project.md Section 3: Pearson > 0.92) can be computed
directly from a run's saved snapshots.

Weight decay: matches Appendix B's literal value of 1.0. Phase 1 spent a long detour
on this: at 50 epochs, circulant deviation on the NN dips near the grokking
transition then rises back up for the rest of training, unlike RFM's clean monotonic
convergence. That looked like an artifact of weight decay never letting the network
settle (RFM's M_t is a fixed-point iteration that stops changing once converged; AdamW
decay keeps shrinking weights regardless of task gradient). Lowering weight_decay
"fixed" the monotonicity, but checking against the reference repo's own actual NN
check (nfa_no_diag_corr in nmallinar/rfm-grokking's train_net.py -- NFM-vs-sqrt(AGOP)
correlation on the off-diagonal only, which is more meaningful than the full-matrix
version since the diagonal trivially inflates it) showed the opposite: at wd=1.0 that
correlation recovers and stabilizes above the paper's 0.92 bar given enough epochs
(the reference repo's own default is 1000, not project.md's 50), while at the lowered
weight_decay it degrades instead. Extending wd=1.0 to 300 epochs also showed circulant
deviation itself never converges either way -- it wanders in a band regardless of
weight decay or training length, unlike RFM. Conclusion: circulant deviation is not a
reliable NN-side check (it isn't an implemented/verified paper metric for the NN case
either -- grep of the reference repo found no circulant-deviation code anywhere, only
imshow heatmap logging), so it's treated as informative for RFM only going forward.
Weight decay is kept at the paper's literal 1.0 rather than tuned further.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data import make_data_splits, one_hot_labels, one_hot_pairs, operation_mod_p_data
from .base import Learner, Snapshot


@dataclass
class NeuralConfig:
    operation: str = "x+y"
    p: int = 61
    training_fraction: float = 0.5
    width: int = 1024
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1.0
    # Decoupled weight decay on W_1 is exactly gradient descent on trace(NFM) =
    # trace(W_1^T W_1) = ||W_1||_F^2. w2_weight_decay lets W_2's decay be set
    # independently of W_1's; defaults to weight_decay (both layers decay uniformly,
    # matching plain AdamW). A Phase 1 ablation found decoupling it made no
    # difference to the circulant-deviation drift investigated there (see module
    # docstring) -- kept as a knob since it's cheap, but not used by default.
    w2_weight_decay: Optional[float] = None
    snapshot_every: int = 1
    seed: int = 0
    device: str = "cpu"  # "cuda" to train on GPU; snapshot/AGOP extraction stays on CPU either way
    random_labels: bool = False  # non-grokker control, see RFMConfig


class _QuadNet(nn.Module):
    """f(x) = W_2 (W_1 x)^2, no biases -- standard PyTorch (kaiming-uniform) init."""

    def __init__(self, d: int, h: int, p: int):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, p, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) ** 2)


class NNLearner(Learner):
    def __init__(self, config: NeuralConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        rng = np.random.default_rng(config.seed)

        inputs, labels = operation_mod_p_data(config.operation, config.p)
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw = make_data_splits(
            inputs, labels, config.training_fraction, rng
        )
        if config.random_labels:
            y_tr_raw = rng.integers(0, config.p, size=y_tr_raw.shape)
            y_te_raw = rng.integers(0, config.p, size=y_te_raw.shape)

        self.X_tr = torch.from_numpy(one_hot_pairs(X_tr_raw, config.p)).float().to(self.device)
        self.y_tr = torch.from_numpy(one_hot_labels(y_tr_raw, config.p)).float().to(self.device)
        self.X_te = torch.from_numpy(one_hot_pairs(X_te_raw, config.p)).float().to(self.device)
        self.y_te = torch.from_numpy(one_hot_labels(y_te_raw, config.p)).float().to(self.device)

        d = self.X_tr.shape[1]
        self.model = _QuadNet(d, config.width, config.p).to(self.device)
        w2_wd = config.weight_decay if config.w2_weight_decay is None else config.w2_weight_decay
        self.opt = torch.optim.AdamW(
            [
                {"params": self.model.w1.parameters(), "weight_decay": config.weight_decay},
                {"params": self.model.w2.parameters(), "weight_decay": w2_wd},
            ],
            lr=config.lr,
            betas=(0.9, 0.98),  # matches train_net.py; PyTorch's default is (0.9, 0.999)
        )

    def steps(self) -> Iterator[Snapshot]:
        cfg = self.config
        n = self.X_tr.shape[0]
        perm_rng = np.random.default_rng(cfg.seed + 1)

        for epoch in range(cfg.epochs + 1):
            if epoch % cfg.snapshot_every == 0 or epoch == cfg.epochs:
                yield self._snapshot(epoch)

            if epoch == cfg.epochs:
                break

            perm = torch.from_numpy(perm_rng.permutation(n)).to(self.device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                xb, yb = self.X_tr[idx], self.y_tr[idx]
                self.opt.zero_grad()
                pred = self.model(xb)
                loss = torch.mean((pred - yb) ** 2)
                loss.backward()
                self.opt.step()

    def _snapshot(self, epoch: int) -> Snapshot:
        with torch.no_grad():
            W1 = self.model.w1.weight.detach().cpu().numpy().astype(np.float64)  # (h, d)
            W2 = self.model.w2.weight.detach().cpu().numpy().astype(np.float64)  # (p, h)
            nfm = W1.T @ W1

            train_metrics = _eval(self.model, self.X_tr, self.y_tr)
            test_metrics = _eval(self.model, self.X_te, self.y_te)

        # AGOP extraction is a one-shot O(n h^2) numpy op per snapshot (see
        # kernels.py's docstring for why this closed-form beats per-sample
        # Jacobians) -- cheap enough already that it stays on CPU regardless of
        # where training ran, per project.md: "this is a CPU workload."
        agop = _quad_net_agop(self.X_tr.detach().cpu().numpy().astype(np.float64), W1, W2)

        metrics = {
            "train/accuracy": train_metrics["accuracy"],
            "train/loss": train_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/loss": test_metrics["loss"],
            "trace_M": float(np.trace(nfm)),
        }
        return Snapshot(t=epoch, M=nfm, metrics=metrics, agop=agop)


def _eval(model: nn.Module, X: torch.Tensor, y_onehot: torch.Tensor) -> dict:
    with torch.no_grad():
        preds = model(X)
        loss = float(torch.mean((preds - y_onehot) ** 2))
        acc = float((preds.argmax(-1) == y_onehot.argmax(-1)).float().mean())
    return {"accuracy": acc, "loss": loss}


def _quad_net_agop(X: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """G(f) for f(x) = W_2 (W_1 x)^2, derived by hand and checked against
    torch.autograd.functional.jacobian on a tiny random network before use (see the
    project history for that verification) -- same O(n h^2) algebraic-shortcut style
    as kernels.py, avoiding the O(n h d) per-sample Jacobian tensor.
    """
    n = X.shape[0]
    Z = X @ W1.T  # (n, h)
    B = W2.T @ W2  # (h, h)
    return (4.0 / n) * W1.T @ (B * (Z.T @ Z)) @ W1

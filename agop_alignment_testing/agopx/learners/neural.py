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

Update to that conclusion: part of the discrepancy was a measurement bug, not the
network. The paper computes the Fig. 5B progress measures on the square root of the
AGOP, not on the NFM -- "Given that the square root of the AGOP of neural networks
exhibits block-circulant structure, we can use circulant deviation and AGOP alignment
to measure gradual progress" -- while the probes were reading Snapshot.M, which is
the NFM. Measured on a 50-epoch x+y run, circulant deviation over the trajectory
falls 0.0162 -> 0.0125 on the NFM but 0.0162 -> 0.0095 on sqrt(AGOP), i.e. roughly
twice the descent, and closer to Fig. 5B's 0.015 -> 0.005. Probes now take a `source`
argument (features.select_matrix) and corpus entries for NN runs set
probe_source="sqrt_agop". The reversal after the minimum persists on both matrices,
so this is a partial explanation, not a resolution -- but the residual gap should be
re-examined against sqrt(AGOP) and a longer horizon before drawing conclusions about
the network.
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
    # "adamw" is the paper's Appendix B spec. "sgd" exists for the Appendix Fig. 5
    # no-regularization control, which is vanilla SGD (lr 1.0, batch 128, width 512,
    # weight decay 1e-5, 40% training fraction) run for 200k epochs -- AdamW with
    # weight_decay=0 is NOT that control: it groks, just later (epoch ~25 instead of
    # ~14 at p=61, r=0.5), which is why it cannot serve as a non-grokker.
    optimizer: str = "adamw"
    betas: tuple = (0.9, 0.98)  # adamw only; matches train_net.py (PyTorch default is 0.999)
    momentum: float = 0.0  # sgd only
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

        # AGOP extraction needs the training inputs as float64 numpy every snapshot;
        # convert once rather than round-tripping off the device each time.
        self._X_tr_np = one_hot_pairs(X_tr_raw, config.p)

        d = self.X_tr.shape[1]
        self.model = _QuadNet(d, config.width, config.p).to(self.device)
        w2_wd = config.weight_decay if config.w2_weight_decay is None else config.w2_weight_decay
        param_groups = [
            {"params": self.model.w1.parameters(), "weight_decay": config.weight_decay},
            {"params": self.model.w2.parameters(), "weight_decay": w2_wd},
        ]
        if config.optimizer == "adamw":
            self.opt = torch.optim.AdamW(param_groups, lr=config.lr, betas=tuple(config.betas))
        elif config.optimizer == "sgd":
            self.opt = torch.optim.SGD(param_groups, lr=config.lr, momentum=config.momentum)
        else:
            raise ValueError(f"unknown optimizer {config.optimizer!r}; have ['adamw', 'sgd']")

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
        agop = _quad_net_agop(self._X_tr_np, W1, W2)

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

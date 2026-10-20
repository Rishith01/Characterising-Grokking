"""TrainLoop: the single implementation of an optimizer step, shared by Trainer and
ReplayEngine (DESIGN.md §6, "One code path").

Only training-side tensors enter here: no test data, so replay can be built from a
TrainContext without ever loading the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ..config import RunConfig
from ..losses import compute_loss
from ..models import MLP
from ..optim import build_optimizer, reg_spec
from ..seeding import SeedStreams, batch_indices
from .state import TrainState


@dataclass
class StepInfo:
    """Raw quantities from one update t -> t+1, kept only when the next step is logged.

    All per-layer lists are indexed by Linear layer (MLP.layer_params order). `w_prev`
    quantities refer to the weights *before* the update.
    """
    batch_loss: float
    g_norm: list[float]
    reg_g_norm: list[float]
    update_norm: list[float]
    update_radial: list[float]
    update_tangential: list[float]
    cos_g_w: list[float]


class NonFiniteLoss(RuntimeError):
    pass


class TrainLoop:
    def __init__(self, cfg: RunConfig, x_train: torch.Tensor, y_train: torch.Tensor,
                 kind: str, d_in: int, d_out: int, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.x_train = x_train.to(self.device)
        self.y_train = y_train.to(self.device)
        self.kind = kind
        self.n = self.x_train.shape[0]

        streams = SeedStreams(cfg.seed)
        m = cfg.model
        self.model = MLP(d_in, d_out, m.depth, m.width, m.activation, m.init_scale,
                         generator=streams.torch("init")).to(self.device)
        self.opt = build_optimizer(cfg.optim, self.model.parameters())
        self.reg = reg_spec(cfg.optim)
        self._order_seed = streams.int_seed("order")

    # ---- one update ----------------------------------------------------------------
    def step(self, t: int, want_info: bool) -> Optional[StepInfo]:
        """Perform the update that takes the state from step t to step t+1.

        Extra quantities (want_info) are read-only views on the gradient and the update;
        computing them never changes the trajectory.
        """
        if self.cfg.batch_size is None:
            xb, yb = self.x_train, self.y_train
        else:
            idx = batch_indices(self._order_seed, t, self.n, self.cfg.batch_size).to(self.device)
            xb, yb = self.x_train[idx], self.y_train[idx]

        self.opt.zero_grad(set_to_none=True)
        loss = compute_loss(self.cfg.loss, self.model(xb), yb, self.kind)
        loss.backward()
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"non-finite loss at step {t}")

        info = None
        if want_info:
            layers = self.model.layer_params()
            w_prev = [[p.detach().clone() for p in ps] for ps in layers]
            g_norm, reg_g_norm, cos_g_w = [], [], []
            gamma = self.reg.gamma if self.reg.kind == "coupled" else 0.0
            for ps, wp in zip(layers, w_prev):
                g = _flat([p.grad for p in ps])
                w = _flat(wp)
                g_norm.append(float(g.norm()))
                reg_g_norm.append(float(gamma * w.norm()))
                cos_g_w.append(_cos(g, w))

        self.opt.step()

        if want_info:
            upd_n, upd_r, upd_t = [], [], []
            with torch.no_grad():
                for ps, wp in zip(layers, w_prev):
                    w0 = _flat(wp)
                    dw = _flat([p.detach() for p in ps]) - w0
                    w0n = w0.norm()
                    radial = (dw @ w0) / w0n if w0n > 0 else torch.zeros((), device=dw.device)
                    tang = (dw - radial * w0 / w0n).norm() if w0n > 0 else dw.norm()
                    upd_n.append(float(dw.norm()))
                    upd_r.append(float(radial))
                    upd_t.append(float(tang))
            info = StepInfo(float(loss.detach()), g_norm, reg_g_norm, upd_n, upd_r, upd_t, cos_g_w)
        return info

    def advance(self, from_step: int, to_step: int,
                want_info: Callable[[int], bool] = lambda t: False,
                on_step: Optional[Callable[[int, Optional[StepInfo]], None]] = None) -> None:
        """Run updates from_step -> to_step. on_step(t_new, info) is called after each
        update, with info present when want_info(t_new) is true."""
        for t in range(from_step, to_step):
            info = self.step(t, want_info(t + 1))
            if on_step is not None:
                on_step(t + 1, info)

    # ---- state ---------------------------------------------------------------------
    def capture(self, step: int) -> TrainState:
        return TrainState.capture(step, self.model, self.opt)

    def restore(self, state: TrainState) -> None:
        state.restore(self.model, self.opt)


def _flat(ts) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))

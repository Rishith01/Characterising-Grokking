"""Train-side recording: telemetry, training-set predictions, weight fingerprints, dense
summaries (DESIGN.md §5).

Everything here runs under torch.no_grad() and uses no training RNG stream, so recording
never changes the trajectory. The Recorder sees training data only. The oracle (test)
evaluation lives in the Trainer, not here, which is what lets ReplayEngine reuse the
Recorder without loading test data.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from ..losses import compute_loss
from ..seeding import SeedStreams
from ..tasks import Task
from .loop import StepInfo, TrainLoop

PER_LAYER_FIELDS = ("w_norm", "g_norm", "reg_g_norm", "update_norm",
                    "update_radial", "update_tangential", "cos_g_w")


class Recorder:
    def __init__(self, loop: TrainLoop, task: Task, n_proj: int, seed: int,
                 n_probe: int = 8, probe_batch: int = 256):
        self.loop = loop
        self.task = task
        self.n_layers = len(loop.model.layers)
        streams = SeedStreams(seed)
        # Fixed probe minibatches for gradient-noise statistics, from their own stream:
        # the same batches at every summary step, never touching the training order.
        g = streams.torch("probe")
        b = min(probe_batch, loop.n)
        self._probes = [torch.randperm(loop.n, generator=g)[:b].to(loop.device) for _ in range(n_probe)]
        # Fixed random unit vectors per parameter tensor, regenerable from the seed.
        rng = streams.numpy("fingerprint")
        self._proj = []
        for _, p in loop.model.param_list():
            r = rng.normal(size=(n_proj, p.numel()))
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            self._proj.append(torch.tensor(r, dtype=torch.float64, device=loop.device))

    # ---- every log step --------------------------------------------------------------
    @torch.no_grad()
    def log(self, step: int, info: Optional[StepInfo]) -> tuple[dict, np.ndarray, dict]:
        lp = self.loop
        pred = lp.model(lp.x_train)
        row = {
            "step": step,
            "train_loss": float(compute_loss(lp.cfg.loss, pred, lp.y_train, lp.kind)),
            "train_acc": self.task.accuracy(pred, lp.y_train),
            "lr": float(lp.opt.param_groups[0]["lr"]),
            "nan_flag": 0,
        }
        for i, ps in enumerate(lp.model.layer_params()):
            row[f"l{i}_w_norm"] = float(torch.cat([p.reshape(-1) for p in ps]).norm())
            for name in PER_LAYER_FIELDS[1:]:
                row[f"l{i}_{name}"] = getattr(info, name)[i] if info is not None else math.nan
        preds = self.task.train_pred_record(pred, lp.y_train)
        return row, preds, self.fingerprint(step)

    @torch.no_grad()
    def fingerprint(self, step: int) -> dict:
        fp = {"step": step}
        for (name, p), r in zip(self.loop.model.param_list(), self._proj):
            v = p.detach().reshape(-1).to(torch.float64)
            key = name.replace(".", "_")
            fp[f"{key}_sum"] = float(v.sum())
            fp[f"{key}_sumsq"] = float(v @ v)
            for j, x in enumerate((r @ v).tolist()):
                fp[f"{key}_proj{j}"] = x
        return fp

    # ---- every summary step ----------------------------------------------------------
    @torch.no_grad()
    def summaries(self) -> dict[str, np.ndarray]:
        """Per layer: singular values of W, the spectrum of the training-set activation
        covariance (hidden layers), and gradient-noise statistics over the fixed probe
        minibatches. Computed in float64, stored as float32.

        Only spectra and scalars: O(width) numbers per layer. Phase 1 also stored W^T W
        here, which was O(width^2) per layer and made summaries 85-93% of every run's
        size (1.8 GB for a depth-8 MNIST run). Full matrices come from replay instead.
        """
        lp = self.loop
        out = {}
        for i, layer in enumerate(lp.model.layers):
            W = layer.weight.detach().to(torch.float64).cpu()
            out[f"sv/l{i}"] = torch.linalg.svdvals(W).numpy()
        for i, h in enumerate(lp.model.features(lp.x_train)):
            h = h.to(torch.float64).cpu()
            h = h - h.mean(0, keepdim=True)
            out[f"actspec/l{i}"] = (torch.linalg.svdvals(h) ** 2 / h.shape[0]).numpy()
        out.update(self._gradient_noise())
        return {k: v.astype(np.float32) for k, v in out.items()}

    def _gradient_noise(self) -> dict[str, np.ndarray]:
        """gnoise/l{i} = [‖mean g‖², tr Cov(g), mean ‖g‖²] of the per-probe-batch data-loss
        gradient for layer i, in float64.

        Uses torch.autograd.grad, so p.grad and the optimizer are never touched.
        """
        lp = self.loop
        layers = lp.model.layer_params()
        sums = [None] * len(layers)
        sq_norms = [0.0] * len(layers)
        for idx in self._probes:
            with torch.enable_grad():
                loss = compute_loss(lp.cfg.loss, lp.model(lp.x_train[idx]), lp.y_train[idx], lp.kind)
                grads = torch.autograd.grad(loss, [p for ps in layers for p in ps])
            k = 0
            for i, ps in enumerate(layers):
                g = torch.cat([x.reshape(-1) for x in grads[k:k + len(ps)]]).to(torch.float64)
                k += len(ps)
                sums[i] = g if sums[i] is None else sums[i] + g
                sq_norms[i] += float(g @ g)
        n = len(self._probes)
        out = {}
        for i in range(len(layers)):
            mean = sums[i] / n
            mean_sq = float(mean @ mean)
            mean_norm_sq = sq_norms[i] / n
            out[f"gnoise/l{i}"] = np.array([mean_sq, mean_norm_sq - mean_sq, mean_norm_sq])
        return out

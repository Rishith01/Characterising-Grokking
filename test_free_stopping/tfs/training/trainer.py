"""Trainer: one run, config -> run directory (DESIGN.md §6)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..config import RunConfig
from ..losses import compute_loss
from ..seeding import SeedStreams
from ..storage import layout
from ..storage.writer import RunWriter
from ..tasks import get_task
from . import determinism
from .loop import NonFiniteLoss, TrainLoop
from .recorder import Recorder
from .schedule import Schedule

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Trainer:
    def __init__(self, cfg: RunConfig, runs_root: Path, device: str = "cpu", overwrite: bool = False):
        determinism.apply()
        self.cfg = cfg
        self.device = torch.device(device)
        self.task = get_task(cfg.task.name)
        data = self.task.build(cfg.task.params, SeedStreams(cfg.seed))
        self.kind = data.kind
        self.x_test = data.x_test.to(self.device)
        self.y_test = data.y_test.to(self.device)
        self.loop = TrainLoop(cfg, data.x_train, data.y_train, data.kind, data.d_in, data.d_out, self.device)
        self.recorder = Recorder(self.loop, self.task, cfg.record.n_fingerprint_proj, cfg.seed,
                                 cfg.record.n_probe, cfg.record.probe_batch)
        self.schedule = Schedule.build(cfg.steps, cfg.record)
        self.run_dir = layout.run_dir(runs_root, cfg.app, cfg.run_id)
        self.writer = RunWriter(self.run_dir, cfg, overwrite=overwrite)

    @torch.no_grad()
    def _oracle(self, step: int) -> dict:
        pred = self.loop.model(self.x_test)
        return {
            "step": step,
            "test_loss": float(compute_loss(self.cfg.loss, pred, self.y_test, self.kind)),
            "test_acc": self.task.accuracy(pred, self.y_test),
        }

    def run(self) -> Path:
        t0 = time.time()
        log_set, summary_set, key_set = self.schedule.sets()
        status, diverged_at, info = "complete", None, None
        for t in range(self.cfg.steps + 1):
            if t in key_set:
                self.writer.keyframe(self.loop.capture(t))
            if t in log_set:
                row, preds, fp = self.recorder.log(t, info)
                self.writer.log(row, preds, fp, self._oracle(t))
            if t in summary_set:
                self.writer.summary(t, self.recorder.summaries())
            if t == self.cfg.steps:
                break
            try:
                info = self.loop.step(t, want_info=(t + 1) in log_set)
            except NonFiniteLoss:
                status, diverged_at = "diverged", t
                break

        model = self.loop.model
        meta = {
            "diverged_at": diverged_at,
            "reg": self.loop.reg.to_dict(),
            "env": determinism.EnvRecord.capture(str(self.device)).to_dict(),
            "code": determinism.code_version(PROJECT_ROOT),
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "param_shapes": {n: list(p.shape) for n, p in model.param_list()},
            "wall_time_s": time.time() - t0,
            "versions": _versions(),
        }
        return self.writer.finalize(status, meta)


def _versions() -> dict:
    import pandas, pyarrow, safetensors  # noqa: E401

    return {"torch": torch.__version__, "numpy": np.__version__, "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__, "safetensors": safetensors.__version__}

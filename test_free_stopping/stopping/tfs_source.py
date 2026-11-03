"""The single adapter between `stopping` and the training code (DESIGN.md §13.1).

TfsRunHandle reads a run directory written by tfs (RunReader), regenerates the training
data from the seed (TrainContext) and replays models (ReplayEngine). DirRunSource serves
runs already on disk; HfRunSource downloads each run from an HF dataset folder such as
advay333/tfs:phase3/mnist, fetching only the parts the signals need.
"""
from __future__ import annotations

import shutil
from functools import cached_property
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfs.causal.view import TrainContext
from tfs.seeding import SeedStreams, batch_indices
from tfs.storage import layout
from tfs.storage.hub import HfRemote, PrefixRemote, Remote, pull_run, read_manifest
from tfs.storage.reader import RunReader
from tfs.training.replay import ReplayEngine
from tfs.training.schedule import Schedule

from .source import REQUIREMENTS, RunContext, RunHandle, RunSource, TrainData

# Requirement -> run parts to download (tfs.storage.layout.PARTS). Replay reads the
# telemetry for its log-step list, so "models" needs it too.
PARTS = {
    "telemetry": ("telemetry",),
    "preds": ("preds",),
    "summaries": ("summaries",),
    "models": ("keyframes", "telemetry"),
    "heldout": ("oracle",),
    "train_data": (),
}


def parts_for(needs: Iterable[str]) -> list[str]:
    needs = set(needs)
    unknown = needs - REQUIREMENTS
    if unknown:
        raise ValueError(f"unknown requirement(s) {sorted(unknown)}")
    return sorted({p for n in needs for p in PARTS[n]})


def _loss_parts(loss: str, kind: str, y: torch.Tensor, d_out: int):
    """(target, loss_fn) with loss_fn(pred, target) == tfs.losses.compute_loss(loss, pred, y).
    The one-hot target is built here, outside any vmap."""
    if loss == "ce":
        return y, lambda pred, t: F.cross_entropy(pred, t)
    if loss == "mse":
        t = F.one_hot(y, d_out).to(torch.float32) if kind == "classification" else y
        return t, lambda pred, tt: torch.mean((pred - tt) ** 2)
    raise ValueError(f"unknown loss {loss!r}")


class TfsRunHandle(RunHandle):
    def __init__(self, run_dir: Path, device: str = "cpu", verify_replay: bool = False,
                 allow_env_mismatch: bool = False, delete_on_close: bool = False):
        self.run_dir = Path(run_dir)
        self.reader = RunReader(self.run_dir)
        self.device = device
        self.verify_replay = verify_replay
        self.allow_env_mismatch = allow_env_mismatch
        self.delete_on_close = delete_on_close

    @cached_property
    def context(self) -> RunContext:
        cfg = self.reader.config
        sched = Schedule.build(cfg.steps, cfg.record)
        o = cfg.optim
        decay = "none" if o.weight_decay == 0 else ("decoupled" if o.name == "adamw" else "coupled")
        lr_eff = o.lr / (1 - o.momentum) if o.name == "sgd_momentum" else o.lr
        kind, d_in, d_out, n = self._shape
        return RunContext(
            run_id=cfg.run_id, app=cfg.app, config=cfg.to_dict(), optimizer=o.name, lr=o.lr,
            lr_eff=lr_eff, momentum=o.momentum, gamma=o.weight_decay, decay=decay, loss=cfg.loss,
            kind=kind, d_in=d_in, d_out=d_out, n_train=n, batch_size=cfg.batch_size, steps=cfg.steps,
            log_steps=sched.log_steps, summary_steps=sched.summary_steps, task_params=dict(cfg.task.params),
            device=self.device, _train_data=lambda: self._train_data)

    @cached_property
    def _tctx(self) -> TrainContext:
        return TrainContext.from_config(self.reader.config)

    @cached_property
    def _shape(self) -> tuple[str, int, int, int]:
        """(kind, d_in, d_out, n_train) from meta and the task params, without building
        the data (that needs the MNIST download)."""
        cfg, shapes = self.reader.config, self.reader.meta["param_shapes"]
        weights = [v for k, v in shapes.items() if k.endswith(".weight")]
        d_in, d_out = int(weights[0][1]), int(weights[-1][0])
        p = cfg.task.params
        if cfg.app == "modular":
            from tfs.tasks.modular import modular_pairs
            n = int(round(float(p["train_frac"]) * len(modular_pairs(p["op"], int(p["p"]))[2])))
        elif "n_train" in p:
            n = int(p["n_train"])
        else:
            n = int(self._tctx.x_train.shape[0])
        return ("regression" if cfg.app == "teacher_student" else "classification"), d_in, d_out, n

    @cached_property
    def _train_data(self) -> TrainData:
        t, cfg = self._tctx, self.reader.config
        x, y = t.x_train.to(self.device), t.y_train.to(self.device)
        target, loss_fn = _loss_parts(cfg.loss, t.kind, y, t.d_out)
        n, bs = x.shape[0], cfg.batch_size
        order = SeedStreams(cfg.seed).int_seed("order")

        def batch(step: int) -> Optional[torch.Tensor]:
            return None if bs is None else batch_indices(order, step, n, bs).to(self.device)
        return TrainData(x, y, target, loss_fn, batch)

    def telemetry(self) -> pd.DataFrame:
        return self.reader.telemetry()

    def train_preds(self) -> np.ndarray:
        return self.reader.train_preds()

    def summaries(self, step: int) -> dict[str, np.ndarray]:
        return self.reader.summaries(step)

    def models(self, steps: Iterable[int]) -> Iterator[tuple[int, torch.nn.Module]]:
        eng = ReplayEngine(self.run_dir, self.device, allow_env_mismatch=self.allow_env_mismatch)
        yield from eng.iterate(steps, verify=self.verify_replay)

    def heldout(self) -> pd.DataFrame:
        from .heldout import read_heldout  # the only path to test data (DESIGN.md §10)
        return read_heldout(self.run_dir)

    def close(self) -> None:
        if self.delete_on_close:
            shutil.rmtree(self.run_dir, ignore_errors=True)


class DirRunSource(RunSource):
    """Runs already on disk under runs_root/<app>/<run_id>/."""

    def __init__(self, runs_root: Path, device: str = "cpu", **handle_kw):
        self.root = Path(runs_root)
        self.device = device
        self.handle_kw = handle_kw

    def _dirs(self) -> dict[str, Path]:
        return {d.name: d for d in self.root.glob("*/*") if (d / layout.DONE).exists()}

    def run_ids(self) -> list[str]:
        return sorted(self._dirs())

    def open(self, run_id: str, needs: Iterable[str]) -> TfsRunHandle:
        parts_for(needs)  # validates the names
        return TfsRunHandle(self._dirs()[run_id], self.device, **self.handle_kw)


class HfRunSource(RunSource):
    """Runs in an HF dataset folder (e.g. repo advay333/tfs, prefix phase3/mnist), each
    downloaded on open with only the needed parts, and deleted on close unless keep."""

    def __init__(self, repo: str = "advay333/tfs", prefix: str = "phase3/mnist",
                 work: Path = Path("/tmp/tfs_phase4"), device: str = "cpu", token: Optional[str] = None,
                 keep: bool = False, remote: Optional[Remote] = None, **handle_kw):
        inner = remote if remote is not None else HfRemote(repo, token=token, create=False)
        self.remote = PrefixRemote(inner, prefix) if prefix else inner
        self.work = Path(work)
        self.device = device
        self.keep = keep
        self.handle_kw = handle_kw

    @cached_property
    def manifest(self) -> dict:
        return read_manifest(self.remote)

    def run_ids(self) -> list[str]:
        return sorted(self.manifest)

    def open(self, run_id: str, needs: Iterable[str]) -> TfsRunHandle:
        app = self.manifest[run_id]["app"]
        run_dir = pull_run(self.remote, app, run_id, self.work, parts=parts_for(needs))
        return TfsRunHandle(run_dir, self.device, delete_on_close=not self.keep, **self.handle_kw)

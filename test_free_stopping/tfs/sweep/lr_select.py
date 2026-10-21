"""Learning-rate selection from a short scan (PLAN.md §6 Phase 2 block 2; DESIGN.md §12).

Reads only training telemetry: choosing a learning rate must not look at test data.

A scan run is **stable** when it completed (no NaN/inf), its train loss over the last
20% of logged steps is below the initial loss (it trained), and that tail is within
`spike_ratio` of the lowest loss it reached (it is not blowing up at the end).

Choice per (variant, optimizer):
- Adam family: the published lr (PUBLISHED_ADAM_LR) if it is stable, otherwise the
  largest stable lr below it.
- SGD family: per alpha, one grid step below the largest stable lr (a safety margin:
  the scan is 5k steps, the corpus runs 1e5). The (variant, optimizer) lr is the
  smallest of those over alpha, so one lr is stable at every alpha level; alpha levels
  with no stable lr at all are reported and left out of that minimum.

The result is a table for the human to confirm before the lrs are frozen.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..config import RunConfig
from ..storage.reader import RunReader

ADAM_FAMILY = ("adamw", "adam_l2")
SGD_FAMILY = ("sgd", "sgd_momentum")
# Omnigrok's settings, which Phase 1 reproduced.
PUBLISHED_ADAM_LR = {"teacher_student": 3e-4}
DEFAULT_ADAM_LR = 1e-3


def variant_label(cfg: RunConfig) -> str:
    """The application variant: lr and gamma are chosen per (variant, optimizer)."""
    if cfg.app == "modular":
        return f"modular {cfg.task.params['op']}"
    if cfg.app == "mnist":
        return f"mnist d{cfg.model.depth}"
    return cfg.app


def scan_row(run_dir: Path, tail_frac: float = 0.2, spike_ratio: float = 10.0) -> dict:
    r = RunReader(run_dir)
    c = r.config
    tel = r.telemetry()
    loss = tel["train_loss"].to_numpy(dtype=float)
    row = {"run_id": c.run_id, "variant": variant_label(c), "optim": c.optim.name, "alpha": c.model.init_scale,
           "lr": c.optim.lr, "status": r.status, "last_step": int(tel["step"].iloc[-1])}
    finite = np.isfinite(loss)
    row["loss_0"] = float(loss[0])
    if r.status != "complete" or not finite.all():
        return {**row, "stable": False, "reason": "diverged"}
    tail = loss[-max(1, int(round(tail_frac * len(loss)))):]
    tail_med = float(np.median(tail))
    row.update(loss_end=tail_med, loss_min=float(loss.min()),
               train_acc_end=float(tel["train_acc"].iloc[-1]))
    if not tail_med < loss[0]:
        return {**row, "stable": False, "reason": "no progress"}
    if tail_med > spike_ratio * loss.min():
        return {**row, "stable": False, "reason": "blowing up"}
    return {**row, "stable": True, "reason": ""}


def _step_below(lr: float, grid: list[float]) -> Optional[float]:
    lower = [g for g in grid if g < lr]
    return max(lower) if lower else None


def select_lrs(rows: Iterable[dict]) -> list[dict]:
    """One row per (variant, optimizer) with the chosen lr, plus per-alpha detail rows
    for the SGD family (alpha given; chosen lr per alpha)."""
    groups: dict = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["optim"])].append(r)
    out = []
    for (variant, optim), rs in sorted(groups.items()):
        if optim in ADAM_FAMILY:
            out.append(_select_adam(variant, optim, rs))
        else:
            out.extend(_select_sgd(variant, optim, rs))
    return out


def _select_adam(variant: str, optim: str, rs: list[dict]) -> dict:
    published = PUBLISHED_ADAM_LR.get(variant, DEFAULT_ADAM_LR)
    stable = sorted(r["lr"] for r in rs if r["stable"])
    row = {"variant": variant, "optim": optim, "alpha": rs[0]["alpha"],
           "max_stable_lr": stable[-1] if stable else None, "published_lr": published}
    if any(r["stable"] and np.isclose(r["lr"], published) for r in rs):
        return {**row, "chosen_lr": published, "note": "published lr is stable"}
    below = [lr for lr in stable if lr < published]
    if below:
        return {**row, "chosen_lr": below[-1], "note": "published lr unstable; largest stable below it"}
    return {**row, "chosen_lr": None, "note": "no stable lr at or below the published one"}


def _select_sgd(variant: str, optim: str, rs: list[dict]) -> list[dict]:
    grid = sorted({r["lr"] for r in rs})
    by_alpha: dict = defaultdict(list)
    for r in rs:
        by_alpha[r["alpha"]].append(r)
    detail, usable = [], []
    for alpha in sorted(by_alpha):
        stable = sorted(r["lr"] for r in by_alpha[alpha] if r["stable"])
        if not stable:
            detail.append({"variant": variant, "optim": optim, "alpha": alpha, "max_stable_lr": None,
                           "chosen_lr": None, "note": "no stable lr: drop this alpha level"})
            continue
        chosen = _step_below(stable[-1], grid) or stable[-1]
        usable.append(chosen)
        detail.append({"variant": variant, "optim": optim, "alpha": alpha, "max_stable_lr": stable[-1],
                       "chosen_lr": chosen, "note": "one grid step below the largest stable lr"})
    summary = {"variant": variant, "optim": optim, "alpha": "all usable",
               "max_stable_lr": None, "chosen_lr": min(usable) if usable else None,
               "note": "smallest per-alpha choice" if usable else "no alpha level trains"}
    return [summary] + detail

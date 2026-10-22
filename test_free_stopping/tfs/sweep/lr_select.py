"""Learning-rate selection from a short scan (PLAN.md §6 Phase 2 block 2; DESIGN.md §12).

Reads only training telemetry: choosing a learning rate must not look at test data.

A scan run is **usable** (`stable` in the rows) when:
- it completed with finite losses (else "diverged");
- its train loss over the last 20% of logged steps is below the initial loss (else
  "no progress") and within `spike_ratio` of the lowest loss reached (else "blowing up");
- it **fits** the training data: final train accuracy >= FIT_ACC, or the loss fell by
  at least FIT_LOSS_DROP (else "not fitting"). Phase 2a showed why the loss alone is not
  enough: MNIST depth 8 at alpha = 1 drops from 0.13 to 0.09, the loss of predicting the
  mean target, and never learns anything. The loss-drop clause covers teacher-student,
  whose accuracy (RMS error < 0.01) can be capped by coupled weight decay.

Choice per (variant, optimizer):
- Adam family: the published lr (PUBLISHED_ADAM_LR) if it is usable, otherwise the
  largest usable lr below it.
- SGD family, per alpha: the largest usable lr, stepped down one grid point if the next
  larger lr was unstable ("diverged" / "blowing up"): a margin against divergence, since
  the scan is 5k steps and the corpus runs 1e5. The (variant, optimizer) row gives the
  smallest of those over alpha (one shared lr); alpha levels with no usable lr are
  reported and left out.

The result is a table for the human to confirm before the lrs are frozen.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import RunConfig
from ..storage.reader import RunReader

ADAM_FAMILY = ("adamw", "adam_l2")
SGD_FAMILY = ("sgd", "sgd_momentum")
# Omnigrok's settings, which Phase 1 reproduced.
PUBLISHED_ADAM_LR = {"teacher_student": 3e-4}
DEFAULT_ADAM_LR = 1e-3
FIT_ACC = 0.9
FIT_LOSS_DROP = 0.05  # final / initial train loss (teacher-student at alpha 0.5 floors at ~0.015 under L2)
UNSTABLE = ("diverged", "blowing up")


def variant_label(cfg: RunConfig) -> str:
    """The application variant: lr and gamma are chosen per (variant, optimizer)."""
    if cfg.app == "modular":
        return f"modular {cfg.task.params['op']}"
    if cfg.app == "mnist":
        return f"mnist d{cfg.model.depth}"
    return cfg.app


def scan_row(run_dir: Path, tail_frac: float = 0.2) -> dict:
    r = RunReader(run_dir)
    c = r.config
    tel = r.telemetry()
    loss = tel["train_loss"].to_numpy(dtype=float)
    row = {"run_id": c.run_id, "variant": variant_label(c), "optim": c.optim.name, "alpha": c.model.init_scale,
           "lr": c.optim.lr, "status": r.status, "last_step": int(tel["step"].iloc[-1]), "loss_0": float(loss[0])}
    if r.status == "complete" and np.isfinite(loss).all():
        tail = loss[-max(1, int(round(tail_frac * len(loss)))):]
        row.update(loss_end=float(np.median(tail)), loss_min=float(loss.min()),
                   train_acc_end=float(tel["train_acc"].iloc[-1]))
    return classify(row)


def classify(row: dict, spike_ratio: float = 10.0) -> dict:
    """Set `stable` and `reason` from a row's statistics (works on stored scan rows too)."""
    if row.get("status") != "complete" or row.get("loss_end") is None or not np.isfinite(row["loss_end"]):
        return {**row, "stable": False, "reason": "diverged"}
    l0, end = row["loss_0"], row["loss_end"]
    if not end < l0:
        return {**row, "stable": False, "reason": "no progress"}
    if end > spike_ratio * row["loss_min"]:
        return {**row, "stable": False, "reason": "blowing up"}
    if not (row["train_acc_end"] >= FIT_ACC or end <= FIT_LOSS_DROP * l0):
        return {**row, "stable": False, "reason": "not fitting"}
    return {**row, "stable": True, "reason": ""}


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
        return {**row, "chosen_lr": published, "note": "published lr is usable"}
    below = [lr for lr in stable if lr < published]
    if below:
        return {**row, "chosen_lr": below[-1], "note": "published lr not usable; largest usable below it"}
    return {**row, "chosen_lr": None, "note": "no usable lr at or below the published one"}


def _select_sgd(variant: str, optim: str, rs: list[dict]) -> list[dict]:
    by_alpha: dict = defaultdict(list)
    for r in rs:
        by_alpha[r["alpha"]].append(r)
    detail, usable = [], []
    for alpha in sorted(by_alpha):
        runs = sorted(by_alpha[alpha], key=lambda r: r["lr"])
        ok = [i for i, r in enumerate(runs) if r["stable"]]
        if not ok:
            detail.append({"variant": variant, "optim": optim, "alpha": alpha, "max_stable_lr": None,
                           "chosen_lr": None, "note": "no usable lr: drop this alpha level"})
            continue
        i = ok[-1]
        above_unstable = i + 1 < len(runs) and runs[i + 1].get("reason") in UNSTABLE
        j = i - 1 if above_unstable and i > 0 else i
        usable.append(runs[j]["lr"])
        detail.append({"variant": variant, "optim": optim, "alpha": alpha, "max_stable_lr": runs[i]["lr"],
                       "chosen_lr": runs[j]["lr"], "train_acc_at_chosen": runs[j].get("train_acc_end"),
                       "note": "one step below: next lr unstable" if j < i else "largest usable lr"})
    summary = {"variant": variant, "optim": optim, "alpha": "all usable",
               "max_stable_lr": None, "chosen_lr": min(usable) if usable else None,
               "note": "smallest per-alpha choice" if usable else "no alpha level trains"}
    return [summary] + detail

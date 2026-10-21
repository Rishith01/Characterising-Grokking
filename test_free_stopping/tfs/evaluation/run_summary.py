"""One summary row and compact curves per finished run (evaluation side: reads the oracle).

Sweeps compute these as each run finishes, before the run is uploaded and its local copy
deleted, so pilot tables and plots never need the runs downloaded again.

`provisional_category` sorts runs into the PLAN.md §3 categories with simple fixed
rules. It exists for gate G2 ("the categories all appear"); Phase 5's GroundTruth
replaces it with the definitive rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..storage import layout
from ..storage.reader import RunReader
from ..sweep.lr_select import variant_label
from .oracle import Oracle

EPS = 0.01  # PLAN.md §5


def _first_step(steps: np.ndarray, mask: np.ndarray) -> Optional[int]:
    return int(steps[mask.argmax()]) if mask.any() else None


def run_summary(run_dir: Path) -> dict:
    r = RunReader(run_dir)
    c = r.config
    tel = r.telemetry()
    orc = Oracle(run_dir).table()
    steps, test = orc["step"].to_numpy(), orc["test_acc"].to_numpy(dtype=float)
    finite = np.isfinite(test)
    mx = float(test[finite].max()) if finite.any() else float("nan")
    size = {"modular": "train_frac", "mnist": "n_train", "teacher_student": "n_train"}.get(c.app)
    report = run_dir / layout.REPLAY_REPORT
    row = {
        "run_id": c.run_id, "variant": variant_label(c), "optim": c.optim.name, "lr": c.optim.lr,
        "gamma": c.optim.weight_decay, "alpha": c.model.init_scale,
        "size": c.task.params.get(size) if size else None, "seed": c.seed, "steps": c.steps,
        "status": r.status, "diverged_at": r.meta.get("diverged_at"),
        "replay_level": json.loads(report.read_text())["level"] if report.exists() else None,
        "wall_s": round(float(r.meta.get("wall_time_s", 0.0)), 1),
        "size_mb": round(sum(f.stat().st_size for f in Path(run_dir).iterdir() if f.is_file()) / 1e6, 1),
        "train99_step": _first_step(tel["step"].to_numpy(), tel["train_acc"].to_numpy() >= 0.99),
        "max_test": round(mx, 4),
        "max_test_step": int(steps[np.nanargmax(np.where(finite, test, -1))]) if finite.any() else None,
        "t_eps": _first_step(steps, finite & (test >= mx - EPS)),
        "final_test": round(float(test[-1]), 4),
        "final_train_acc": round(float(tel["train_acc"].iloc[-1]), 4),
    }
    row["category"] = provisional_category(row)
    return row


def provisional_category(row: dict) -> str:
    """diverged | never | late_collapse | grokker | joint_plateau | fast | no_fit."""
    if row["status"] == "diverged":
        return "diverged"
    if not row["max_test"] >= 0.5:
        return "never"
    if row["final_test"] < row["max_test"] - 0.1:
        return "late_collapse"
    fit, t_eps = row["train99_step"], row["t_eps"]
    if fit is None:
        return "no_fit"  # generalizes somewhat but never fits the training set
    if t_eps >= 10 * max(fit, 1) and t_eps - fit >= 1000:
        return "grokker"
    if fit >= 2000 and t_eps <= 2 * fit:
        return "joint_plateau"  # train and test stall together, then rise together
    return "fast"


def run_curves(run_dir: Path, n_points: int = 200) -> dict:
    """Train/test accuracy and total weight norm at ~n_points log-spaced log steps."""
    tel = RunReader(run_dir).telemetry()
    orc = Oracle(run_dir).table()
    steps = tel["step"].to_numpy()
    idx = np.unique(np.clip(np.round(np.geomspace(1, len(steps), n_points)).astype(int) - 1, 0, len(steps) - 1))
    norm_cols = [col for col in tel.columns if col.endswith("_w_norm")]
    w_norm = np.sqrt((tel[norm_cols].to_numpy(dtype=float) ** 2).sum(axis=1))
    test = orc.set_index("step")["test_acc"].reindex(steps).to_numpy(dtype=float)
    return {"step": steps[idx].tolist(),
            "train_acc": np.round(tel["train_acc"].to_numpy(dtype=float)[idx], 4).tolist(),
            "test_acc": np.round(test[idx], 4).tolist(),
            "train_loss": tel["train_loss"].to_numpy(dtype=float)[idx].tolist(),
            "w_norm": np.round(w_norm[idx], 4).tolist()}

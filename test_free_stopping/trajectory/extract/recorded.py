"""Stage E from recorded data: no replay, no GPU.

From the telemetry (every log step, 50 steps in the corpus): step, learning rate, train
loss and accuracy, and the norm of each layer's parameters (weight and bias together;
weight decay acts on both). From the summaries (every summary step, 500): each layer's
singular values (sv/l{i}) and each hidden layer's training-set activation covariance
spectrum (actspec/l{i}), exactly as tfs.training.recorder.Recorder stored them.

Training-side data only; no test data is read (the quarantine applies to this module).
"""
from __future__ import annotations

import re

import numpy as np

VERSION = 1
_W_NORM = re.compile(r"^l(\d+)_w_norm$")
_SUMMARY = re.compile(r"^(sv|actspec)/l(\d+)$")


def run_meta(handle) -> dict:
    """Facts the quantities need: optimizer and decay settings, schedules, shapes."""
    ctx = handle.context
    cfg = ctx.config
    rec = cfg["record"]
    return {
        "run_id": ctx.run_id, "app": ctx.app, "optimizer": ctx.optimizer, "lr": ctx.lr, "gamma": ctx.gamma,
        "momentum": ctx.momentum, "betas": list(cfg["optim"].get("betas", ())), "decay": ctx.decay,
        "loss": ctx.loss, "steps": ctx.steps, "log_every": int(rec["log_every"]),
        "summary_every": int(rec["summary_every"]), "keyframe_every": int(rec["keyframe_every"]),
        "d_in": ctx.d_in, "d_out": ctx.d_out, "n_train": ctx.n_train, "batch_size": ctx.batch_size,
        "depth": int(cfg["model"]["depth"]), "width": int(cfg["model"]["width"]),
        "init_scale": float(cfg["model"]["init_scale"]),
    }


def extract_recorded(handle) -> tuple[dict[str, np.ndarray], dict]:
    """(arrays, meta) for one run. The handle needs the 'telemetry' and 'summaries' parts."""
    tel = handle.telemetry().sort_values("step").reset_index(drop=True)
    cols = sorted((int(m.group(1)), c) for c in tel.columns for m in [_W_NORM.match(c)] if m)
    arrays: dict[str, np.ndarray] = {
        "log_steps": tel["step"].to_numpy(np.int64),
        "lr": tel["lr"].to_numpy(np.float64),
        "train_loss": tel["train_loss"].to_numpy(np.float64),
        "train_acc": tel["train_acc"].to_numpy(np.float64),
        "w_norm": tel[[c for _, c in cols]].to_numpy(np.float64),  # [n_log, n_layers]
    }
    steps = [int(s) for s in handle.reader.summary_steps()]
    per_key: dict[str, list[np.ndarray]] = {}
    for s in steps:
        for k, v in handle.summaries(s).items():
            m = _SUMMARY.match(k)
            if m:
                per_key.setdefault(f"{m.group(1)}_l{m.group(2)}", []).append(np.asarray(v, dtype=np.float64))
    arrays["summary_steps"] = np.asarray(steps, dtype=np.int64)
    for k, rows in per_key.items():
        arrays[k] = np.stack(rows)  # [n_summary, spectrum length]
    meta = {**run_meta(handle), "stage": "recorded", "version": VERSION, "n_layers": len(cols)}
    return arrays, meta

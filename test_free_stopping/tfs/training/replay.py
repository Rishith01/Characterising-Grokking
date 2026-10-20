"""ReplayEngine: regenerate the training state at any step from the nearest earlier
keyframe, verifying against the recorded run as it goes (DESIGN.md §6.5, PLAN.md §4.1).

Levels:
    L0      bit-identical: keyframe tensor hashes, telemetry, predictions, fingerprints
    L1      every comparison within ReplayTolerance
    failed  anything out of tolerance (iterate() raises ReplayDivergence immediately)

Replay only moves forward and never opens a keyframe after the largest requested step,
so producing the model at step t uses nothing recorded after t.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional, Union

import numpy as np
import torch
import yaml

from ..causal.view import TrainContext
from ..models import MLP
from ..storage import layout
from ..storage.reader import RunReader
from . import determinism
from .loop import TrainLoop
from .recorder import Recorder


@dataclass(frozen=True)
class ReplayTolerance:
    weight_rel: float = 1e-6  # per tensor ||a - b|| / ||b||, at keyframes
    optim_rel: float = 1e-5  # per optimizer-state tensor, at keyframes
    fingerprint_rel: float = 1e-5  # per entry, scaled by the parameter's norm
    telemetry_rel: float = 1e-5  # per column, |a - b| / max(|b|, telemetry_atol)
    telemetry_atol: float = 1e-8
    pred_agreement: float = 0.999  # fraction of identical train predictions per log step

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ReplayTolerance":
        path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "replay_tolerance.yaml"
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls(**(yaml.safe_load(f) or {}))


class ReplayDivergence(RuntimeError):
    pass


class EnvironmentMismatch(RuntimeError):
    pass


class CorruptRun(RuntimeError):
    pass


@dataclass
class ReplayReport:
    level: Literal["L0", "L1", "failed"] = "L0"
    steps_checked: list = field(default_factory=list)  # log steps compared
    keyframes_checked: list = field(default_factory=list)
    max_err: dict = field(default_factory=lambda: {"weights": 0.0, "optim": 0.0, "fingerprint": 0.0, "telemetry": 0.0})
    min_pred_agreement: float = 1.0
    first_failure: Optional[str] = None
    env_mismatches: list = field(default_factory=list)

    def _degrade(self, exact: bool, within: bool, where: str) -> None:
        if not within:
            if self.level != "failed":
                self.first_failure = where
            self.level = "failed"
        elif not exact and self.level == "L0":
            self.level = "L1"

    def to_dict(self) -> dict:
        return asdict(self)


class ReplayEngine:
    def __init__(self, run_dir: Union[str, Path], device: str = "cpu",
                 tolerance: Optional[ReplayTolerance] = None, allow_env_mismatch: bool = False):
        determinism.apply()
        self.reader = RunReader(run_dir)
        # Replay compares against the stored keyframes and streams, so they must be the
        # bytes that were recorded. A partial download is fine (absent files are skipped).
        bad = self.reader.verify_checksums()
        if bad:
            raise CorruptRun(f"{self.reader.run_dir}: checksum mismatch in {bad}")
        self.cfg = self.reader.config
        self.tol = tolerance or ReplayTolerance.load()
        self.device = torch.device(device)

        here = determinism.EnvRecord.capture(str(self.device))
        self.env_mismatches = self.reader.env.mismatches(here)
        if self.env_mismatches and not allow_env_mismatch:
            raise EnvironmentMismatch(
                f"replay environment differs from the recorded run in {self.env_mismatches}; "
                "pass allow_env_mismatch=True to replay anyway (the run must then still pass L1)"
            )

        ctx = TrainContext.from_config(self.cfg)
        self.loop = TrainLoop(self.cfg, ctx.x_train, ctx.y_train, ctx.kind, ctx.d_in, ctx.d_out, self.device)
        self.recorder = Recorder(self.loop, ctx.task, self.cfg.record.n_fingerprint_proj, self.cfg.seed)
        self._log_steps = set(self.reader.log_steps.tolist())
        self._key_steps = self.reader.keyframe_steps
        self._stored = None  # lazily loaded stored streams for verification

    # ---- public API ----------------------------------------------------------------
    def iterate(self, steps: Iterable[int], verify: bool = True,
                report: Optional[ReplayReport] = None) -> Iterator[tuple[int, MLP]]:
        """Yield (step, model) for each requested step, in increasing order. The model
        object is the live training model: read it, do not modify it."""
        steps = sorted(set(int(s) for s in steps))
        if not steps:
            return
        report = report if report is not None else ReplayReport(env_mismatches=self.env_mismatches)
        last = steps[-1]
        if last > int(self._key_steps.max()):
            raise ValueError(f"step {last} is beyond the recorded horizon")

        start_key = int(self._key_steps[self._key_steps <= steps[0]].max())
        self.loop.restore(self.reader.keyframe(start_key))
        later_keys = [int(k) for k in self._key_steps if start_key < k <= last]
        wanted = set(steps)

        t, info = start_key, None
        while True:
            if verify and t in self._log_steps:
                # Right after a restore there is no StepInfo for the update that led to
                # step t, so its update columns are skipped; at step 0 they are NaN on
                # both sides. Every later log step is compared in full.
                self._check_log(t, info, report, compare_update=info is not None or t == 0)
            if verify and t in later_keys:
                self._check_keyframe(t, report)
            if t in wanted:
                yield t, self.loop.model
            if t == last:
                break
            info = self.loop.step(t, want_info=verify and (t + 1) in self._log_steps)
            t += 1

    def verify(self, intervals: Union[Literal["all"], list[tuple[int, int]]] = "all") -> ReplayReport:
        """Replay keyframe intervals and compare everything. "all" = the whole run."""
        keys = [int(k) for k in self._key_steps]
        if intervals == "all":
            intervals = [(keys[0], keys[-1])]
        report = ReplayReport(env_mismatches=self.env_mismatches)
        for a, b in intervals:
            try:
                for _ in self.iterate([a, b], verify=True, report=report):
                    pass
            except ReplayDivergence:
                pass  # recorded in the report; stop this interval
        return report

    def write_report(self, report: ReplayReport) -> Path:
        path = self.reader.run_dir / layout.REPLAY_REPORT
        with open(path, "w") as f:
            json.dump({**report.to_dict(), "device": str(self.device)}, f, indent=2)
        return path

    # ---- comparisons ---------------------------------------------------------------
    def _load_stored(self):
        if self._stored is None:
            tel = self.reader.telemetry().set_index("step")
            fps = self.reader.fingerprints().set_index("step")
            preds = self.reader.train_preds()
            pred_index = {int(s): i for i, s in enumerate(tel.index)}
            self._stored = (tel, fps, preds, pred_index)
        return self._stored

    def _check_log(self, t: int, info, report: ReplayReport, compare_update: bool) -> None:
        tel, fps, preds, pred_index = self._load_stored()
        row, pred, fp = self.recorder.log(t, info)
        where = f"log step {t}"

        cols = [c for c in tel.columns if compare_update or not _is_update_col(c)]
        a = np.array([row[c] for c in cols], dtype=np.float64)
        b = tel.loc[t, cols].to_numpy(dtype=np.float64)
        exact = np.array_equal(a, b, equal_nan=True)
        err = _rel_err(a, b, self.tol.telemetry_atol)
        report.max_err["telemetry"] = max(report.max_err["telemetry"], err)
        report._degrade(exact, err <= self.tol.telemetry_rel, f"{where}: telemetry rel err {err:.3g}")

        stored_pred = preds[pred_index[t]]
        if np.issubdtype(stored_pred.dtype, np.integer):
            agree = float(np.mean(pred == stored_pred))
            p_exact = agree == 1.0
        else:  # regression: per-sample error, compare as numbers
            p_exact = np.array_equal(pred, stored_pred)
            agree = float(np.mean(np.isclose(pred, stored_pred, rtol=1e-3, atol=1e-6)))
        report.min_pred_agreement = min(report.min_pred_agreement, agree)
        report._degrade(p_exact, agree >= self.tol.pred_agreement, f"{where}: prediction agreement {agree:.4f}")

        s = fps.loc[t]
        f_exact, f_err = True, 0.0
        for name, _ in self.loop.model.param_list():
            key = name.replace(".", "_")
            scale = math.sqrt(max(s[f"{key}_sumsq"], 0.0)) or 1.0
            for col in [c for c in fps.columns if c.startswith(key + "_")]:
                x, y = fp[col], float(s[col])
                f_exact &= x == y
                denom = abs(y) if col.endswith("_sumsq") else scale
                f_err = max(f_err, abs(x - y) / max(denom, 1e-30))
        report.max_err["fingerprint"] = max(report.max_err["fingerprint"], f_err)
        report._degrade(f_exact, f_err <= self.tol.fingerprint_rel, f"{where}: fingerprint rel err {f_err:.3g}")

        report.steps_checked.append(t)
        if report.level == "failed":
            raise ReplayDivergence(report.first_failure)

    def _check_keyframe(self, t: int, report: ReplayReport) -> None:
        replayed = self.loop.capture(t)
        stored_hashes = self.reader.keyframe_hashes()[t]
        exact = replayed.hashes() == stored_hashes
        within = True
        if not exact:
            stored = self.reader.keyframe(t)
            w_err = _max_tensor_err(replayed.params, stored.params)
            o_err = _max_tensor_err(replayed.optim, stored.optim)
            report.max_err["weights"] = max(report.max_err["weights"], w_err)
            report.max_err["optim"] = max(report.max_err["optim"], o_err)
            within = w_err <= self.tol.weight_rel and o_err <= self.tol.optim_rel
            report._degrade(False, within, f"keyframe {t}: weight err {w_err:.3g}, optim err {o_err:.3g}")
        report.keyframes_checked.append(t)
        if report.level == "failed":
            raise ReplayDivergence(report.first_failure)


_UPDATE_SUFFIXES = ("_g_norm", "_reg_g_norm", "_update_norm", "_update_radial",
                    "_update_tangential", "_cos_g_w")


def _is_update_col(col: str) -> bool:
    return col.endswith(_UPDATE_SUFFIXES)


def _rel_err(a: np.ndarray, b: np.ndarray, atol: float) -> float:
    both_nan = np.isnan(a) & np.isnan(b)
    if np.any(np.isnan(a) != np.isnan(b)):
        return math.inf
    a, b = a[~both_nan], b[~both_nan]
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), atol)))


def _max_tensor_err(a: dict, b: dict) -> float:
    if a.keys() != b.keys():
        return math.inf
    worst = 0.0
    for k in a:
        x, y = a[k].double(), b[k].double()
        if x.shape != y.shape:
            return math.inf
        diff = float((x - y).norm())
        ref = float(y.norm())
        worst = max(worst, diff / ref if ref > 0 else diff)
    return worst

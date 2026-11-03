"""The signal runner (DESIGN.md §13.3).

compute_run walks one run's log steps in increasing order and feeds every signal the
Observation for its cadence. All signals that need models share one replay pass.
compute() does this over many runs, caching each (signal, run) trace in a TraceStore and
skipping what is already cached, so it resumes after an interruption.
"""
from __future__ import annotations

import time
import traceback
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .signal import SignalConfig
from .source import Observation, RunHandle, RunSource
from .store import TraceStore


def eval_grid(log_steps: Sequence[int], n_log: int = 200, n_uniform: int = 400) -> np.ndarray:
    """Default eval steps: n_log log-spaced plus n_uniform evenly spaced steps over the
    run, each snapped to the nearest log step (DESIGN.md §7.1)."""
    log_steps = np.asarray(log_steps, dtype=np.int64)
    last = int(log_steps[-1])
    want = np.concatenate([[0], np.geomspace(1, max(last, 1), n_log), np.linspace(0, last, n_uniform)])
    idx = np.clip(np.searchsorted(log_steps, want), 0, len(log_steps) - 1)
    lower = np.clip(idx - 1, 0, len(log_steps) - 1)
    pick = np.where(np.abs(log_steps[lower] - want) <= np.abs(log_steps[idx] - want), lower, idx)
    return np.unique(log_steps[pick])


def needs(configs: Iterable[SignalConfig]) -> set[str]:
    out = set()
    for c in configs:
        out |= set(c.build().requires)
    return out


def compute_run(handle: RunHandle, configs: Sequence[SignalConfig],
                eval_steps: Optional[Sequence[int]] = None, max_step: Optional[int] = None,
                eval_kw: Optional[dict] = None) -> dict[str, pd.DataFrame]:
    """Traces for one run: signal config key -> DataFrame (step + the signal's columns).
    max_step truncates the run (used by the causality test)."""
    ctx = handle.context
    sigs = [(c, c.build()) for c in configs]
    for _, s in sigs:
        s.reset(ctx)
    need = set().union(*(s.requires for _, s in sigs)) if sigs else set()

    steps = ctx.log_steps if max_step is None else ctx.log_steps[ctx.log_steps <= max_step]
    summary_set = set(int(s) for s in ctx.summary_steps)
    any_eval = any(s.cadence == "eval" for _, s in sigs)
    if eval_steps is None:
        eval_steps = eval_grid(ctx.log_steps, **(eval_kw or {})) if any_eval else []
    eval_list = sorted(int(t) for t in set(eval_steps) & set(int(s) for s in steps)) if any_eval else []
    eval_set = set(eval_list)

    tel = handle.telemetry().to_dict("records") if "telemetry" in need else None
    preds = handle.train_preds() if "preds" in need else None
    held = handle.heldout().set_index("step") if "heldout" in need else None
    models = handle.models(eval_list) if "models" in need and eval_list else None

    rows: dict[str, list[dict]] = {c.key: [] for c, _ in sigs}
    for i, t in enumerate(int(s) for s in steps):
        obs = Observation(t)
        if tel is not None:
            obs.telemetry = tel[i]
        if preds is not None:
            obs.train_preds = preds[i]
        if "summaries" in need and t in summary_set:
            obs.summaries = handle.summaries(t)
        if models is not None and t in eval_set:
            got, model = next(models)
            if got != t:
                raise RuntimeError(f"replay yielded step {got}, expected {t}")
            obs.model = model
        held_row = held.loc[t].to_dict() if held is not None and t in held.index else None

        for c, s in sigs:
            due = (s.cadence == "log" or (s.cadence == "summary" and t in summary_set)
                   or (s.cadence == "eval" and t in eval_set))
            if not due:
                continue
            obs.heldout = held_row if s.uses_heldout else None
            row = s.update(obs)
            if row is not None:
                rows[c.key].append({"step": t, **row})
        obs.heldout = None
    if models is not None:
        models.close()

    return {k: pd.DataFrame(v) if v else pd.DataFrame({"step": pd.Series(dtype="int64")})
            for k, v in rows.items()}


def compute(source: RunSource, run_ids: Iterable[str], configs: Sequence[SignalConfig],
            store: TraceStore, eval_kw: Optional[dict] = None,
            log: Callable[[str], None] = print) -> list[dict]:
    """Compute and cache every (signal, run) trace not already in the store. Returns one
    status row per run; a failing run is logged and skipped, never aborting the rest."""
    status = []
    run_ids = list(run_ids)
    for n, run_id in enumerate(run_ids, 1):
        pending = [c for c in configs if not store.has(c, run_id)]
        row = {"run_id": run_id, "signals": len(pending), "ok": True, "seconds": 0.0, "error": ""}
        if pending:
            t0 = time.time()
            try:
                with source.open(run_id, needs(pending)) as h:
                    traces = compute_run(h, pending, eval_kw=eval_kw)
                for c in pending:
                    store.write(c, run_id, traces[c.key])
            except Exception as e:  # keep going; the run is retried next time
                row.update(ok=False, error=f"{type(e).__name__}: {e}")
                log(traceback.format_exc())
            row["seconds"] = round(time.time() - t0, 1)
        log(f"[{n}/{len(run_ids)}] {run_id}: {len(pending)} signal(s), "
            f"{'ok' if row['ok'] else 'FAILED ' + row['error']} ({row['seconds']} s)")
        status.append(row)
    return status

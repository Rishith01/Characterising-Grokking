"""Phase 4 pilot: compute the built signals on a handful of runs and plot them against
train/test accuracy (DESIGN.md §13.6). Evaluation side: it reads test curves to plot and
tabulate, so it is exempt from the quarantine test, like stopping.heldout.

This is for looking at the signals, not for scoring methods: the rule settings are
uncalibrated placeholders (stopping.methods). Scoring is Phase 5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from tfs.evaluation.report import rows_to_markdown

from .compute import compute
from .heldout import b6_method, read_heldout
from .method import Method
from .methods import signal_configs
from .plotting import plot_run
from .source import RunSource
from .store import TraceStore

CATEGORY_ORDER = ("grokker", "joint_plateau", "fast", "late_collapse", "never", "no_fit")


def pick_runs(rows: list[dict], per_category: int = 3, seed: int = 0) -> list[dict]:
    """Up to per_category runs of each provisional category, seed `seed` only, spread
    over (variant, optimizer): round-robin over the groups, in a fixed order."""
    out = []
    for cat in CATEGORY_ORDER:
        cand = sorted((r for r in rows if r["category"] == cat and r["seed"] == seed),
                      key=lambda r: (r["variant"], r["optim"], r["alpha"], r["gamma"], r["run_id"]))
        groups: dict = {}
        for r in cand:
            groups.setdefault((r["variant"], r["optim"]), []).append(r)
        lists = list(groups.values())
        i = 0
        while len([r for r in out if r["category"] == cat]) < per_category and any(lists):
            lst = lists[i % len(lists)]
            if lst:
                out.append(lst.pop(len(lst) // 2))
            i += 1
    return out


def _curves(source: RunSource, run_id: str) -> dict:
    with source.open(run_id, {"telemetry", "heldout"}) as h:
        tel = h.telemetry()
        test = h.heldout().set_index("step")["test_acc"].reindex(tel["step"]).to_numpy(dtype=float)
    return {"step": tel["step"].tolist(), "train_acc": tel["train_acc"].tolist(), "test_acc": test.tolist()}


def _acc_at(curves: dict, step: Optional[int]) -> float:
    if step is None:
        return float("nan")
    i = int(np.searchsorted(np.asarray(curves["step"]), step))
    return float(curves["test_acc"][min(i, len(curves["step"]) - 1)])


def run_pilot(source: RunSource, runs: list[dict], methods: list[Method], store: TraceStore,
              out_dir: Path, eval_kw: Optional[dict] = None,
              log: Callable[[str], None] = print) -> dict:
    """Compute every method's signals (plus B6) on `runs`, then one figure per run and a
    table of stop steps. runs: rows with at least run_id (and category for the table).
    Returns {"status", "stops", "plots"}."""
    methods = list(methods) + [b6_method()]
    cfgs = signal_configs(methods)
    ids = [r["run_id"] for r in runs]
    status = compute(source, ids, cfgs, store, eval_kw=eval_kw, log=log)

    stops, plots = [], []
    for r in runs:
        rid = r["run_id"]
        if not all(store.has(c, rid) for c in cfgs):
            continue
        traces = {c.key: store.read(c, rid) for c in cfgs}
        curves = _curves(source, rid)
        decisions = {m.name: m.decide(traces) for m in methods}
        for m, d in decisions.items():
            stops.append({"run_id": rid, "category": r.get("category"), "variant": r.get("variant"),
                          "optim": r.get("optim"), "method": m, "stop": d.step,
                          "test_at_stop": round(_acc_at(curves, d.step), 4),
                          "max_test": r.get("max_test"), "t_eps": r.get("t_eps")})
        labels = {c.key: c.name for c in cfgs}
        cols = {"weight_norm": "value", "erank": "value"}
        title = (f"{r.get('variant')} {r.get('optim')} a={r.get('alpha')} g={r.get('gamma')} "
                 f"[{r.get('category')}] {rid}")
        shown = {labels[k]: v for k, v in traces.items() if labels[k] != "step"}  # B0's panel is a diagonal
        plots.append(plot_run(curves, shown, decisions,
                              Path(out_dir) / "plots" / f"{rid}.png", title=title,
                              columns={labels[k]: cols.get(labels[k], "value") for k in traces}))
        log(f"plotted {rid}")
    stops_df = pd.DataFrame(stops)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stops_df.to_csv(Path(out_dir) / "stops.csv", index=False)
    return {"status": status, "stops": stops_df, "plots": plots}


def stops_markdown(stops: pd.DataFrame) -> str:
    """One row per run, one column per method: stop step (test acc at stop)."""
    if stops.empty:
        return "(no runs)"
    rows = []
    for rid, g in stops.groupby("run_id", sort=False):
        first = g.iloc[0]
        row = {"run": rid[:10], "category": first["category"], "max_test": first["max_test"],
               "t_eps": first["t_eps"]}
        for _, s in g.iterrows():
            row[s["method"]] = "never" if pd.isna(s["stop"]) else f"{int(s['stop'])} ({s['test_at_stop']:.2f})"
        rows.append(row)
    return rows_to_markdown(rows)

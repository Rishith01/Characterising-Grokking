"""Which runs are in scope, and their metadata (PHASE5_PLAN.md §0).

The run list and per-run metadata come from the Phase 3 reports on HF
(phase3/<app>/reports/corpus_runs.json): hyperparameters, wall time, and the provisional
outcome labels (category, t_eps, train99_step, max_test). The labels are hindsight
metadata, used only to group and plot runs, never by an extraction or a quantity.

Duplicates: at gamma = 0, AdamW and Adam+L2 are the same algorithm, so each such Adam+L2
run is a copy of its AdamW twin (same variant, lr, alpha, size, seed). Copies are marked
with `duplicate_of` and skipped by default.
"""
from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

APPS = ("modular", "mnist")
SCOPE = ("modular x+y", "mnist d3", "mnist d8")
COLUMNS = ("run_id", "app", "variant", "optim", "lr", "gamma", "alpha", "size", "seed", "steps", "wall_s",
           "size_mb", "replay_level", "category", "t_eps", "train99_step", "max_test", "final_test")
_TWIN_KEY = ("variant", "lr", "alpha", "size", "seed")


def app_of(run_id: str) -> str:
    return run_id.split("__", 1)[0]


def corpus_table(rows: Iterable[dict], variants: Iterable[str] = SCOPE) -> pd.DataFrame:
    """One row per in-scope run, sorted by run_id, with `duplicate_of` (run_id or None)."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=list(COLUMNS) + ["duplicate_of"])
    df = df[df["variant"].isin(list(variants))].copy()
    df["app"] = df["run_id"].map(app_of)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[list(COLUMNS)].sort_values("run_id").reset_index(drop=True)
    twins = df[(df["optim"] == "adamw") & (df["gamma"] == 0)].set_index(list(_TWIN_KEY))["run_id"]
    dup = []
    for _, r in df.iterrows():
        key = tuple(r[k] for k in _TWIN_KEY)
        dup.append(twins.get(key) if r["optim"] == "adam_l2" and r["gamma"] == 0 and key in twins.index else None)
    df["duplicate_of"] = pd.Series(dup, index=df.index, dtype=object)
    df["duplicate_of"] = df["duplicate_of"].where(df["duplicate_of"].notna(), None)
    return df


def load_corpus(remote, variants: Iterable[str] = SCOPE, apps: Iterable[str] = APPS) -> pd.DataFrame:
    """corpus_table from the Phase 3 reports. `remote` is the repo root (tfs.storage.hub.Remote)."""
    from tfs.storage.hub import PrefixRemote

    rows = []
    for app in apps:
        rows += PrefixRemote(remote, f"phase3/{app}").read_json("reports/corpus_runs.json") or []
    return corpus_table(rows, variants)


def pilot_runs(table: pd.DataFrame, candidates: Optional[Iterable[str]] = None) -> list[dict]:
    """One run per (variant, optimizer), so a pilot covers both MNIST networks and every
    optimizer path (including Adam+L2 with gamma > 0, the only user of AdamDecayProbe).
    Prefers runs in `candidates` (e.g. F2's runs, to compare against) and gamma > 0."""
    t = table[table["duplicate_of"].isna()].copy()
    cand = set(candidates or ())
    t["_cand"] = t["run_id"].isin(cand)
    t["_decay"] = t["gamma"] > 0
    t = t.sort_values(["_cand", "_decay", "run_id"], ascending=[False, False, True])
    pick = t.groupby(["variant", "optim"], sort=True).head(1)
    return pick.drop(columns=["_cand", "_decay"]).sort_values("wall_s", ascending=False).to_dict("records")


def work_list(table: pd.DataFrame, include_duplicates: bool = False, limit: Optional[int] = None) -> list[dict]:
    """Runs to process, longest first (by recorded training wall time), as plain dicts."""
    t = table if include_duplicates else table[table["duplicate_of"].isna()]
    t = t.sort_values("wall_s", ascending=False, na_position="last")
    out = t.to_dict("records")
    return out[:limit] if limit else out

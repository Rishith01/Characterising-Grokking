"""Phase 2b analysis: gate G2 checks and the Phase 3/4 cost projection (PLAN.md §6).

Both work on the summary rows that scripts/sweep.py produces (evaluation/run_summary.py),
so they never need the runs themselves.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

# Phase 3 runs per application variant (PLAN.md §3, "Grid size").
PHASE3_RUNS = {"teacher_student": 576, "modular x+y": 576, "modular x*y": 128, "modular x^2+xy+y^2": 128,
               "mnist d3": 128, "mnist d8": 128}
GENERALIZED = ("grokker", "fast", "joint_plateau", "late_collapse", "no_fit")
G2_SPREAD_DECADES = 1.5
G2_MAX_GB = 120.0


def _spread(t_eps: list[int]) -> float | None:
    ts = [max(t, 50) for t in t_eps if t is not None]  # one log step: the resolution of t_eps
    return round(math.log10(max(ts) / min(ts)), 2) if len(ts) >= 2 else None


def g2_checks(rows: list[dict], by: tuple[str, ...] = ("variant",)) -> list[dict]:
    """One row per group (default: per application variant, pooled over optimizers).
    Grok-time spread = log10(max / min) of t_eps over the runs that generalized."""
    groups: dict = defaultdict(list)
    for r in rows:
        if "variant" in r:
            groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        cats = Counter(r["category"] for r in rs)
        gen = [r["t_eps"] for r in rs if r["category"] in GENERALIZED]
        grok = [r["t_eps"] for r in rs if r["category"] == "grokker"]
        levels = Counter(r.get("replay_level") for r in rs)
        spread = _spread(gen)
        out.append({**dict(zip(by, key)), "runs": len(rs),
                    "categories": dict(sorted(cats.items())),
                    "t_eps_spread_decades": spread, "grokker_spread_decades": _spread(grok),
                    "spread_ok": spread is not None and spread >= G2_SPREAD_DECADES,
                    "has_never": cats["never"] > 0,
                    "replay": dict(levels), "replay_ok": set(levels) <= {"L0", "L1"}})
    return out


def project(rows: list[dict], workers: int, horizon: int = 100_000,
            runs: dict[str, int] = PHASE3_RUNS, ts_horizon: int | None = None) -> list[dict]:
    """GPU-hours and GB for Phase 3 (training) and Phase 4 (one replay pass per run).

    Per-run wall time is measured with `workers` runs in parallel, so the aggregate time
    for n runs is n * wall / workers. Runs at another horizon are rescaled to `horizon`
    (wall time and size both grow about linearly with steps). The replay pass is taken
    to cost the same as training; signal math comes on top. `ts_horizon` projects the
    teacher-student runs at a longer horizon instead.
    """
    per: dict = defaultdict(list)
    for r in rows:
        if r.get("status") in ("complete", "diverged") and r.get("wall_s") and r.get("variant") in runs:
            per[r["variant"]].append(r)
    out, tot = [], {"train_gpu_h": 0.0, "size_gb": 0.0}
    for v, n in runs.items():
        rs = per.get(v, [])
        if not rs:
            out.append({"variant": v, "phase3_runs": n, "note": "no pilot runs"})
            continue
        h = (ts_horizon if (v == "teacher_student" and ts_horizon) else horizon)
        wall = sum(r["wall_s"] * h / r["steps"] for r in rs) / len(rs)
        size = sum(r["size_mb"] * h / r["steps"] for r in rs) / len(rs)
        row = {"variant": v, "phase3_runs": n, "horizon": h, "wall_s_per_run": round(wall),
               "size_mb_per_run": round(size, 1),
               "train_gpu_h": round(n * wall / workers / 3600, 1), "size_gb": round(n * size / 1e3, 1)}
        tot["train_gpu_h"] += row["train_gpu_h"]
        tot["size_gb"] += row["size_gb"]
        out.append(row)
    total = {"variant": "TOTAL", "phase3_runs": sum(runs.values()), "train_gpu_h": round(tot["train_gpu_h"], 1),
             "size_gb": round(tot["size_gb"], 1), "phase4_replay_gpu_h": round(tot["train_gpu_h"], 1),
             "molab_sessions_12h": math.ceil(2 * tot["train_gpu_h"] / 12),
             "size_ok": tot["size_gb"] <= G2_MAX_GB}
    return out + [total]

"""Running the stages over many runs: queue, worker processes, upload, resume, curves.

Stage E runs in N worker processes (trajectory.worker), each claiming runs from a shared
queue; this process polls for finished raw files and uploads them to HF in batches, so a
session can be stopped at any point and the next one resumes from what is on HF
(`resume`). The same code serves the recorded stage (CPU) and the replay stage (GPU).
Stage Q (`build_curves`) runs in this process: it only reads raw files.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import layout
from .quantities import compute_curves
from .rawfile import load_raw

Log = Callable[[str], None]
_RUN_KEYS = ("run_id", "app", "variant", "optim", "wall_s")


def done(work: Path, stage: str, run_id: str) -> bool:
    return layout.raw_path(work, stage, run_id).exists()


def resume(remote, work: Path, stage: str, run_ids: set, log: Log = print) -> int:
    """Download raw files of this stage already on the remote (an earlier session)."""
    try:
        files = remote.list_files(f"raw/{stage}")
    except Exception as e:
        log(f"nothing to resume for {stage} ({type(e).__name__})")
        return 0
    got = 0
    for path in files:
        rid = Path(path).name[: -len(".npz")]
        if rid in run_ids and not done(work, stage, rid):
            remote.download(path, layout.raw_path(work, stage, rid))
            got += 1
    log(f"{stage}: resumed {got} raw file(s) from the remote")
    return got


def _kill_stale(pid_file: Path, log: Log) -> None:
    """Stop workers of an earlier call that outlived a restarted notebook kernel. Only
    processes still running trajectory.worker are touched (checked via /proc on Linux)."""
    if not pid_file.exists():
        return
    try:
        pids = json.loads(pid_file.read_text())
    except ValueError:
        pids = []
    killed = []
    for pid in pids:
        cmd = Path(f"/proc/{pid}/cmdline")
        try:
            if cmd.exists() and b"trajectory.worker" in cmd.read_bytes():
                os.kill(int(pid), signal.SIGTERM)
                killed.append(pid)
        except (OSError, ValueError):
            pass
    if killed:
        time.sleep(5)
        for pid in killed:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
        log(f"stopped {len(killed)} leftover worker(s): {killed}")
    pid_file.unlink()


def launch(stage: str, runs: list[dict], work: Path, project_dir: Path, workers: int, source: dict,
           device: str = "cpu", verify: bool = True, allow_env_mismatch: bool = False,
           mem_fraction: Optional[float] = None, out_remote=None, upload_every: int = 20, poll: float = 15.0,
           env: Optional[dict] = None, log: Log = print) -> list[dict]:
    """Extract `stage` ("recorded" | "replay") for every run without a raw file, with
    `workers` processes. source: {"kind": "hf", "repo": ...} or {"kind": "dir", "root": ...}.
    The HF token, if needed, goes in env["HF_TOKEN"]. Returns the status rows."""
    if stage not in layout.STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    work = Path(work)
    for d in (f"claims_{stage}", f"status_{stage}", f"logs_{stage}"):
        (work / d).mkdir(parents=True, exist_ok=True)
    _kill_stale(work / f"pids_{stage}.json", log)
    for d in (f"claims_{stage}", f"status_{stage}"):  # left by an interrupted call
        for p in (work / d).iterdir():
            p.unlink()
    spec = {"stage": stage, "work": str(work), "source": source, "device": device, "verify": verify,
            "allow_env_mismatch": allow_env_mismatch, "mem_fraction": mem_fraction}
    spec_path = work / f"spec_{stage}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    (work / f"queue_{stage}.json").write_text(json.dumps([{k: r.get(k) for k in _RUN_KEYS} for r in runs],
                                                         default=str))
    pending: dict[str, Path] = {}

    def flush() -> None:
        if out_remote is not None and pending:
            out_remote.commit(dict(pending), message=f"phase5: {len(pending)} {stage} raw file(s)")
            log(f"uploaded {len(pending)} raw file(s)")
            pending.clear()

    if out_remote is not None:  # finished locally by an earlier call, never uploaded
        try:
            have = set(out_remote.list_files(f"raw/{stage}"))
        except Exception:
            have = set()
        for r in runs:
            rel = layout.raw_rel(stage, r["run_id"])
            if rel not in have and done(work, stage, r["run_id"]):
                pending[rel] = layout.raw_path(work, stage, r["run_id"])

    todo = [r["run_id"] for r in runs if not done(work, stage, r["run_id"])]
    log(f"{stage}: {len(runs) - len(todo)} of {len(runs)} runs done; {len(todo)} to do with "
        f"{min(workers, len(todo))} worker(s)")
    procs = []
    for i in range(min(workers, len(todo))):
        fh = open(work / f"logs_{stage}" / f"w{i}.log", "a")
        procs.append((subprocess.Popen([sys.executable, "-m", "trajectory.worker", str(spec_path), str(i)],
                                       cwd=str(project_dir), env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
    (work / f"pids_{stage}.json").write_text(json.dumps([p.pid for p, _ in procs]))
    t0, seen = time.time(), set()
    while procs:
        running = [p for p, _ in procs if p.poll() is None]
        new = [rid for rid in todo if rid not in seen and done(work, stage, rid)]
        seen.update(new)
        for rid in new:
            pending[layout.raw_rel(stage, rid)] = layout.raw_path(work, stage, rid)
        if len(pending) >= upload_every:
            flush()
        if new or not running:
            el = time.time() - t0
            eta = el / len(seen) * (len(todo) - len(seen)) if seen else float("nan")
            log(f"{stage}: {len(seen)}/{len(todo)} done, {el / 60:.1f} min, ETA {eta / 60:.0f} min, "
                f"{len(running)} worker(s) running")
        if not running:
            break
        time.sleep(poll)
    for _, fh in procs:
        fh.close()
    flush()
    bad = [i for i, (p, _) in enumerate(procs) if p.returncode != 0]
    if bad:
        log(f"worker(s) {bad} exited with an error; log tail:\n"
            + (work / f"logs_{stage}" / f"w{bad[0]}.log").read_text()[-3000:])
    status = []
    for f in sorted((work / f"status_{stage}").glob("*.jsonl")):
        status += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return status


def build_curves(table: pd.DataFrame, work: Path, log: Log = print) -> tuple[pd.DataFrame, list[dict]]:
    """Stage Q for every run with a recorded raw file (plus its replay file if present).
    Returns (curves in long format with run_id, problems). Writes work/curves.parquet."""
    work = Path(work)
    parts, problems = [], []
    for rid in table["run_id"]:
        rec_p = layout.raw_path(work, "recorded", rid)
        if not rec_p.exists():
            continue
        rep_p = layout.raw_path(work, "replay", rid)
        try:
            df = compute_curves(load_raw(rec_p), load_raw(rep_p) if rep_p.exists() else None)
        except Exception as e:
            problems.append({"run_id": rid, "error": f"{type(e).__name__}: {e}"})
            continue
        df.insert(0, "run_id", rid)
        parts.append(df)
    curves = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(curves):
        curves["metric"] = curves["metric"].astype("category")
        curves["variant"] = curves["variant"].astype("category")
        curves.to_parquet(work / layout.CURVES, index=False)
    log(f"curves: {len(parts)} run(s), {len(curves)} rows, {len(problems)} problem(s)")
    return curves, problems

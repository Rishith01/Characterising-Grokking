"""Run a sweep and upload it as it goes (DESIGN.md §6 and §12; used by Phase 2b and Phase 3).

    expand -> skip runs already in the remote manifest -> train (workers in parallel)
           -> one-interval replay check per run, in the same worker slot
           -> summary row + compact curves (before the local copy is deleted)
           -> push to the remote in batches, verify, delete locally

Uploads run in a background thread, so training never waits on Hugging Face. Re-running
the same sweep in a new Molab session resumes it: runs already in the manifest are
skipped.

Usage: HF_TOKEN=... python -m scripts.sweep <sweep.yaml> --workers 16 --repo advay333/tfs --prefix phase3
       (no --repo: train and check only, keep runs locally)
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
from pathlib import Path
from typing import Optional

from tfs.evaluation.run_summary import run_curves, run_summary
from tfs.storage.hub import HfRemote, PrefixRemote, Remote, push_runs, read_manifest
from tfs.sweep.grid import expand
from tfs.sweep.launcher import LaunchResult, run_parallel


class _Uploader:
    """Pushes finished runs in batches from one background thread (one writer of the
    manifest at a time)."""

    def __init__(self, remote: Remote, batch_size: int, log):
        self.remote, self.batch_size, self.log = remote, batch_size, log
        self.q: queue.Queue = queue.Queue()
        self.pushed: list[str] = []
        self.failed: list = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def put(self, run_dir: Path):
        self.q.put(run_dir)

    def close(self):
        self.q.put(None)
        self.thread.join()

    def _loop(self):
        batch, closing = [], False
        while not closing:
            item = self.q.get()
            if item is None:
                closing = True
            else:
                batch.append(item)
            if batch and (closing or len(batch) >= self.batch_size):
                self._push(batch)
                batch = []

    def _push(self, batch):
        try:
            res = push_runs(self.remote, batch, delete_local=True, batch_size=len(batch), log=self.log)
            self.pushed += res["pushed"]
            self.failed += res["verify_failed"] + [(str(d), "incomplete") for d in res["skipped_incomplete"]]
        except Exception as e:  # keep training; the runs stay on disk for a later push
            self.log(f"upload of {len(batch)} run(s) failed, kept locally: {e!r}")
            self.failed += [(str(d), repr(e)) for d in batch]


def sweep(configs, runs_root: Path, device: str, workers: int, remote: Optional[Remote] = None,
          batch_size: int = 10, replay_check: bool = True, log=print,
          on_row=None) -> tuple[list[dict], dict, dict]:
    """Returns (summary rows, {run_id: curves}, upload info). `on_row(row, curves)` is
    called as each run is summarized (e.g. to checkpoint the tables)."""
    done = read_manifest(remote) if remote is not None else {}
    todo = [c for c in configs if c.run_id not in done]
    if len(todo) < len(configs):
        log(f"{len(configs) - len(todo)} run(s) already on the remote, skipped")
    rows, curves = [], {}
    uploader = _Uploader(remote, batch_size, log) if remote is not None else None

    def on_result(res: LaunchResult):
        if res.status in ("complete", "diverged"):
            try:
                row = run_summary(res.run_dir)
                curves[res.run_id] = run_curves(res.run_dir)
            except Exception as e:
                row = {"run_id": res.run_id, "status": res.status, "error": repr(e)}
            row["replay_level"] = res.replay_level or row.get("replay_level")
            row["launch_wall_s"] = round(res.wall_s, 1)
            rows.append(row)
            if on_row is not None:
                on_row(row, curves.get(res.run_id))
            if uploader is not None:
                uploader.put(res.run_dir)
        elif res.status == "failed":
            rows.append({"run_id": res.run_id, "status": "failed", "error": res.log[-500:]})

    try:
        run_parallel(todo, runs_root, device, workers, log=log, replay_check=replay_check, on_result=on_result)
    finally:
        if uploader is not None:
            uploader.close()
    info = {"skipped_on_remote": len(configs) - len(todo),
            "pushed": len(uploader.pushed) if uploader else 0,
            "upload_failed": uploader.failed if uploader else []}
    return rows, curves, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--repo", default=None, help="HF dataset repo; token from HF_TOKEN")
    ap.add_argument("--prefix", default=None, help="sub-folder in the repo, e.g. phase3")
    ap.add_argument("--no-replay-check", action="store_true")
    ap.add_argument("--out", default=None, help="write summary rows and curves here (JSON)")
    args = ap.parse_args()

    remote = None
    if args.repo:
        remote = HfRemote(args.repo, token=os.environ.get("HF_TOKEN"), create=True)
        if args.prefix:
            remote = PrefixRemote(remote, args.prefix)
    rows, curves, info = sweep(expand(args.sweep), Path(args.runs_root), args.device, args.workers, remote,
                               replay_check=not args.no_replay_check)
    print(json.dumps(info, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "curves": curves, "info": info}, indent=2, default=str))


if __name__ == "__main__":
    main()

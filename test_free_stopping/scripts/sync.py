"""Manifest-driven Hugging Face synchronization.

Usage:
    python -m scripts.sync push   --repo advay333/<name> [--runs-root runs] [--keep-local]
    python -m scripts.sync pull   --repo advay333/<name> --app <app> --run-id <id> [--parts telemetry keyframes ...]
    python -m scripts.sync status --repo advay333/<name>

The token is read from the HF_TOKEN environment variable. `push` uploads every complete
local run, verifies each file on the remote, and deletes the local copy (unless
--keep-local), so the remote always holds the latest version (PLAN.md workflow rules).
"""
from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path

from tfs.storage import layout
from tfs.storage.hub import HfRemote, pull_run, push_runs, read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull", "status"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument("--app")
    ap.add_argument("--run-id")
    ap.add_argument("--parts", nargs="*", default=None, choices=list(layout.PARTS))
    args = ap.parse_args()

    remote = HfRemote(args.repo, token=os.environ.get("HF_TOKEN"), create=args.action == "push")
    if args.action == "push":
        run_dirs = sorted(p.parent for p in Path(args.runs_root).glob(f"*/*/{layout.DONE}"))
        res = push_runs(remote, run_dirs, delete_local=not args.keep_local)
        print({k: len(v) for k, v in res.items()})
    elif args.action == "pull":
        if not (args.app and args.run_id):
            ap.error("pull needs --app and --run-id")
        print(pull_run(remote, args.app, args.run_id, Path(args.runs_root), args.parts))
    else:
        manifest = read_manifest(remote)
        counts = collections.Counter((e.get("app"), e.get("status"), e.get("replay_level")) for e in manifest.values())
        print(f"{len(manifest)} runs")
        for (app, status, level), n in sorted(counts.items(), key=str):
            print(f"  {app:24s} {status:10s} {level:12s} {n}")


if __name__ == "__main__":
    main()

"""Determinism gate G1-det (PLAN.md §6, Phase 1).

For every config in a sweep file:
  1. train it twice, independently (separate processes);
  2. check the two runs are bit-identical (keyframe hashes, telemetry, fingerprints,
     predictions);
  3. replay run A end to end on the same device and record its level (L0 / L1 / failed)
     and worst errors.

Usage: python -m scripts.det_check configs/sweeps/phase1_det.yaml --device cuda --workers 8 \\
           --runs-root /tmp/det --out det_report.json
Exit code 1 if any pair differs or any replay fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from tfs.storage import layout
from tfs.sweep.grid import expand
from tfs.sweep.launcher import replay_parallel, run_parallel

COMPARED = (layout.KEYFRAME_HASHES, layout.TELEMETRY, layout.FINGERPRINTS, layout.TRAIN_PREDS)


def det_check(sweep: Path, device: str, workers: int, runs_root: Path, log=print,
              steps: int | None = None, keyframe_every: int | None = None) -> list[dict]:
    configs = expand(sweep)
    if steps is not None or keyframe_every is not None:
        # Shorter pass (e.g. CPU): keep log/summary cadence, shrink horizon and keyframes.
        configs = [replace(c, steps=steps or c.steps,
                           record=replace(c.record, keyframe_every=keyframe_every or c.record.keyframe_every))
                   for c in configs]
    res_a = {r.run_id: r for r in run_parallel(configs, runs_root / "a", device, workers, log)}
    res_b = {r.run_id: r for r in run_parallel(configs, runs_root / "b", device, workers, log)}
    replay_parallel([r.run_dir for r in res_a.values() if r.status != "failed"], device, workers, True, log)

    rows = []
    for cfg in configs:
        a, b = res_a[cfg.run_id], res_b[cfg.run_id]
        row = {"run_id": cfg.run_id, "app": cfg.app, "optim": cfg.optim.name, "device": device,
               "status_a": a.status, "status_b": b.status, "wall_s": round(a.wall_s, 1)}
        if a.status == "failed" or b.status == "failed":
            row.update(identical=False, replay_level="n/a", note=(a.log or b.log)[-500:])
            rows.append(row)
            continue
        sa = json.loads((a.run_dir / layout.CHECKSUMS).read_text())
        sb = json.loads((b.run_dir / layout.CHECKSUMS).read_text())
        row["identical"] = all(sa[n] == sb[n] for n in COMPARED)
        row["differing_files"] = [n for n in COMPARED if sa[n] != sb[n]]
        rep_path = a.run_dir / layout.REPLAY_REPORT
        rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
        row["replay_level"] = rep.get("level", "missing")
        row["max_err"] = rep.get("max_err")
        row["first_failure"] = rep.get("first_failure")
        row["log_steps_checked"] = len(rep.get("steps_checked", []))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--runs-root", default="det_runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override every config's horizon")
    ap.add_argument("--keyframe-every", type=int, default=None)
    args = ap.parse_args()

    rows = det_check(Path(args.sweep), args.device, args.workers, Path(args.runs_root),
                     steps=args.steps, keyframe_every=args.keyframe_every)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
    ok = True
    for r in rows:
        good = r["identical"] and r["replay_level"] in ("L0", "L1")
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {r['app']:16s} {r['optim']:13s} identical={r['identical']} "
              f"replay={r['replay_level']} wall={r['wall_s']}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

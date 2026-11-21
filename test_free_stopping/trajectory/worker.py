"""One extraction worker process: `python -m trajectory.worker <spec.json> <index>`.

Started by trajectory.runner.launch. Reads the queue, claims runs one at a time with an
exclusive file create (so N workers share the queue without doing anything twice), and
for each run: downloads the parts the stage needs, extracts, writes the raw file
atomically, deletes the download, and appends a status line. The HF token, if any,
comes from the HF_TOKEN environment variable and is never written to disk.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import layout
from .rawfile import save_raw

NEEDS = {"recorded": {"telemetry", "summaries"}, "replay": {"models"}}


def claim(work: Path, stage: str, run_id: str) -> bool:
    d = Path(work) / f"claims_{stage}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / run_id, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _source(spec: dict, app: str, idx: int):
    src, stage = spec["source"], spec["stage"]
    device = spec["device"] if stage == "replay" else "cpu"
    kw = {"verify_replay": bool(spec.get("verify", True)) and stage == "replay"}
    if src["kind"] == "dir":
        from stopping.tfs_source import DirRunSource
        return DirRunSource(Path(src["root"]), device=device, **kw)
    if app == "mnist" and stage == "replay":
        from tfs.tasks.mnist import ensure_mnist
        ensure_mnist()  # replay rebuilds the training set from the seed
    from stopping.tfs_source import HfRunSource
    return HfRunSource(src["repo"], f"phase3/{app}", work=Path(spec["work"]) / f"runs_{stage}_w{idx}",
                       device=device, token=os.environ.get("HF_TOKEN"), **kw)


def process(spec: dict, run: dict, sources: dict, idx: int) -> dict:
    stage, work = spec["stage"], Path(spec["work"])
    rid, app = run["run_id"], run["app"]
    t0, err = time.time(), ""
    try:
        if app not in sources:
            sources[app] = _source(spec, app, idx)
        with sources[app].open(rid, NEEDS[stage]) as h:
            if stage == "recorded":
                from .extract.recorded import extract_recorded
                arrays, meta = extract_recorded(h)
            else:
                from .extract.replay import extract_replay
                arrays, meta = extract_replay(h.run_dir, device=spec["device"], verify=bool(spec.get("verify", True)),
                                              allow_env_mismatch=bool(spec.get("allow_env_mismatch", False)))
        save_raw(layout.raw_path(work, stage, rid), arrays, meta)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    return {"run_id": rid, "app": app, "stage": stage, "ok": not err, "seconds": round(time.time() - t0, 1),
            "error": err, "worker": idx}


def main(spec_path: str, idx: int) -> None:
    spec = json.loads(Path(spec_path).read_text())
    stage, work = spec["stage"], Path(spec["work"])
    queue = json.loads((work / f"queue_{stage}.json").read_text())
    status_f = work / f"status_{stage}" / f"w{idx}.jsonl"
    status_f.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[w{idx} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    torch = None
    if stage == "replay":
        from tfs.training import determinism
        determinism.apply()
        import torch
        if torch.cuda.is_available() and spec.get("mem_fraction"):
            torch.cuda.set_per_process_memory_fraction(float(spec["mem_fraction"]))
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    sources: dict = {}
    for run in queue:
        rid = run["run_id"]
        if layout.raw_path(work, stage, rid).exists() or not claim(work, stage, rid):
            continue
        for attempt in range(3):  # several replays peaking at once can run out of GPU memory
            row = process(spec, run, sources, idx)
            if row["ok"] or "out of memory" not in row["error"].lower() or torch is None:
                break
            torch.cuda.empty_cache()
            log(f"{rid}: out of GPU memory, retry {attempt + 1} in 60 s")
            time.sleep(60)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        with open(status_f, "a") as f:
            f.write(json.dumps(row) + "\n")
        log(f"{rid}: {'ok' if row['ok'] else 'FAILED ' + row['error']} in {row['seconds']} s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

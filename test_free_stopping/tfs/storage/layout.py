"""File names and paths: one source of truth, shared by writer, reader and hub."""
from __future__ import annotations

import hashlib
from pathlib import Path

META = "meta.json"
TELEMETRY = "telemetry.parquet"
TRAIN_PREDS = "train_preds.npy"
SUMMARIES = "summaries.safetensors"
FINGERPRINTS = "fingerprints.parquet"
KEYFRAMES = "keyframes.safetensors"
KEYFRAME_HASHES = "keyframe_hashes.json"
ORACLE = "oracle.parquet"  # quarantined: read only by tfs/evaluation/oracle.py
CHECKSUMS = "checksums.json"  # sha256 of every immutable file above
DONE = "DONE"  # written last; presence = run complete
REPLAY_REPORT = "replay_report.json"  # mutable, outside checksums: latest replay verification

IMMUTABLE_FILES = (META, TELEMETRY, TRAIN_PREDS, SUMMARIES, FINGERPRINTS, KEYFRAMES,
                   KEYFRAME_HASHES, ORACLE)

# Download parts (hub.pull_run): name -> files. meta + checksums are always fetched.
PARTS = {
    "telemetry": (TELEMETRY,),
    "preds": (TRAIN_PREDS,),
    "summaries": (SUMMARIES,),
    "fingerprints": (FINGERPRINTS,),
    "keyframes": (KEYFRAMES, KEYFRAME_HASHES),
    "oracle": (ORACLE,),
}
ALWAYS = (META, CHECKSUMS, DONE)

MANIFEST = "manifest.json"  # repo root on the remote


def run_dir(runs_root: Path, app: str, run_id: str) -> Path:
    return Path(runs_root) / app / run_id


def remote_prefix(app: str, run_id: str) -> str:
    return f"runs/{app}/{run_id}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

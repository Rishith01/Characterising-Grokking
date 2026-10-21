"""Remote run storage: push runs, verify the remote copy, delete locally, pull back
(DESIGN.md §5, "hub.py and manifest.py").

Two interchangeable remotes:
    HfRemote     a Hugging Face dataset repo (production; used from Molab)
    LocalRemote  a plain directory with the same interface (tests)
and PrefixRemote, which puts any remote's files under a sub-folder, so one repo can hold
every phase (`advay333/tfs`: phase2/, phase3/, ...), each with its own runs/ and manifest.

A local run directory is deleted only after every file is confirmed on the remote with
a matching content hash. The token is passed in by the caller (a notebook password
field or the HF_TOKEN environment variable) and is never written anywhere.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from . import layout
from .reader import RunReader


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    sha256: Optional[str]  # content sha256, when the remote exposes it (LFS / local)
    git_sha1: Optional[str]  # git blob id, for small non-LFS files on HF


class Remote(ABC):
    @abstractmethod
    def commit(self, files: dict[str, Path], message: str, deletes: Iterable[str] = ()) -> str:
        """Upload {path_in_repo: local_path} and delete `deletes` in one atomic commit.
        A delete path ending in "/" removes that whole folder. Returns a commit id."""

    @abstractmethod
    def list_files(self, prefix: str) -> dict[str, RemoteFile]: ...

    @abstractmethod
    def download(self, path: str, dest: Path) -> Path: ...

    def read_json(self, path: str) -> Optional[dict]:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                local = self.download(path, Path(tmp) / Path(path).name)
            except FileNotFoundError:
                return None
            return json.loads(local.read_text())


class LocalRemote(Remote):
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def commit(self, files: dict[str, Path], message: str, deletes: Iterable[str] = ()) -> str:
        for rel in deletes:
            target = self.root / rel.rstrip("/")
            if rel.endswith("/"):
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        for rel, src in files.items():
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        self._n += 1
        return f"local-{self._n}"

    def list_files(self, prefix: str) -> dict[str, RemoteFile]:
        base = self.root / prefix
        if not base.exists():
            return {}
        out = {}
        for p in base.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                out[rel] = RemoteFile(rel, p.stat().st_size, layout.file_sha256(p), None)
        return out

    def download(self, path: str, dest: Path) -> Path:
        src = self.root / path
        if not src.exists():
            raise FileNotFoundError(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest


class HfRemote(Remote):
    def __init__(self, repo_id: str, token: Optional[str], create: bool = True, private: bool = False):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)
        if create:
            self.api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    def commit(self, files: dict[str, Path], message: str, deletes: Iterable[str] = ()) -> str:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        ops = [CommitOperationDelete(path_in_repo=rel, is_folder=rel.endswith("/")) for rel in deletes]
        ops += [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(src)) for rel, src in files.items()]
        info = self.api.create_commit(self.repo_id, operations=ops, commit_message=message, repo_type="dataset")
        return info.oid

    def squash_history(self, message: str) -> None:
        """Collapse the repo's history into one commit. Deleted files keep using storage
        while any commit still references them; this is what actually frees it."""
        self.api.super_squash_history(self.repo_id, repo_type="dataset", commit_message=message)

    def list_files(self, prefix: str) -> dict[str, RemoteFile]:
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.hf_api import RepoFile

        try:
            entries = self.api.list_repo_tree(self.repo_id, path_in_repo=prefix, recursive=True,
                                              expand=True, repo_type="dataset")
            out = {}
            for e in entries:
                if isinstance(e, RepoFile):
                    sha = e.lfs.sha256 if e.lfs is not None else None
                    out[e.path] = RemoteFile(e.path, e.size, sha, e.blob_id)
            return out
        except EntryNotFoundError:
            return {}

    def download(self, path: str, dest: Path) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        with tempfile.TemporaryDirectory() as tmp:
            try:
                got = hf_hub_download(self.repo_id, path, repo_type="dataset", token=self.token, local_dir=tmp)
            except EntryNotFoundError as e:
                raise FileNotFoundError(path) from e
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(got, dest)
        return dest


class PrefixRemote(Remote):
    """Another remote, seen through a sub-folder: every path gets `prefix/` in front.
    Run prefixes, the manifest and verification all work unchanged inside it."""

    def __init__(self, inner: Remote, prefix: str):
        self.inner = inner
        self.prefix = prefix.strip("/")

    def _p(self, rel: str) -> str:
        return f"{self.prefix}/{rel}"

    def commit(self, files: dict[str, Path], message: str, deletes: Iterable[str] = ()) -> str:
        return self.inner.commit({self._p(k): v for k, v in files.items()}, message,
                                 deletes=[self._p(d) for d in deletes])

    def list_files(self, prefix: str) -> dict[str, RemoteFile]:
        n = len(self.prefix) + 1
        return {k[n:]: RemoteFile(k[n:], f.size, f.sha256, f.git_sha1)
                for k, f in self.inner.list_files(self._p(prefix)).items()}

    def download(self, path: str, dest: Path) -> Path:
        return self.inner.download(self._p(path), dest)


# ---- manifest --------------------------------------------------------------------------

def _manifest_entry(reader: RunReader, files: dict[str, Path]) -> dict:
    replay = _read_json(reader.run_dir / layout.REPLAY_REPORT)
    return {
        "app": reader.meta["app"],
        "group_id": reader.meta["group_id"],
        "seed": reader.meta["config"]["seed"],
        "status": reader.status,
        "replay_level": replay["level"] if replay else "unverified",
        "n_files": len(files),
        "size_bytes": sum(p.stat().st_size for p in files.values()),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }


def read_manifest(remote: Remote) -> dict:
    return remote.read_json(layout.MANIFEST) or {}


# ---- push / verify / pull ----------------------------------------------------------------

def run_files(run_dir: Path) -> dict[str, Path]:
    """Every file of a complete run, keyed by its path on the remote."""
    reader = RunReader(run_dir)
    prefix = layout.remote_prefix(reader.meta["app"], reader.meta["run_id"])
    names = list(layout.IMMUTABLE_FILES) + [layout.CHECKSUMS, layout.DONE]
    if (run_dir / layout.REPLAY_REPORT).exists():
        names.append(layout.REPLAY_REPORT)
    return {f"{prefix}/{n}": run_dir / n for n in names}


def verify_remote(remote: Remote, run_dir: Path) -> list[str]:
    """Remote paths that are missing or whose content differs from the local run.

    Uses the content hash the remote reports when it has one (LFS sha256; local
    remote). Otherwise, e.g. small files known only by git blob id, or Xet-stored files
    without an LFS sha256, it downloads the file and hashes it: a file is never
    declared verified on size alone.
    """
    files = run_files(run_dir)
    prefix = next(iter(files)).rsplit("/", 1)[0]
    listed = remote.list_files(prefix)
    bad = []
    for rel, local in files.items():
        rf = listed.get(rel)
        if rf is None or rf.size != local.stat().st_size:
            bad.append(rel)
            continue
        if rf.sha256 is not None:
            ok = rf.sha256 == layout.file_sha256(local)
        elif rf.git_sha1 is not None and rf.git_sha1 == _git_blob_sha1(local):
            ok = True
        else:
            ok = _downloaded_sha256(remote, rel) == layout.file_sha256(local)
        if not ok:
            bad.append(rel)
    return bad


def _downloaded_sha256(remote: Remote, rel: str) -> Optional[str]:
    with tempfile.TemporaryDirectory() as tmp:
        try:
            return layout.file_sha256(remote.download(rel, Path(tmp) / "f"))
        except FileNotFoundError:
            return None


def push_runs(remote: Remote, run_dirs: Iterable[Path], delete_local: bool = True,
              batch_size: int = 10, log=print) -> dict:
    """Upload complete runs in batched commits, update the manifest in the same commit,
    verify every file, then delete the local copy (if delete_local)."""
    run_dirs = [Path(d) for d in run_dirs]
    result = {"pushed": [], "verify_failed": [], "skipped_incomplete": []}
    complete = []
    for d in run_dirs:
        (complete if (d / layout.DONE).exists() else result["skipped_incomplete"]).append(d)

    for i in range(0, len(complete), batch_size):
        batch = complete[i : i + batch_size]
        files: dict[str, Path] = {}
        per_run = {}
        for d in batch:
            rf = run_files(d)
            files.update(rf)
            per_run[d] = rf
        manifest = read_manifest(remote)
        with tempfile.TemporaryDirectory() as tmp:
            # The manifest is committed together with the runs, so it never lists a run
            # whose files are not on the remote. (The commit id lives in repo history.)
            for d, rf in per_run.items():
                reader = RunReader(d)
                manifest[reader.meta["run_id"]] = _manifest_entry(reader, rf)
            mpath = Path(tmp) / layout.MANIFEST
            mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            commit = remote.commit({**files, layout.MANIFEST: mpath},
                                   message=f"add {len(batch)} run(s)")
        log(f"committed {len(batch)} run(s): {commit}")

        for d in batch:
            bad = verify_remote(remote, d)
            run_id = RunReader(d).meta["run_id"]
            if bad:
                result["verify_failed"].append((str(d), bad))
                log(f"  {run_id}: remote verification FAILED for {bad}; local copy kept")
                continue
            result["pushed"].append(run_id)
            if delete_local:
                shutil.rmtree(d)
                log(f"  {run_id}: verified on remote, local copy deleted")
            else:
                log(f"  {run_id}: verified on remote")
    return result


def delete_runs(remote: Remote, run_ids: Iterable[str], log=print) -> list[str]:
    """Delete whole run folders and their manifest entries in one commit. Every run must
    be in the manifest (that is where its app comes from); unknown ids raise before
    anything is deleted. Returns the folders deleted."""
    run_ids = list(run_ids)
    manifest = read_manifest(remote)
    missing = [r for r in run_ids if r not in manifest]
    if missing:
        raise KeyError(f"not in the remote manifest: {missing}")
    folders = [layout.remote_prefix(manifest[r]["app"], r) + "/" for r in run_ids]
    for r in run_ids:
        del manifest[r]
    with tempfile.TemporaryDirectory() as tmp:
        mpath = Path(tmp) / layout.MANIFEST
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        commit = remote.commit({layout.MANIFEST: mpath}, message=f"delete {len(run_ids)} run(s)", deletes=folders)
    log(f"deleted {len(folders)} run folder(s) in {commit}")
    return folders


def pull_run(remote: Remote, app: str, run_id: str, runs_root: Path,
             parts: Optional[Iterable[str]] = None) -> Path:
    """Download a run (all parts by default) and check sha256 against checksums.json."""
    parts = list(layout.PARTS) if parts is None else list(parts)
    names = list(layout.ALWAYS) + [n for part in parts for n in layout.PARTS[part]]
    prefix = layout.remote_prefix(app, run_id)
    dest = layout.run_dir(runs_root, app, run_id)
    for n in names:
        remote.download(f"{prefix}/{n}", dest / n)
    try:
        remote.download(f"{prefix}/{layout.REPLAY_REPORT}", dest / layout.REPLAY_REPORT)
    except FileNotFoundError:
        pass
    bad = RunReader(dest).verify_checksums()
    if bad:
        raise IOError(f"{run_id}: checksum mismatch after download for {bad}")
    return dest


def update_replay_level(remote: Remote, run_dir: Path) -> str:
    """Push a run's replay_report.json and record its level in the manifest."""
    reader = RunReader(run_dir)
    report = _read_json(run_dir / layout.REPLAY_REPORT)
    if report is None:
        raise FileNotFoundError(f"{run_dir} has no {layout.REPLAY_REPORT}")
    manifest = read_manifest(remote)
    run_id = reader.meta["run_id"]
    manifest.setdefault(run_id, {})["replay_level"] = report["level"]
    manifest[run_id]["updated_utc"] = datetime.now(timezone.utc).isoformat()
    prefix = layout.remote_prefix(reader.meta["app"], run_id)
    with tempfile.TemporaryDirectory() as tmp:
        mpath = Path(tmp) / layout.MANIFEST
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return remote.commit({f"{prefix}/{layout.REPLAY_REPORT}": run_dir / layout.REPLAY_REPORT,
                              layout.MANIFEST: mpath}, message=f"replay level {run_id}")


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _read_json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.exists() else None

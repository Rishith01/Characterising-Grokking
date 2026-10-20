# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "safetensors",
#     "huggingface_hub",
#     "pyyaml",
# ]
# ///
"""Phase 0, gate G0 (PLAN.md §6): the full storage cycle on Molab.

1. Train the dummy run twice on CPU (and twice on CUDA if a GPU is present); check that
   the two trainings are bit-identical.
2. Replay each run end to end from its step-0 keyframe (the CPU run must reach L0).
3. Push the runs to a public Hugging Face dataset repo, verify every file remotely, and
   delete the local copies.
4. Pull them back, check checksums, replay again from the downloaded copy, push the
   replay level to the manifest, and delete the local copies again.

Fill in the HF token field (it is never stored), then press "Run G0".
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import shutil
    import subprocess
    import sys
    import traceback
    from pathlib import Path

    import marimo as mo

    mo.md(
        """
        # Phase 0, gate G0: storage, upload and replay cycle

        **Step 1:** check the repo location below and press **Clone / pull**.

        **Step 2:** enter your Hugging Face write token and press **Run G0**.
        """
    )
    return Path, json, mo, shutil, subprocess, sys, traceback


@app.cell
def _(Path, mo):
    _here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    _guess = _here.parent if (_here.parent / "tfs").is_dir() else Path.cwd() / "Characterising-Grokking" / "test_free_stopping"
    repo_url = mo.ui.text(value="https://github.com/Rishith01/Characterising-Grokking.git", label="git repo", full_width=True)
    project_dir = mo.ui.text(value=str(_guess), label="test_free_stopping directory", full_width=True)
    pull_button = mo.ui.run_button(label="Clone / pull")
    mo.vstack([repo_url, project_dir, pull_button])
    return project_dir, pull_button, repo_url


@app.cell
def _(Path, mo, project_dir, pull_button, repo_url, subprocess):
    _sp = subprocess
    mo.stop(not pull_button.value, mo.md("_Press **Clone / pull** to fetch the latest code._"))
    _proj = Path(project_dir.value)
    _repo_root = _proj.parent
    if not (_repo_root / ".git").exists():
        _out = _sp.run(["git", "clone", repo_url.value, str(_repo_root)], capture_output=True, text=True)
    else:
        _out = _sp.run(["git", "-C", str(_repo_root), "pull", "--ff-only"], capture_output=True, text=True)
    _head = _sp.run(["git", "-C", str(_repo_root), "log", "-1", "--oneline"], capture_output=True, text=True).stdout
    code_ready = (_proj / "tfs").is_dir()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**`tfs/` not found. Check the directory.**"))
    return (code_ready,)


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs-g0-test", label="HF dataset repo for this test", full_width=True)
    run_button = mo.ui.run_button(label="Run G0")
    mo.vstack([hf_token, hf_repo, run_button])
    return hf_repo, hf_token, run_button


@app.cell
def _(Path, code_ready, hf_repo, hf_token, json, mo, project_dir, run_button, shutil, sys, traceback):
    _json, _shutil, _sys, _tb, _Path = json, shutil, sys, traceback, Path

    mo.stop(not run_button.value, mo.md("_Press **Run G0** when ready._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))

    _proj = _Path(project_dir.value)
    if str(_proj) not in _sys.path:
        _sys.path.insert(0, str(_proj))

    from tfs.training import determinism as _det

    _det.apply()  # before any CUDA work
    import torch as _torch
    from tfs.config import RunConfig as _RunConfig
    from tfs.storage import layout as _layout
    from tfs.storage.hub import HfRemote as _HfRemote
    from tfs.storage.hub import pull_run as _pull, push_runs as _push, read_manifest as _manifest
    from tfs.storage.hub import update_replay_level as _update_level
    from tfs.training.replay import ReplayEngine as _Replay
    from tfs.training.trainer import Trainer as _Trainer

    results = []
    _log_lines = []

    def _check(name, ok, detail=""):
        results.append({"check": name, "pass": bool(ok), "detail": str(detail)})

    _work = _Path("/tmp/tfs_g0") if _Path("/tmp").exists() else _proj / "g0_work"
    _shutil.rmtree(_work, ignore_errors=True)
    _base = _RunConfig.from_yaml(_proj / "configs" / "runs" / "dummy.yaml")
    _devices = ["cpu"] + (["cuda"] if _torch.cuda.is_available() else [])
    _check("CUDA available", True, _torch.cuda.get_device_name(0) if "cuda" in _devices else "no GPU: CPU only")

    _to_push = {}
    try:
        for _dev in _devices:
            # different seed per device so the two runs have different run ids on the remote
            _cfg = _base.with_seed(0 if _dev == "cpu" else 1)
            _a = _Trainer(_cfg, _work / f"{_dev}_a", _dev).run()
            _b = _Trainer(_cfg, _work / f"{_dev}_b", _dev).run()
            _ha = _json.loads((_a / _layout.KEYFRAME_HASHES).read_text())
            _hb = _json.loads((_b / _layout.KEYFRAME_HASHES).read_text())
            _check(f"[{_dev}] two trainings bit-identical", _ha == _hb, f"{len(_ha)} keyframes")

            _eng = _Replay(_a, _dev)
            _rep = _eng.verify("all")
            _eng.write_report(_rep)
            _want = ("L0",) if _dev == "cpu" else ("L0", "L1")
            _check(f"[{_dev}] replay from step-0 keyframe", _rep.level in _want,
                   f"level={_rep.level}, max_err={_rep.max_err}, first_failure={_rep.first_failure}")
            _to_push[_dev] = (_a, _cfg.app, _cfg.run_id, _ha)
            _shutil.rmtree(_b)

        _remote = _HfRemote(hf_repo.value, token=hf_token.value, create=True, private=False)
        _res = _push(_remote, [v[0] for v in _to_push.values()], delete_local=True, log=_log_lines.append)
        _check("push + remote verification", not _res["verify_failed"] and len(_res["pushed"]) == len(_to_push), _res)
        _check("local copies deleted after push", all(not v[0].exists() for v in _to_push.values()))

        for _dev, (_a, _app, _rid, _ha) in _to_push.items():
            _back = _pull(_remote, _app, _rid, _work / "pulled")
            _check(f"[{_dev}] pull + checksums", True, _back)
            _hback = _json.loads((_back / _layout.KEYFRAME_HASHES).read_text())
            _check(f"[{_dev}] pulled keyframe hashes == trained", _hback == _ha)
            _eng = _Replay(_back, _dev)
            _rep = _eng.verify("all")
            _eng.write_report(_rep)
            _check(f"[{_dev}] replay from downloaded copy", _rep.level in (("L0",) if _dev == "cpu" else ("L0", "L1")),
                   f"level={_rep.level}")
            _update_level(_remote, _back)
            _shutil.rmtree(_back)

        _m = _manifest(_remote)
        _check("manifest has replay levels", all(_m.get(v[2], {}).get("replay_level") in ("L0", "L1") for v in _to_push.values()),
               {k: (e["status"], e["replay_level"]) for k, e in _m.items()})
    except Exception:
        _check("unexpected error", False, _tb.format_exc())
    finally:
        _shutil.rmtree(_work, ignore_errors=True)

    _all_ok = all(r["pass"] for r in results)
    mo.vstack([
        mo.md(f"## G0 {'PASSED' if _all_ok else 'FAILED'}"),
        mo.ui.table(results, selection=None),
        mo.md("Push log:\n```\n" + "\n".join(_log_lines) + "\n```"),
        mo.md("Copy this table back into the chat."),
    ])
    return (results,)


if __name__ == "__main__":
    app.run()

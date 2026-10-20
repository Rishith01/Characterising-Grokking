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
#     "matplotlib",
# ]
# ///
"""Phase 1 gates on Molab (PLAN.md §6).

G1-det  every application x every optimizer: two independent trainings must be
        bit-identical, and replay must reach L0 (or at least L1). 20k steps on CUDA,
        plus a short 1k-step pass on CPU.
G1      reproductions of the published grokking setups (configs/sweeps/phase1_repro.yaml),
        1e5 steps each on CUDA, plotted as train/test accuracy and weight norm.

Results go to the HF dataset repo (default advay333/tfs-phase1):
    runs/...                     the reproduction runs (local copies deleted after verification)
    phase1/det_{cuda,cpu}.json   determinism reports
    phase1/plots/*.png           reproduction plots
Determinism runs are not uploaded; they are deleted after their report is written.
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
    import time
    import traceback
    from pathlib import Path

    import marimo as mo

    mo.md(
        """
        # Phase 1: determinism gate (G1-det) and grokking reproductions (G1)

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token, choose what to run, and press **Run phase 1**.
           Everything takes roughly 1–2 hours on the Blackwell GPU at 16 workers. GPU utilization
           is logged every 30 s and summarized at the end.
        3. Copy the two tables at the bottom back into the chat. The plots are uploaded to HF.
        """
    )
    return Path, json, mo, shutil, subprocess, sys, time, traceback


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
    mo.stop(not pull_button.value, mo.md("_Press **Clone / pull** to fetch the latest code._"))
    _proj = Path(project_dir.value)
    _root = _proj.parent
    if not (_root / ".git").exists():
        _out = subprocess.run(["git", "clone", repo_url.value, str(_root)], capture_output=True, text=True)
    else:
        _out = subprocess.run(["git", "-C", str(_root), "pull", "--ff-only"], capture_output=True, text=True)
    _head = subprocess.run(["git", "-C", str(_root), "log", "-1", "--oneline"], capture_output=True, text=True).stdout
    code_ready = (_proj / "tfs").is_dir() and (_proj / "configs" / "sweeps" / "phase1_det.yaml").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Phase 1 code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs-phase1", label="HF dataset repo", full_width=True)
    workers = mo.ui.number(start=1, stop=32, value=16, label="parallel workers (GPU runs)")
    do_det_cuda = mo.ui.checkbox(value=True, label="G1-det on CUDA (16 configs x 2 trainings, 20k steps)")
    do_det_cpu = mo.ui.checkbox(value=True, label="G1-det on CPU (16 configs x 2 trainings, 1k steps)")
    do_repro = mo.ui.checkbox(value=True, label="G1 reproductions (10 runs, 1e5 steps, CUDA)")
    run_button = mo.ui.run_button(label="Run phase 1")
    mo.vstack([hf_token, hf_repo, workers, do_det_cuda, do_det_cpu, do_repro, run_button])
    return do_det_cpu, do_det_cuda, do_repro, hf_repo, hf_token, run_button, workers


@app.cell
def _(Path, code_ready, do_det_cpu, do_det_cuda, do_repro, hf_repo, hf_token, json, mo,
      project_dir, run_button, shutil, sys, time, traceback, workers):
    mo.stop(not run_button.value, mo.md("_Press **Run phase 1** when ready._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))

    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from tfs.training import determinism as _det

    _det.apply()
    import torch as _torch
    from scripts.det_check import det_check as _det_check
    from tfs.evaluation.plots import plot_runs as _plot_runs
    from tfs.storage.hub import HfRemote as _HfRemote, push_runs as _push
    from tfs.sweep.grid import expand as _expand
    from tfs.sweep.gpumon import GpuMonitor as _GpuMonitor
    from tfs.sweep.launcher import replay_parallel as _replay_parallel, run_parallel as _run_parallel
    from tfs.tasks.mnist import ensure_mnist as _ensure_mnist

    assert _torch.cuda.is_available(), "phase 1 needs the GPU"
    _work = Path("/tmp/tfs_phase1")
    shutil.rmtree(_work, ignore_errors=True)
    _work.mkdir(parents=True)
    _ensure_mnist()  # download once, before parallel workers start
    _remote = _HfRemote(hf_repo.value, token=hf_token.value, create=True, private=False)
    _n = int(workers.value)
    det_rows, repro_rows, plot_paths, errors = [], [], [], []
    _uploads = {}

    def _log(msg):
        print(msg, flush=True)

    _t0 = time.time()
    _mon = _GpuMonitor(_work / "gpu_util.csv", every_s=30).__enter__()
    try:
        if do_det_cuda.value:
            _log("== G1-det CUDA ==")
            _rows = _det_check(_proj / "configs/sweeps/phase1_det.yaml", "cuda", _n, _work / "det_cuda", _log)
            det_rows += _rows
            (_work / "det_cuda.json").write_text(json.dumps(_rows, indent=2))
            _uploads["phase1/det_cuda.json"] = _work / "det_cuda.json"
            shutil.rmtree(_work / "det_cuda", ignore_errors=True)
        if do_det_cpu.value:
            _log("== G1-det CPU ==")
            _rows = _det_check(_proj / "configs/sweeps/phase1_det.yaml", "cpu", min(_n, 8), _work / "det_cpu", _log,
                               steps=1000, keyframe_every=500)
            det_rows += _rows
            (_work / "det_cpu.json").write_text(json.dumps(_rows, indent=2))
            _uploads["phase1/det_cpu.json"] = _work / "det_cpu.json"
            shutil.rmtree(_work / "det_cpu", ignore_errors=True)
        if do_repro.value:
            _log("== G1 reproductions ==")
            _cfgs = _expand(_proj / "configs/sweeps/phase1_repro.yaml")
            _res = _run_parallel(_cfgs, _work / "repro", "cuda", _n, _log)
            _ok = [r for r in _res if r.status in ("complete", "diverged")]
            _replay_parallel([r.run_dir for r in _ok], "cuda", _n, full=False, log=_log)
            for _r in _res:
                _rep_path = _r.run_dir / "replay_report.json"
                _rep = json.loads(_rep_path.read_text()) if _rep_path.exists() else {}
                repro_rows.append({"run_id": _r.run_id, "status": _r.status, "wall_s": round(_r.wall_s),
                                   "replay_check": _rep.get("level", "n/a")})
            for _app in sorted({c.app for c in _cfgs}):
                _dirs = [r.run_dir for r in _ok if r.run_id.startswith(_app + "__")]
                if _dirs:
                    _p = _plot_runs(_dirs, _work / "plots" / f"repro_{_app}.png")
                    plot_paths.append(_p)
                    _uploads[f"phase1/plots/{_p.name}"] = _p
            _push_res = _push(_remote, [r.run_dir for r in _ok], delete_local=True, log=_log)
            repro_rows.append({"run_id": "(push)", "status": str({k: len(v) for k, v in _push_res.items()}),
                               "wall_s": "", "replay_check": ""})
        _mon.__exit__(None, None, None)
        gpu_summary = _mon.summary()
        _log(f"GPU utilization: {gpu_summary}")
        if (_work / "gpu_util.csv").exists():
            _uploads["phase1/gpu_util.csv"] = _work / "gpu_util.csv"
        if _uploads:
            _remote.commit(_uploads, message="phase 1 reports and plots")
            _log(f"uploaded {sorted(_uploads)}")
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])
    finally:
        _mon.__exit__(None, None, None)  # idempotent; stops the sampler on errors too
        gpu_summary = _mon.summary()

    _imgs = [mo.image(src=p.read_bytes()) for p in plot_paths if p.exists()]
    _det_ok = all(r.get("identical") and r.get("replay_level") in ("L0", "L1") for r in det_rows)
    _summary = [
        mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
              + (f" — **errors:**\n```\n{errors[-1]}\n```" if errors else "")),
        mo.md(f"### G1-det: {'PASSED' if det_rows and _det_ok else 'see table'}"),
        mo.ui.table([{k: v for k, v in r.items() if k not in ("note",)} for r in det_rows], selection=None),
        mo.md("### GPU utilization (sampled every 30 s while GPU work ran)\n```\n"
              + json.dumps(gpu_summary, indent=2) + "\n```"),
        mo.md("### G1 reproductions"),
        mo.ui.table(repro_rows, selection=None),
        *_imgs,
    ]
    shutil.rmtree(_work / "repro", ignore_errors=True)
    mo.vstack(_summary)
    return det_rows, errors, gpu_summary, plot_paths, repro_rows


if __name__ == "__main__":
    app.run()

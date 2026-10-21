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
"""Phase 1b: the clean-up run that closes every open item from Phase 1.

A  CPU determinism: the same 16-config check twice. First "unpinned" (Phase 1's CPU
   environment), which should reproduce the Phase 1 drift. Then "pinned" (MKL/OpenMP
   dynamic threading off, MKL strict reproducibility, container-aware thread counts),
   which should be clean. 2k steps, keyframes every 1k.
B  CUDA determinism: all 16 configs again at 20k steps. MNIST SGD now uses init scale 1,
   so SGD determinism on MNIST is actually exercised.
C  SGD learning-rate scan on MNIST at init scale 8 (16 short runs).
D  Reproductions: teacher-student with the corrected accuracy, MNIST depth 3 with larger
   weight decay and a 3e5-step horizon, Deep Grokking at 12 layers. Plotted and uploaded.
E  Throughput: the same 32 short runs at 8 / 16 / 32 workers, and under MPS if Molab
   allows it. Checks that results stay bit-identical across all settings.

Results go to the HF dataset repo (default advay333/tfs-phase1) under phase1b/ and runs/.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    from pathlib import Path

    import marimo as mo

    mo.md(
        """
        # Phase 1b: clean-up run

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token, choose sections, and press **Run phase 1b**. With everything
           ticked, expect roughly 2–3 hours.
        3. Copy every table at the bottom back into the chat. Plots and JSON reports go to HF.
        """
    )
    return Path, json, mo, os, shutil, subprocess, sys, time, traceback


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
    code_ready = (_proj / "tfs" / "cpu_env.py").exists() and (_proj / "configs" / "sweeps" / "phase1b_repro.yaml").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Phase 1b code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs-phase1", label="HF dataset repo", full_width=True)
    workers = mo.ui.number(start=1, stop=32, value=16, label="parallel workers (GPU sections)")
    do_cpu = mo.ui.checkbox(value=True, label="A. CPU determinism, unpinned vs pinned (16 configs each, 2k steps)")
    do_cuda = mo.ui.checkbox(value=True, label="B. CUDA determinism (16 configs, 20k steps)")
    do_lr = mo.ui.checkbox(value=True, label="C. SGD learning-rate scan, MNIST alpha=8 (16 runs, 5k steps)")
    do_repro = mo.ui.checkbox(value=True, label="D. Reproductions (11 runs, up to 3e5 steps)")
    do_tp = mo.ui.checkbox(value=True, label="E. Throughput: 8 / 16 / 32 workers, and MPS")
    run_button = mo.ui.run_button(label="Run phase 1b")
    mo.vstack([hf_token, hf_repo, workers, do_cpu, do_cuda, do_lr, do_repro, do_tp, run_button])
    return do_cpu, do_cuda, do_lr, do_repro, do_tp, hf_repo, hf_token, run_button, workers


@app.cell
def _(Path, code_ready, do_cpu, do_cuda, do_lr, do_repro, do_tp, hf_repo, hf_token, json, mo, os,
      project_dir, run_button, shutil, sys, time, traceback, workers):
    mo.stop(not run_button.value, mo.md("_Press **Run phase 1b** when ready._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))

    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    os.chdir(_proj)
    from tfs import cpu_env as _cpu_env
    from tfs.training import determinism as _det

    _det.apply()
    import pandas as _pd
    import torch as _torch
    from scripts.det_check import det_check as _det_check
    from scripts.throughput import throughput as _throughput
    from tfs.evaluation.plots import plot_runs as _plot_runs
    from tfs.storage.hub import HfRemote as _HfRemote, push_runs as _push
    from tfs.storage.reader import RunReader as _RunReader
    from tfs.sweep.gpumon import GpuMonitor as _GpuMonitor
    from tfs.sweep.grid import expand as _expand
    from tfs.sweep.launcher import replay_parallel as _replay_parallel, run_parallel as _run_parallel
    from tfs.tasks.mnist import ensure_mnist as _ensure_mnist

    assert _torch.cuda.is_available(), "phase 1b needs the GPU"
    _work = Path("/tmp/tfs_phase1b")
    shutil.rmtree(_work, ignore_errors=True)
    _work.mkdir(parents=True)
    _ensure_mnist()
    _remote = _HfRemote(hf_repo.value, token=hf_token.value, create=True, private=False)
    _n = int(workers.value)
    tables = {}
    errors, plot_paths = [], []
    _uploads = {}
    _sweeps = _proj / "configs" / "sweeps"

    def _log(msg):
        print(msg, flush=True)

    def _save(name, rows):
        tables[name] = rows
        _p = _work / f"{name}.json"
        _p.write_text(json.dumps(rows, indent=2, default=str))
        _uploads[f"phase1b/{name}.json"] = _p

    def _brief(rows):
        keep = ("app", "optim", "device", "status_a", "status_b", "identical", "differing_files",
                "replay_level", "first_failure", "log_steps_checked", "wall_s")
        return [{k: r.get(k) for k in keep} for r in rows]

    _info = {"available_cpus": _cpu_env.available_cpus(), "os.cpu_count": os.cpu_count(),
             "cpu_model": _cpu_env.cpu_model(), "gpu": _torch.cuda.get_device_name(0)}
    _log(f"machine: {_info}")
    _t0 = time.time()
    _mon = _GpuMonitor(_work / "gpu_util.csv", every_s=30).__enter__()
    try:
        if do_cpu.value:
            _cpu_workers = min(_n, 8)
            for _label, _flag in (("cpu_unpinned", "1"), ("cpu_pinned", None)):
                _log(f"== A. CPU determinism: {_label} ==")
                if _flag:
                    os.environ["TFS_CPU_UNPINNED"] = _flag
                try:
                    _rows = _det_check(_sweeps / "phase1_det.yaml", "cpu", _cpu_workers, _work / _label, _log,
                                       steps=2000, keyframe_every=1000)
                finally:
                    os.environ.pop("TFS_CPU_UNPINNED", None)
                _save(f"det_{_label}", _rows)
                shutil.rmtree(_work / _label, ignore_errors=True)

        if do_cuda.value:
            _log("== B. CUDA determinism ==")
            _rows = _det_check(_sweeps / "phase1_det.yaml", "cuda", _n, _work / "det_cuda", _log)
            _save("det_cuda", _rows)
            shutil.rmtree(_work / "det_cuda", ignore_errors=True)

        if do_lr.value:
            _log("== C. SGD learning-rate scan ==")
            _cfgs = _expand(_sweeps / "phase1b_lr_scan.yaml")
            _res = _run_parallel(_cfgs, _work / "lr_scan", "cuda", _n, _log)
            _rows = []
            for _c, _r in zip(_cfgs, _res):
                _row = {"depth": _c.model.depth, "optim": _c.optim.name, "lr": _c.optim.lr, "status": _r.status}
                if _r.status in ("complete", "diverged"):
                    _tel = _RunReader(_r.run_dir).telemetry()
                    _row.update(train_loss_0=round(float(_tel.train_loss.iloc[0]), 4),
                                train_loss_end=round(float(_tel.train_loss.iloc[-1]), 4),
                                train_acc_end=round(float(_tel.train_acc.iloc[-1]), 3),
                                last_step=int(_tel.step.iloc[-1]))
                _rows.append(_row)
            _save("lr_scan", _rows)
            shutil.rmtree(_work / "lr_scan", ignore_errors=True)

        if do_repro.value:
            _log("== D. Reproductions ==")
            _cfgs = _expand(_sweeps / "phase1b_repro.yaml")
            _res = _run_parallel(_cfgs, _work / "repro", "cuda", _n, _log)
            _ok = [r for r in _res if r.status in ("complete", "diverged")]
            _replay_parallel([r.run_dir for r in _ok], "cuda", _n, full=False, log=_log)
            _rows = []
            for _r in _res:
                _row = {"run_id": _r.run_id, "status": _r.status, "wall_s": round(_r.wall_s)}
                if _r.status in ("complete", "diverged"):
                    _rep = json.loads((_r.run_dir / "replay_report.json").read_text()) \
                        if (_r.run_dir / "replay_report.json").exists() else {}
                    _orc = _pd.read_parquet(_r.run_dir / "oracle.parquet")
                    _tel = _RunReader(_r.run_dir).telemetry()
                    _mx = float(_orc.test_acc.max())
                    _row.update(replay_check=_rep.get("level", "n/a"),
                                train99_step=int(_tel[_tel.train_acc >= 0.99].step.min()) if (_tel.train_acc >= 0.99).any() else None,
                                max_test=round(_mx, 3), max_test_step=int(_orc.step[_orc.test_acc.idxmax()]),
                                t_eps=int(_orc[_orc.test_acc >= _mx - 0.01].step.min()),
                                final_test=round(float(_orc.test_acc.iloc[-1]), 3),
                                size_mb=round(sum(f.stat().st_size for f in _r.run_dir.iterdir()) / 1e6, 1))
                _rows.append(_row)
            _save("repro", _rows)
            for _app in sorted({c.app for c in _cfgs}):
                _dirs = [r.run_dir for r in _ok if r.run_id.startswith(_app + "__")]
                if _dirs:
                    _p = _plot_runs(_dirs, _work / "plots" / f"repro1b_{_app}.png")
                    plot_paths.append(_p)
                    _uploads[f"phase1b/plots/{_p.name}"] = _p
            _push_res = _push(_remote, [r.run_dir for r in _ok], delete_local=True, log=_log)
            _log(f"push: {({k: len(v) for k, v in _push_res.items()})}")
            shutil.rmtree(_work / "repro", ignore_errors=True)

        if do_tp.value:
            _log("== E. Throughput ==")
            _rows = _throughput([8, 16, 32], _work / "tp", mps_workers=16, log=_log)
            _save("throughput", _rows)
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])
    finally:
        _mon.__exit__(None, None, None)
        gpu_summary = _mon.summary()
        _save("machine", [{**_info, "gpu_summary": gpu_summary, "minutes": round((time.time() - _t0) / 60)}])
        if (_work / "gpu_util.csv").exists():
            _uploads["phase1b/gpu_util.csv"] = _work / "gpu_util.csv"
        try:
            _remote.commit(_uploads, message="phase 1b reports and plots")
            _log(f"uploaded {sorted(_uploads)}")
        except Exception:
            errors.append(traceback.format_exc())

    _view = []
    _view.append(mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                       + (f" — **errors:**\n```\n{errors[-1]}\n```" if errors else "")))
    for _name, _rows in tables.items():
        _view.append(mo.md(f"### {_name}"))
        _view.append(mo.ui.table(_brief(_rows) if _name.startswith("det_") else _rows, selection=None))
    _view += [mo.image(src=p.read_bytes()) for p in plot_paths if p.exists()]
    mo.vstack(_view)
    return errors, gpu_summary, plot_paths, tables


if __name__ == "__main__":
    app.run()

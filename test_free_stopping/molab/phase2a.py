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
"""Phase 2a: the first half of the pilot (PLAN.md §6, Phase 2).

0  HF clean-up: delete the six non-teacher-student Phase 1 runs (~6 GB) from
   advay333/tfs-phase1, then squash that repo's history so the space is actually freed.
1  Environment lock: record the exact Molab stack (torch, CUDA, cuDNN, GPU, driver, CPU,
   every package) for configs/env_lock.txt.
4  Throughput at 4 / 8 / 12 / 16 workers (and MPS if ticked), with a bit-identity check.
   The best worker count is used for the scan below.
2  Learning-rate scan: 516 short runs (5k steps) over every (application variant,
   optimizer); the SGD family at all three alpha levels. Runs are summarized and deleted
   as they finish; only the tables, curves and plots are uploaded.

Phase 2b (gamma placement, determinism re-check, corpus projection) needs the learning
rates chosen here, confirmed by a human first.

Reports go to the HF dataset repo advay333/tfs under phase2/2a/.
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
        # Phase 2a: clean-up, environment lock, throughput, learning-rate scan

        1. Check the repo location and press **Clone / pull**.
        2. Check the list of Phase 1 runs to delete (shown after the pull).
        3. Enter the HF token, choose sections, and press **Run phase 2a**. With everything
           ticked, expect roughly 1–2 hours.
        4. Tell Claude "done". Every table is printed to the console and uploaded to
           `advay333/tfs` under `phase2/2a/` (with a `summary.md`), so nothing needs copying.
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
    code_ready = (_proj / "scripts" / "env_lock.py").exists() and (_proj / "configs" / "sweeps" / "phase2_lr_scan.yaml").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Phase 2 code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(Path, code_ready, mo, project_dir, sys):
    # Which Phase 1 runs block 0 deletes: those in phase1_repro.yaml that are neither
    # teacher-student nor re-run by Phase 1b (the six oversized Phase 1 runs).
    mo.stop(not code_ready, mo.md("_Pull the code first._"))
    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from tfs.storage.hub import HfRemote as _HfRemote0, read_manifest as _read_manifest0
    from tfs.sweep.grid import expand as _expand0

    _keep = {c.run_id for c in _expand0(_proj / "configs" / "sweeps" / "phase1b_repro.yaml")}
    cleanup_ids = [c.run_id for c in _expand0(_proj / "configs" / "sweeps" / "phase1_repro.yaml")
                   if c.app != "teacher_student" and c.run_id not in _keep]
    _man = _read_manifest0(_HfRemote0("advay333/tfs-phase1", token=None, create=False))
    _rows = [{"run_id": r, "on_hf": r in _man, "size_gb": round(_man[r]["size_bytes"] / 1e9, 2) if r in _man else None}
             for r in cleanup_ids]
    print("Block 0 would delete:\n" + "\n".join(f"  {r}" for r in cleanup_ids), flush=True)
    mo.vstack([mo.md(f"### Block 0 deletes these {len(cleanup_ids)} runs from `advay333/tfs-phase1`"),
               mo.ui.table(_rows, selection=None)])
    return (cleanup_ids,)


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo (reports go under phase2/2a/)", full_width=True)
    workers = mo.ui.number(start=1, stop=32, value=12, label="workers for the lr scan if throughput is not run")
    do_cleanup = mo.ui.checkbox(value=False, label="0. Delete the runs listed above from advay333/tfs-phase1 and squash its history (irreversible)")
    do_env = mo.ui.checkbox(value=True, label="1. Environment lock")
    do_tp = mo.ui.checkbox(value=True, label="4. Throughput: 4 / 8 / 12 / 16 workers")
    do_mps = mo.ui.checkbox(value=False, label="   ...and once more under MPS at the best worker count")
    do_lr = mo.ui.checkbox(value=True, label="2. Learning-rate scan (516 runs, 5k steps)")
    run_button = mo.ui.run_button(label="Run phase 2a")
    mo.vstack([hf_token, hf_repo, workers, do_cleanup, do_env, do_tp, do_mps, do_lr, run_button])
    return do_cleanup, do_env, do_lr, do_mps, do_tp, hf_repo, hf_token, run_button, workers


@app.cell
def _(Path, cleanup_ids, code_ready, do_cleanup, do_env, do_lr, do_mps, do_tp, hf_repo, hf_token, json, mo,
      os, project_dir, run_button, shutil, sys, time, traceback, workers):
    mo.stop(not run_button.value, mo.md("_Press **Run phase 2a** when ready._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))

    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    os.chdir(_proj)
    from tfs import cpu_env as _cpu_env
    from tfs.training import determinism as _det

    _det.apply()
    import torch as _torch
    from scripts.env_lock import lock_text as _lock_text
    from scripts.throughput import throughput as _throughput
    from tfs.evaluation.plots import plot_lr_scan as _plot_lr_scan
    from tfs.evaluation.report import rows_to_markdown as _md, summary_markdown as _summary_md
    from tfs.evaluation.run_summary import run_curves as _run_curves
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote, delete_runs as _delete_runs
    from tfs.storage.hub import read_manifest as _read_manifest
    from tfs.sweep.gpumon import GpuMonitor as _GpuMonitor
    from tfs.sweep.grid import expand as _expand
    from tfs.sweep.launcher import run_parallel as _run_parallel
    from tfs.sweep.lr_select import scan_row as _scan_row, select_lrs as _select_lrs, variant_label as _vl
    from tfs.tasks.mnist import ensure_mnist as _ensure_mnist

    assert _torch.cuda.is_available(), "phase 2a needs the GPU"
    _work = Path("/tmp/tfs_phase2a")
    shutil.rmtree(_work, ignore_errors=True)
    _work.mkdir(parents=True)
    _ensure_mnist()
    _remote = _PrefixRemote(_HfRemote(hf_repo.value, token=hf_token.value, create=True, private=False), "phase2")
    tables = {}
    errors, plot_paths = [], []
    _uploads = {}
    _n_scan = int(workers.value)

    def _log(msg):
        print(msg, flush=True)

    def _save(name, rows):
        tables[name] = rows
        print("\n## " + name + "\n" + _md(rows), flush=True)  # readable even if the UI is not
        _p = _work / f"{name}.json"
        _p.write_text(json.dumps(rows, indent=2, default=str))
        _uploads[f"2a/{name}.json"] = _p

    _info = {"available_cpus": _cpu_env.available_cpus(), "os.cpu_count": os.cpu_count(),
             "cpu_model": _cpu_env.cpu_model(), "gpu": _torch.cuda.get_device_name(0)}
    _log(f"machine: {_info}")
    _t0 = time.time()
    _mon = _GpuMonitor(_work / "gpu_util.csv", every_s=30).__enter__()
    try:
        if do_cleanup.value:
            _log("== 0. HF clean-up of advay333/tfs-phase1 ==")
            _old = _HfRemote("advay333/tfs-phase1", token=hf_token.value, create=False)
            _man = _read_manifest(_old)
            _present = [r for r in cleanup_ids if r in _man]
            _freed = sum(_man[r]["size_bytes"] for r in _present)
            if _present:
                _delete_runs(_old, _present, log=_log)
            _old.squash_history("squash after deleting oversized Phase 1 runs (Phase 2a block 0)")
            _save("cleanup", [{"deleted": len(_present), "already_gone": len(cleanup_ids) - len(_present),
                               "freed_gb": round(_freed / 1e9, 2), "squashed": True}])

        if do_env.value:
            _log("== 1. Environment lock ==")
            _lock = _lock_text("cuda")
            _p = _work / "env_lock.txt"
            _p.write_text(_lock)
            _uploads["2a/env_lock.txt"] = _p
            print(_lock, flush=True)

        if do_tp.value:
            _log("== 4. Throughput ==")
            _rows = _throughput([4, 8, 12, 16], _work / "tp", log=_log)
            _ok = [r for r in _rows if r.get("identical_to_first") and r.get("runs_ok") == r.get("runs")]
            if _ok:
                _n_scan = max(_ok, key=lambda r: r["agg_steps_per_s"])["workers"]
            if do_mps.value:
                _mps_rows = _throughput([_n_scan], _work / "tp_mps", mps_workers=_n_scan, log=_log)
                _rows += [dict(r, note="MPS re-run") for r in _mps_rows[1:]]
            _save("throughput", _rows)
            _log(f"lr scan will use {_n_scan} workers")

        if do_lr.value:
            _log("== 2. Learning-rate scan ==")
            _cfgs = _expand(_proj / "configs" / "sweeps" / "phase2_lr_scan.yaml")
            _scan, _curves = [], {}

            def _on_result(res):
                # Summarize, then delete: the scan runs themselves are not kept.
                if res.status in ("complete", "diverged"):
                    try:
                        _scan.append({**_scan_row(res.run_dir), "wall_s": round(res.wall_s, 1)})
                        _curves[res.run_id] = _run_curves(res.run_dir, n_points=100)
                    except Exception:
                        errors.append(f"{res.run_id}: {traceback.format_exc()}")
                else:
                    _scan.append({"run_id": res.run_id, "status": res.status, "stable": False,
                                  "reason": "process failed", "error": res.log[-300:]})
                shutil.rmtree(res.run_dir, ignore_errors=True)

            _res = _run_parallel(_cfgs, _work / "lr_scan", "cuda", _n_scan, _log, on_result=_on_result)
            _by_id = {c.run_id: c for c in _cfgs}
            for _r in _scan:  # failed processes have no telemetry: fill their axes from the config
                if "variant" not in _r and _r["run_id"] in _by_id:
                    _c = _by_id[_r["run_id"]]
                    _r.update(variant=_vl(_c), optim=_c.optim.name, alpha=_c.model.init_scale, lr=_c.optim.lr)
            _scan.sort(key=lambda r: (r.get("variant", ""), r.get("optim", ""), r.get("alpha", 0), -r.get("lr", 0)))
            _save("lr_scan", _scan)
            _save("lr_choice", _select_lrs([r for r in _scan if "variant" in r]))
            _cp = _work / "lr_curves.json"
            _cp.write_text(json.dumps(_curves))
            _uploads["2a/lr_curves.json"] = _cp
            for _p in _plot_lr_scan([r for r in _scan if "variant" in r], _curves, _work / "plots"):
                plot_paths.append(_p)
                _uploads[f"2a/plots/{_p.name}"] = _p
            _walls = [r["wall_s"] for r in _scan if "wall_s" in r]
            _save("lr_scan_timing", [{"runs": len(_res), "workers": _n_scan,
                                      "mean_run_s": round(sum(_walls) / max(len(_walls), 1), 1),
                                      "minutes": round((time.time() - _t0) / 60)}])
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])
    finally:
        _mon.__exit__(None, None, None)
        gpu_summary = _mon.summary()
        _save("machine", [{**_info, "gpu_summary": gpu_summary, "minutes": round((time.time() - _t0) / 60)}])
        if (_work / "gpu_util.csv").exists():
            _uploads["2a/gpu_util.csv"] = _work / "gpu_util.csv"
        (_work / "summary.md").write_text(_summary_md("Phase 2a", tables, errors[-3:]))
        _uploads["2a/summary.md"] = _work / "summary.md"
        try:
            _remote.commit(_uploads, message="phase 2a reports and plots")
            _log(f"uploaded {sorted(_uploads)}")
        except Exception:
            errors.append(traceback.format_exc())
            _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **errors ({len(errors)}):**\n```\n{errors[-1]}\n```" if errors else ""))]
    for _name, _rows in tables.items():
        if _name != "lr_scan":
            _view += [mo.md(f"### {_name}"), mo.ui.table(_rows, selection=None)]
    _view += [mo.image(src=p.read_bytes()) for p in plot_paths if p.exists()]
    mo.vstack(_view)
    return errors, gpu_summary, plot_paths, tables


if __name__ == "__main__":
    app.run()

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
"""Phase 2b: the second half of the pilot (PLAN.md §6, Phase 2), with the learning rates
confirmed after Phase 2a.

E  Environment check against configs/env_lock.txt (the run stops on a mismatch unless
   overridden).
A  Throughput under MPS at 16 / 24 / 32 workers. The best count is used below.
B  Determinism re-check on the final configs (configs/sweeps/phase2_det.yaml: 24
   configs, 20k steps), trained twice and fully replayed, under MPS like the corpus.
C  Gamma placement (configs/sweeps/phase2_gamma.yaml: 146 runs, 1e5 steps; two
   teacher-student runs at 3e5). Each run is replay-checked, summarized and uploaded to
   advay333/tfs under phase2/runs/ as it finishes. Re-running resumes: runs already on
   HF are skipped, and the tables are merged with the ones already uploaded.
D  Gate G2 tables and the Phase 3/4 cost projection.

Reports go to advay333/tfs under phase2/2b/.
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
        # Phase 2b: determinism, γ placement, G2 and the corpus projection

        1. Check the repo location and press **Clone / pull**.
        2. Check the environment comparison shown after the pull.
        3. Enter the HF token, choose sections, and press **Run phase 2b**. With everything
           ticked, expect roughly 2–4 hours.
        4. Tell Claude "done". Every table is printed to the console and uploaded to
           `advay333/tfs` under `phase2/2b/` (with a `summary.md`).
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
    code_ready = (_proj / "configs" / "sweeps" / "phase2_gamma.yaml").exists() and (_proj / "tfs" / "evaluation" / "pilot.py").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Phase 2b code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(Path, code_ready, mo, project_dir, subprocess, sys):
    # Compare the live environment with the lock written in Phase 2a. Runs in a fresh
    # process so that CUDA is initialized there, not in the notebook kernel.
    mo.stop(not code_ready, mo.md("_Pull the code first._"))
    _proj = Path(project_dir.value)
    _lock = _proj / "configs" / "env_lock.txt"
    _r = subprocess.run([sys.executable, "-m", "scripts.env_lock", "--check", str(_lock)],
                        cwd=_proj, capture_output=True, text=True)
    env_matches = _r.returncode == 0
    env_report = (_r.stdout + _r.stderr).strip()
    print("env check:", env_report, flush=True)
    mo.md(f"### Environment vs `configs/env_lock.txt`\n```\n{env_report}\n```\n"
          + ("" if env_matches else "**Mismatch.** Runs trained now could not be replayed in the locked "
             "environment. Tick the override below only if you accept that."))
    return env_matches, env_report


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo (phase2/runs/, phase2/2b/)", full_width=True)
    workers = mo.ui.number(start=1, stop=48, value=16, label="workers if throughput is not run")
    use_mps = mo.ui.checkbox(value=True, label="Run everything under MPS (Phase 2a: 1.85x, bit-identical)")
    allow_env = mo.ui.checkbox(value=False, label="Override: run even if the environment does not match the lock")
    do_tp = mo.ui.checkbox(value=True, label="A. Throughput under MPS: 16 / 24 / 32 workers")
    do_det = mo.ui.checkbox(value=True, label="B. Determinism re-check (24 configs x 2, 20k steps, full replay)")
    do_gamma = mo.ui.checkbox(value=True, label="C. Gamma placement (146 runs, uploaded as they finish)")
    run_button = mo.ui.run_button(label="Run phase 2b")
    mo.vstack([hf_token, hf_repo, workers, use_mps, allow_env, do_tp, do_det, do_gamma, run_button])
    return allow_env, do_det, do_gamma, do_tp, hf_repo, hf_token, run_button, use_mps, workers


@app.cell
def _(Path, allow_env, code_ready, do_det, do_gamma, do_tp, env_matches, env_report, hf_repo, hf_token, json,
      mo, os, project_dir, run_button, shutil, sys, time, traceback, use_mps, workers):
    mo.stop(not run_button.value, mo.md("_Press **Run phase 2b** when ready._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))
    mo.stop(not (env_matches or allow_env.value), mo.md("**Environment mismatch; see above.**"))

    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    os.chdir(_proj)
    from tfs import cpu_env as _cpu_env
    from tfs.training import determinism as _det

    _det.apply()
    import contextlib as _contextlib

    import torch as _torch
    from scripts.det_check import det_check as _det_check
    from scripts.sweep import sweep as _sweep
    from scripts.throughput import Mps as _Mps, throughput as _throughput
    from tfs.evaluation.pilot import g2_checks as _g2, project as _project
    from tfs.evaluation.plots import plot_gamma_sweep as _plot_gamma
    from tfs.evaluation.report import rows_to_markdown as _md, summary_markdown as _summary_md
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote
    from tfs.sweep.gpumon import GpuMonitor as _GpuMonitor
    from tfs.sweep.grid import expand as _expand
    from tfs.tasks.mnist import ensure_mnist as _ensure_mnist

    assert _torch.cuda.is_available(), "phase 2b needs the GPU"
    _work = Path("/tmp/tfs_phase2b")
    shutil.rmtree(_work, ignore_errors=True)
    _work.mkdir(parents=True)
    _ensure_mnist()
    _remote = _PrefixRemote(_HfRemote(hf_repo.value, token=hf_token.value, create=True, private=False), "phase2")
    _sweeps = _proj / "configs" / "sweeps"
    tables = {}
    errors, plot_paths = [], []
    _uploads = {}
    _n = int(workers.value)

    def _log(msg):
        print(msg, flush=True)

    def _save(name, rows, show=True):
        tables[name] = rows
        if show:
            print("\n## " + name + "\n" + _md(rows), flush=True)  # readable even if the UI is not
        _p = _work / f"{name}.json"
        _p.write_text(json.dumps(rows, indent=2, default=str))
        _uploads[f"2b/{name}.json"] = _p

    _info = {"available_cpus": _cpu_env.available_cpus(), "cpu_model": _cpu_env.cpu_model(),
             "gpu": _torch.cuda.get_device_name(0), "env_check": env_report, "mps": use_mps.value}
    _log(f"machine: {_info}")
    _t0 = time.time()
    _mon = _GpuMonitor(_work / "gpu_util.csv", every_s=30).__enter__()
    # Previously uploaded gamma tables (a resumed session merges into them).
    _prev_rows = _remote.read_json("2b/gamma_runs.json") or []
    _prev_curves = _remote.read_json("2b/gamma_curves.json") or {}
    try:
        if do_tp.value:
            _log("== A. Throughput under MPS ==")
            _rows = _throughput([16, 24, 32], _work / "tp", log=_log, mps=True)
            _ok = [r for r in _rows if r.get("identical_to_first") and r.get("runs_ok") == r.get("runs")]
            if _ok:
                _n = max(_ok, key=lambda r: r["agg_steps_per_s"])["workers"]
            _save("throughput_mps", _rows)
            _log(f"blocks B and C use {_n} workers")

        with (_Mps() if use_mps.value else _contextlib.nullcontext()) as _daemon:
            if use_mps.value and not _daemon.started:
                _log(f"MPS could not start ({_daemon.error}); continuing with time-slicing")
            _mode = "MPS" if (use_mps.value and _daemon.started) else "time-slicing"

            if do_det.value:
                _log(f"== B. Determinism re-check ({_mode}) ==")
                _rows = _det_check(_sweeps / "phase2_det.yaml", "cuda", _n, _work / "det", _log)
                _keep = ("app", "optim", "status_a", "status_b", "identical", "differing_files", "replay_level",
                         "max_err", "first_failure", "wall_s")
                _save("det", [{k: r.get(k) for k in _keep} for r in _rows])
                shutil.rmtree(_work / "det", ignore_errors=True)

            if do_gamma.value:
                _log(f"== C. Gamma placement ({_mode}) ==")
                # Longest runs first, so the tail of the sweep is not one slow run.
                _cfgs = sorted(_expand(_sweeps / "phase2_gamma.yaml"),
                               key=lambda c: (-c.steps, -c.model.depth, -c.model.width))
                _new_rows, _new_curves = [], {}
                _last = [time.time()]

                def _checkpoint():
                    _rows_all = {r["run_id"]: r for r in _prev_rows + _new_rows}
                    _save("gamma_runs", list(_rows_all.values()), show=False)
                    _cp = _work / "gamma_curves.json"
                    _cp.write_text(json.dumps({**_prev_curves, **_new_curves}))
                    _uploads["2b/gamma_curves.json"] = _cp
                    _remote.commit({k: v for k, v in _uploads.items() if k.startswith("2b/gamma_")},
                                   message="phase 2b gamma tables (checkpoint)")

                def _on_row(row, curves):
                    _new_rows.append(row)
                    if curves:
                        _new_curves[row["run_id"]] = curves
                    if time.time() - _last[0] > 1800:  # every 30 min, survive a session that ends early
                        try:
                            _checkpoint()
                            _last[0] = time.time()
                        except Exception:
                            _log(f"checkpoint failed: {traceback.format_exc()[-300:]}")

                _rows, _curves, _upinfo = _sweep(_cfgs, _work / "runs", "cuda", _n, _remote, batch_size=10,
                                                 replay_check=True, log=_log, on_row=_on_row)
                _save("gamma_upload", [{k: v for k, v in _upinfo.items() if k != "upload_failed"}
                                       | {"upload_failed": len(_upinfo["upload_failed"])}])
                if _upinfo["upload_failed"]:
                    errors.append(f"upload failed for {_upinfo['upload_failed']}")
                _checkpoint()

        # D. G2 and projection, over every gamma run uploaded so far (this and earlier sessions).
        _all_rows = tables.get("gamma_runs") or _prev_rows
        _all_curves = {**_prev_curves, **(_new_curves if do_gamma.value else {})}
        if _all_rows:
            _log("== D. G2 and projection ==")
            _cols = ("variant", "optim", "alpha", "lr", "gamma", "steps", "category", "train99_step", "t_eps",
                     "max_test", "final_test", "replay_level", "wall_s", "size_mb")
            _brief = sorted(([{k: r.get(k) for k in _cols} for r in _all_rows if "variant" in r]),
                            key=lambda r: (r["variant"], r["optim"], r["steps"], r["gamma"]))
            tables["gamma_brief"] = _brief
            print("\n## gamma_brief\n" + _md(_brief), flush=True)
            _save("g2_by_app", _g2(_all_rows))
            _save("g2_by_optimizer", _g2(_all_rows, by=("variant", "optim")))
            _save("projection", _project(_all_rows, _n))
            _save("projection_ts3e5", _project(_all_rows, _n, ts_horizon=300_000))
            for _p in _plot_gamma(_all_rows, _all_curves, _work / "plots"):
                plot_paths.append(_p)
                _uploads[f"2b/plots/{_p.name}"] = _p
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])
    finally:
        _mon.__exit__(None, None, None)
        gpu_summary = _mon.summary()
        _save("machine", [{**_info, "workers": _n, "gpu_summary": gpu_summary,
                           "minutes": round((time.time() - _t0) / 60)}])
        if (_work / "gpu_util.csv").exists():
            _uploads["2b/gpu_util.csv"] = _work / "gpu_util.csv"
        (_work / "summary.md").write_text(_summary_md("Phase 2b", {k: v for k, v in tables.items() if k != "gamma_runs"},
                                                      errors[-3:]))
        _uploads["2b/summary.md"] = _work / "summary.md"
        try:
            _remote.commit(_uploads, message="phase 2b reports and plots")
            _log(f"uploaded {sorted(_uploads)}")
        except Exception:
            errors.append(traceback.format_exc())
            _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **errors ({len(errors)}):**\n```\n{errors[-1]}\n```" if errors else ""))]
    for _name, _rows in tables.items():
        if _name != "gamma_runs":
            _view += [mo.md(f"### {_name}"), mo.ui.table(_rows, selection=None)]
    _view += [mo.image(src=p.read_bytes()) for p in plot_paths if p.exists()]
    mo.vstack(_view)
    return errors, gpu_summary, plot_paths, tables


if __name__ == "__main__":
    app.run()

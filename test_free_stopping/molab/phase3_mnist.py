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
"""Phase 3, MNIST: the corpus runs for one dataset (PLAN.md §6, Phase 3).

configs/sweeps/phase3_mnist.yaml: 256 runs (MNIST depth 3 and depth 8, 4 seeds), about
1.3 GPU-hours under MPS at 16 workers. The other datasets have their own sweep files
(phase3_modular.yaml, phase3_teacher_student.yaml); their notebooks are this file with
DATASET changed.

A  Train: calibration seeds {0, 1} first. Each run is replay-checked over one keyframe
   interval, summarized, and uploaded to advay333/tfs under phase3/mnist/runs/ as it
   finishes. Re-running resumes: runs already on HF are skipped.
B  G3 audit: re-download a random 5% of the runs, check every file, replay them end to
   end, and record the full-replay level in the manifest.
C  (optional) Delete runs whose replay failed, so that A retrains them.
D  Corpus summary: manifest status, run categories, grok-time spread, plots of seed 0.

Reports go to phase3/mnist/reports/.
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

    DATASET = "mnist"  # the only line that differs between the Phase 3 notebooks

    mo.md(
        f"""
        # Phase 3: corpus runs for **{DATASET}**

        1. Check the repo location and press **Clone / pull**.
        2. Check the environment comparison and the corpus status shown after the pull.
        3. Enter the HF token, choose sections, and press **Run**. Training all 256 runs takes
           about 1.5 hours; if the session ends, run it again and it resumes.
        4. Tell Claude "done". Tables are printed to the console and uploaded to
           `advay333/tfs` under `phase3/{DATASET}/reports/` (with a `summary.md`).
        """
    )
    return DATASET, Path, json, mo, os, shutil, subprocess, sys, time, traceback


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
def _(DATASET, Path, mo, project_dir, pull_button, repo_url, subprocess):
    mo.stop(not pull_button.value, mo.md("_Press **Clone / pull** to fetch the latest code._"))
    _proj = Path(project_dir.value)
    _root = _proj.parent
    if not (_root / ".git").exists():
        _out = subprocess.run(["git", "clone", repo_url.value, str(_root)], capture_output=True, text=True)
    else:
        _out = subprocess.run(["git", "-C", str(_root), "pull", "--ff-only"], capture_output=True, text=True)
    _head = subprocess.run(["git", "-C", str(_root), "log", "-1", "--oneline"], capture_output=True, text=True).stdout
    code_ready = (_proj / "configs" / "sweeps" / f"phase3_{DATASET}.yaml").exists() and (_proj / "tfs" / "sweep" / "corpus.py").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Phase 3 code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(DATASET, Path, code_ready, mo, project_dir, subprocess, sys):
    # Environment vs the lock (fresh process, so CUDA starts there), and how far the
    # corpus already is (the HF repo is public: no token needed to read it).
    mo.stop(not code_ready, mo.md("_Pull the code first._"))
    _proj = Path(project_dir.value)
    _r = subprocess.run([sys.executable, "-m", "scripts.env_lock", "--check", str(_proj / "configs" / "env_lock.txt")],
                        cwd=_proj, capture_output=True, text=True)
    env_matches = _r.returncode == 0
    env_report = (_r.stdout + _r.stderr).strip()
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from tfs.storage.hub import HfRemote as _HfRemote0, PrefixRemote as _PrefixRemote0
    from tfs.sweep.corpus import manifest_status as _status0, remote_prefix as _prefix0, sweep_file as _sweep_file0
    from tfs.sweep.grid import expand as _expand0

    _status = _status0(_PrefixRemote0(_HfRemote0("advay333/tfs", token=None, create=False), _prefix0(DATASET)),
                       _expand0(_sweep_file0(_proj, DATASET)))
    print("env check:", env_report, "\ncorpus status:", _status, flush=True)
    mo.vstack([
        mo.md(f"### Environment vs `configs/env_lock.txt`\n```\n{env_report}\n```\n"
              + ("" if env_matches else "**Mismatch.** Runs trained now could not be replayed in the locked "
                 "environment. Tick the override below only if you accept that.")),
        mo.md(f"### Corpus status ({DATASET})"),
        mo.ui.table([{k: v for k, v in _status.items() if k != "replay_failed"}], selection=None),
    ])
    return env_matches, env_report


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo", full_width=True)
    workers = mo.ui.number(start=1, stop=48, value=16, label="workers (Phase 2b: 16 is best under MPS)")
    use_mps = mo.ui.checkbox(value=True, label="Run under MPS")
    allow_env = mo.ui.checkbox(value=False, label="Override: run even if the environment does not match the lock")
    do_train = mo.ui.checkbox(value=True, label="A. Train (skips runs already on HF)")
    do_audit = mo.ui.checkbox(value=True, label="B. G3 audit: full replay of a random 5% of runs")
    do_retrain = mo.ui.checkbox(value=False, label="C. Delete runs whose replay failed, then retrain them")
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([hf_token, hf_repo, workers, use_mps, allow_env, do_train, do_audit, do_retrain, run_button])
    return allow_env, do_audit, do_retrain, do_train, hf_repo, hf_token, run_button, use_mps, workers


@app.cell
def _(DATASET, Path, allow_env, code_ready, do_audit, do_retrain, do_train, env_matches, env_report, hf_repo,
      hf_token, json, mo, os, project_dir, run_button, shutil, sys, time, traceback, use_mps, workers):
    mo.stop(not run_button.value, mo.md("_Press **Run** when ready._"))
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
    from scripts.sweep import sweep as _sweep
    from scripts.throughput import Mps as _Mps
    from tfs.evaluation.pilot import g2_checks as _g2
    from tfs.evaluation.plots import plot_gamma_sweep as _plot_gamma
    from tfs.evaluation.report import rows_to_markdown as _md, summary_markdown as _summary_md
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote
    from tfs.sweep import corpus as _corpus
    from tfs.sweep.gpumon import GpuMonitor as _GpuMonitor
    from tfs.sweep.grid import expand as _expand
    from tfs.tasks.mnist import ensure_mnist as _ensure_mnist

    assert _torch.cuda.is_available(), "phase 3 needs the GPU"
    _work = Path(f"/tmp/tfs_phase3_{DATASET}")
    shutil.rmtree(_work, ignore_errors=True)
    _work.mkdir(parents=True)
    if DATASET == "mnist":
        _ensure_mnist()  # once, before parallel workers would race to download it
    _remote = _PrefixRemote(_HfRemote(hf_repo.value, token=hf_token.value, create=True, private=False),
                            _corpus.remote_prefix(DATASET))
    _cfgs = _corpus.corpus_order(_expand(_corpus.sweep_file(_proj, DATASET)))
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
        _uploads[f"reports/{name}.json"] = _p

    _info = {"dataset": DATASET, "runs_in_grid": len(_cfgs), "gpu": _torch.cuda.get_device_name(0),
             "available_cpus": _cpu_env.available_cpus(), "env_check": env_report, "mps": use_mps.value,
             "workers": _n}
    _log(f"machine: {_info}")
    _t0 = time.time()
    _mon = _GpuMonitor(_work / "gpu_util.csv", every_s=60).__enter__()
    # Summary rows and curves of every run uploaded so far (earlier sessions included).
    _prev_rows = _remote.read_json("reports/corpus_runs.json") or []
    _prev_curves = _remote.read_json("reports/corpus_curves.json") or {}
    _new_rows, _new_curves = [], {}

    def _checkpoint():
        _save("corpus_runs", list({r["run_id"]: r for r in _prev_rows + _new_rows}.values()), show=False)
        _cp = _work / "corpus_curves.json"
        _cp.write_text(json.dumps({**_prev_curves, **_new_curves}))
        _uploads["reports/corpus_curves.json"] = _cp
        _remote.commit({k: v for k, v in _uploads.items() if k.startswith("reports/corpus_")},
                       message=f"phase 3 {DATASET}: corpus tables (checkpoint)")

    _last = [time.time()]

    def _on_row(row, curves):
        _new_rows.append(row)
        if curves:
            _new_curves[row["run_id"]] = curves
        if time.time() - _last[0] > 1200:  # every 20 min: survive a session that ends early
            try:
                _checkpoint()
                _last[0] = time.time()
            except Exception:
                _log(f"checkpoint failed: {traceback.format_exc()[-300:]}")

    try:
        with (_Mps() if use_mps.value else _contextlib.nullcontext()) as _daemon:
            if use_mps.value and not _daemon.started:
                _log(f"MPS could not start ({_daemon.error}); continuing with time-slicing")

            if do_retrain.value:
                _log("== C. Delete runs whose replay failed ==")
                _failed = _corpus.manifest_status(_remote, _cfgs)["replay_failed"]
                _corpus.drop_failed(_remote, _failed, log=_log)
                _prev_rows = [r for r in _prev_rows if r.get("run_id") not in set(_failed)]
                _save("retrained", [{"run_id": r} for r in _failed])

            if do_train.value or do_retrain.value:
                _log(f"== A. Train {DATASET} ({len(_cfgs)} runs in the grid) ==")
                _rows, _curves, _upinfo = _sweep(_cfgs, _work / "runs", "cuda", _n, _remote, batch_size=10,
                                                 replay_check=True, log=_log, on_row=_on_row)
                _save("train_upload", [{**{k: v for k, v in _upinfo.items() if k != "upload_failed"},
                                        "upload_failed": len(_upinfo["upload_failed"]),
                                        "minutes": round((time.time() - _t0) / 60)}])
                if _upinfo["upload_failed"]:
                    errors.append(f"upload failed for {_upinfo['upload_failed']}")
                _checkpoint()

            if do_audit.value:
                _log("== B. G3 audit ==")
                _save("audit", _corpus.audit(_remote, _cfgs, _work / "audit", "cuda", _n, frac=0.05, log=_log))

        # D. Corpus summary, over everything uploaded so far.
        _status = _corpus.manifest_status(_remote, _cfgs)
        _save("status", [_status])
        _all = tables.get("corpus_runs") or _prev_rows
        if _all:
            _save("categories_by_variant", _g2(_all))
            _save("categories_by_optimizer", _g2(_all, by=("variant", "optim")))
            _save("categories_by_alpha", _g2(_all, by=("variant", "alpha")))
            _seed0 = [r for r in _all if r.get("seed") == 0]
            for _p in _plot_gamma(_seed0, {**_prev_curves, **_new_curves}, _work / "plots"):
                plot_paths.append(_p)
                _uploads[f"reports/plots/{_p.name}"] = _p
        _done = _status["missing"] == 0 and not _status["replay_failed"]
        _log(f"G3 ({DATASET}): {'all runs uploaded, none failed replay' if _done else 'NOT complete'}: {_status}")
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])
    finally:
        _mon.__exit__(None, None, None)
        gpu_summary = _mon.summary()
        _save("machine", [{**_info, "gpu_summary": gpu_summary, "minutes": round((time.time() - _t0) / 60)}])
        if (_work / "gpu_util.csv").exists():
            _uploads["reports/gpu_util.csv"] = _work / "gpu_util.csv"
        (_work / "summary.md").write_text(_summary_md(f"Phase 3: {DATASET}",
                                                      {k: v for k, v in tables.items() if k != "corpus_runs"},
                                                      errors[-3:]))
        _uploads["reports/summary.md"] = _work / "summary.md"
        try:
            _remote.commit(_uploads, message=f"phase 3 {DATASET}: reports")
            _log(f"uploaded {sorted(_uploads)}")
        except Exception:
            errors.append(traceback.format_exc())
            _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **errors ({len(errors)}):**\n```\n{errors[-1]}\n```" if errors else ""))]
    for _name, _rows in tables.items():
        if _name != "corpus_runs":
            _view += [mo.md(f"### {_name}"), mo.ui.table(_rows, selection=None)]
    _view += [mo.image(src=p.read_bytes()) for p in plot_paths if p.exists()]
    mo.vstack(_view)
    return errors, gpu_summary, plot_paths, tables


if __name__ == "__main__":
    app.run()

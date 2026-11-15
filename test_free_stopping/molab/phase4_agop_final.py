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
"""Phase 4 final F2 study (stopping/agop_final.py, report_AGOP/).

Modular and MNIST: whole hyperparameter cells (all 4 seeds) are replayed once each to
record the agop_path trace. Every F2 rule (remaining-angle bound, angle over a lag
proportional to t, angle over a constant lag, settling tests), a fixed-time stop and
the oracles that know the final AGOP are then scored offline. Settings are chosen on
calibration runs (seeds 0-1; also: x+y, or MNIST depth 3) and scored on the held-out
runs (seeds 2-3; x*y and x/y; depth 8).

Several replays run at once, each in its own process, so the GPU is shared the way it
was when the corpus was trained (one replay alone uses under 10% of it). Traces from
the earlier F2 experiment (phase4/agop_study) are reused where the run is the same.

Resumable: traces upload to advay333/tfs under phase4/agop_final/ as they finish.
Outputs there: rows.parquet, series.parquet, diag.csv, report_AGOP/ (REPORT_auto.md,
figures/, tables/), status.json.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import time
    import traceback
    from pathlib import Path

    import marimo as mo

    OUT_PREFIX = "phase4/agop_final"
    OLD_PREFIX = "phase4/agop_study"

    mo.md(
        f"""
        # Phase 4: final F2 study (calibrate, then test on held-out runs)

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token, check the options, press **Preview selection** and look at
           the run list and the time estimate.
        3. Press **Run**. Traces upload to `advay333/tfs` under `{OUT_PREFIX}/` as they finish,
           so an interrupted session resumes where it stopped (press Run again).
        4. Tell Claude "done". Everything Claude needs is in `{OUT_PREFIX}/`.
        """
    )
    return OLD_PREFIX, OUT_PREFIX, Path, json, mo, os, subprocess, sys, time, traceback


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
    code_ready = (_proj / "stopping" / "agop_final.py").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Final-study code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(mo, os, subprocess):
    # hardware, for the default worker counts
    _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    gpu_line = _gpu.stdout.strip().splitlines()[0] if _gpu.returncode == 0 and _gpu.stdout.strip() else ""
    gpu_mem_gb = float(gpu_line.split(",")[-1]) / 1024 if gpu_line else 0.0
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # One replay is one Python process feeding small kernels: it needs about one CPU core and
    # peaks at ~4 GB of GPU memory (MNIST depth 8: 1.3 GB weight history + 0.9 GB G/M
    # history + AGOP Jacobian + CUDA context; modular ~1.5 GB). Budget: 60% of GPU memory
    # at 4 GB per replay, one core left for this notebook, and at most 16 (beyond that the
    # GPU time-slices the processes with little gain). RTX Pro 6000 (96 GB): min(16, cores-1, 14).
    _default = max(1, min(16, n_cpu - 1, int(0.6 * gpu_mem_gb // 4))) if gpu_mem_gb else 1

    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo", full_width=True)
    mod_cells = mo.ui.text(value="modular x+y:6, modular x*y:2, modular x/y:2",
                           label="modular: cells per (variant, optimizer); each cell = 4 seeds", full_width=True)
    mnist_cells = mo.ui.text(value="mnist d3:2, mnist d8:2", label="MNIST: cells per (variant, optimizer)",
                             full_width=True)
    workers = mo.ui.number(start=1, stop=32, value=_default,
                           label="parallel replays (modular and MNIST share one pool, longest runs first)")
    verify = mo.ui.checkbox(value=True, label="verify every replay against the recorded run (telemetry, "
                                              "train predictions, keyframes; a diverging run fails)")
    an_workers = mo.ui.number(start=1, stop=64, value=max(1, n_cpu), label="parallel analysis processes (CPU)")
    reuse_old = mo.ui.checkbox(value=True, label="reuse traces from the earlier F2 experiment (phase4/agop_study)")
    analysis_only = mo.ui.checkbox(value=False, label="analysis only: no replay, use the traces already uploaded")
    allow_env = mo.ui.checkbox(value=False, label="Allow replay in an environment that differs from the run's")
    upload_every = mo.ui.number(start=1, stop=100, value=10, label="upload traces every N finished runs")
    preview_button = mo.ui.run_button(label="Preview selection")
    mo.vstack([mo.md(f"GPU: **{gpu_line or 'none found'}** ({gpu_mem_gb:.0f} GB); CPUs: **{n_cpu}**. Default "
                     f"parallel replays: **{_default}** (about 1 core and at most ~4 GB of GPU memory each)."),
               hf_token, hf_repo, mod_cells, mnist_cells, workers, verify, an_workers, reuse_old,
               analysis_only, allow_env, upload_every, preview_button])
    return (allow_env, an_workers, analysis_only, gpu_line, gpu_mem_gb, hf_repo, hf_token, mnist_cells, mod_cells,
            n_cpu, preview_button, reuse_old, upload_every, verify, workers)


@app.cell
def _(Path, an_workers, code_ready, hf_repo, hf_token, mnist_cells, mo, mod_cells, preview_button, project_dir,
      sys, verify, workers):
    mo.stop(not preview_button.value, mo.md("_Press **Preview selection**._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))
    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from stopping.agop_final import pick_final_runs as _pick
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote

    def _parse(text):
        out = {}
        for part in text.split(","):
            if ":" in part:
                k, v = part.rsplit(":", 1)
                out[k.strip()] = int(v)
        return out

    _hf = _HfRemote(hf_repo.value, token=hf_token.value, create=False)
    selection, _notes = [], []
    for _app, _spec in (("modular", _parse(mod_cells.value)), ("mnist", _parse(mnist_cells.value))):
        if not _spec:
            continue
        _rows = _PrefixRemote(_hf, f"phase3/{_app}").read_json("reports/corpus_runs.json") or []
        _sel = _pick(_rows, _app, _spec)
        _cats = {}
        for _r in _sel:
            if _r["seed"] == 0:
                _cats[_r["cell_category"]] = _cats.get(_r["cell_category"], 0) + 1
        _notes.append(f"- **{_app}**: {len(_rows)} corpus runs; {len(_sel)} selected = {len(_sel) // 4} cells x 4 "
                      f"seeds; cells per seed-0 category {_cats}")
        selection += _sel
    _n_mod = sum(r["app"] == "modular" for r in selection)
    _n_mn = len(selection) - _n_mod
    # measured in the earlier F2 experiment: one replay alone ~140 s (modular), ~200 s (MNIST);
    # sharing the GPU slows each replay down (factor 1.5 assumed), verification adds ~20%
    _h = (_n_mod * 140 + _n_mn * 200) / int(workers.value) * 1.5 * (1.2 if verify.value else 1.0) / 3600
    _ha = len(selection) * 40 / int(an_workers.value) / 3600
    _cols = ("app", "variant", "optim", "lr", "gamma", "alpha", "seed", "cell_category", "category", "max_test",
             "t_eps", "run_id")
    mo.vstack([
        mo.md("\n".join(_notes) + f"\n\n**{len(selection)} runs.** Rough time: replay about **{_h:.1f} h** "
              f"(less for runs already traced by the earlier experiment), analysis about **{_ha:.1f} h**. "
              "Press **Run** below when this looks right."),
        mo.ui.table([{k: r.get(k) for k in _cols} for r in selection], selection=None, page_size=20),
    ])
    return (selection,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run")
    run_button
    return (run_button,)


@app.cell
def _(OLD_PREFIX, OUT_PREFIX, Path, allow_env, an_workers, analysis_only, hf_repo, hf_token, json, mo, n_cpu, os,
      project_dir, reuse_old, run_button, selection, subprocess, time, traceback, upload_every, verify, workers):
    mo.stop(not run_button.value, mo.md("_Press **Run** when ready._"))
    mo.stop(not selection, mo.md("**Preview a non-empty selection first.**"))

    import pandas as _pd
    import torch as _torch
    from stopping import agop_final as _af
    from stopping.signals import AgopPathConfig as _Cfg
    from stopping.store import TraceStore as _TraceStore
    from tfs.storage import hub as _hub

    _proj = Path(project_dir.value)
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    _work = Path("/tmp/tfs_phase4_agop_final")
    _work.mkdir(parents=True, exist_ok=True)
    _hf = _hub.HfRemote(hf_repo.value, token=hf_token.value, create=False)
    _out = _hub.PrefixRemote(_hf, OUT_PREFIX)
    _cfg = _Cfg(weights=True)  # same config (hash) as the earlier experiment, so its traces are reused
    _store = _TraceStore(_work / "traces")
    errors, status = [], {}
    _t0 = time.time()

    def _gpu():
        _q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                            capture_output=True, text=True)
        return _q.stdout.strip().replace("\n", " | ") if _q.returncode == 0 else "n/a"

    def _log(msg):
        if msg.startswith("worker:"):
            msg += f"  [GPU util, memory: {_gpu()}]"
        print(msg, flush=True)

    def _env(workers):
        _e = dict(os.environ)
        _e["HF_TOKEN"] = hf_token.value  # workers read it from the environment; it is never written to disk
        _n = str(max(1, n_cpu // max(1, workers)))
        _e.update(OMP_NUM_THREADS=_n, MKL_NUM_THREADS=_n, TFS_NUM_THREADS=_n,
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")  # less fragmentation with many processes
        return _e

    try:
        _ids = {r["run_id"] for r in selection}
        _af.resume_from(_out, OUT_PREFIX, _store, _cfg, _work, _ids, log=_log)
        if reuse_old.value:
            _af.resume_from(_hub.PrefixRemote(_hf, OLD_PREFIX), OLD_PREFIX, _store, _cfg, _work, _ids, log=_log)
        if not analysis_only.value:
            if any(r["app"] == "mnist" for r in selection):
                from tfs.tasks.mnist import ensure_mnist as _ensure_mnist
                _ensure_mnist()  # once, before the workers start
            _w = int(workers.value)
            _log(f"--- replaying {len(selection)} runs, {_w} at a time (verify: {verify.value}) ---")
            status["replay"] = _af.replay_parallel(
                selection, _store, _cfg, _work, _proj, workers=_w,
                source={"kind": "hf", "repo": hf_repo.value, "allow_env": allow_env.value, "verify": verify.value},
                eval_kw={"n_log": 150, "n_uniform": 150}, device=_device, out_remote=_out,
                upload_every=int(upload_every.value),
                mem_fraction=min(0.5, max(0.08, 1.8 / _w)),  # ~2x a fair share: room for peaks, no runaway
                env=_env(_w), log=_log)
        _log("--- analyzing ---")
        status["analysis"] = _af.analyze_parallel(selection, _store, _cfg, _work, _work / "analysis", _proj,
                                                  workers=int(an_workers.value), env=_env(int(an_workers.value)),
                                                  log=_log)
        _meta = _af.build_meta(selection)
        _rows, _series, _diag = _af.load_analysis(_work / "analysis", _meta["run_id"])
        _have = set(_rows["run_id"]) if len(_rows) else set()
        _log(f"analyzed {len(_have)} of {len(selection)} runs; building the report ...")
        _meta = _meta[_meta["run_id"].isin(_have)].reset_index(drop=True)
        _files = _af.build_report(_rows, _series, _diag, _meta, _work / "report_AGOP")
        _files = {f"report_AGOP/{k}": v for k, v in _files.items()}
        for _name, _df in (("rows.parquet", _rows), ("series.parquet", _series)):
            _df.to_parquet(_work / _name, index=False)
            _files[_name] = _work / _name
        _diag.to_csv(_work / "diag.csv", index=False)
        _files["diag.csv"] = _work / "diag.csv"
        (_work / "status.json").write_text(json.dumps(
            {"device": _device, "minutes": round((time.time() - _t0) / 60, 1), "config": _cfg.to_dict(),
             "workers": {"replay": int(workers.value), "analysis": int(an_workers.value)},
             "verify": verify.value, "runs": status}, indent=2, default=str))
        _files["status.json"] = _work / "status.json"
        _out.commit(_files, message="agop final: analysis and report")
        _log(f"uploaded {len(_files)} files to {OUT_PREFIX}/")
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **error:**\n```\n{errors[-1]}\n```" if errors else ""))]
    _rep = _work / "report_AGOP"
    if (_rep / "REPORT_auto.md").exists():
        _view.append(mo.md((_rep / "REPORT_auto.md").read_text()))
        _view += [mo.image(src=p.read_bytes()) for p in sorted((_rep / "figures").glob("*.png"))]
    _failed = [s for v in status.values() for s in v if not s.get("ok")]
    if _failed:
        _view.append(mo.md(f"**{len(_failed)} run(s) failed** (they are skipped; press Run again to retry):"))
        _view.append(mo.ui.table(_failed, selection=None))
    mo.vstack(_view)
    return errors, status


if __name__ == "__main__":
    app.run()

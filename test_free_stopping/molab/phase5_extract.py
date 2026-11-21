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
"""Phase 5: the metrics of new_metrics.md on modular x+y and MNIST (PHASE5_PLAN.md).

Three buttons, run in order (each resumes from what is already on HF):
1. Recorded stage (CPU, ~10 min): weight norms, activation and weight spectra from the
   stored telemetry and summaries.
2. Replay stage (GPU, ~12-15 h for all runs; a session ends after 12 h, then press it
   again): one verified replay per run computes the AGOP products, the early grid points
   and the Adam+L2 optimizer state.
3. Curves (CPU, minutes): every metric on the common grid -> phase5/curves.parquet.

Outputs on advay333/tfs under phase5/: raw/recorded/, raw/replay/, curves.parquet,
runs.parquet, status_<stage>.json.
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

    OUT_PREFIX = "phase5"
    WORK = Path("/tmp/tfs_phase5")
    # measured in the F2 replay pass (seconds per run, one of 14 parallel workers)
    REPLAY_S = {"modular x+y": 837, "mnist d3": 591, "mnist d8": 1527}
    mo.md(
        f"""
        # Phase 5: metrics on modular x+y and MNIST

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token and press **Preview** to see the runs and the time estimate.
        3. Press the stage buttons in order: **Recorded**, **Replay**, **Curves**. Each one
           resumes from what is already in `advay333/tfs:{OUT_PREFIX}/`; if a session ends
           mid-replay, open a new one and press **Replay** again.
        4. Tell Claude "done".
        """
    )
    return OUT_PREFIX, Path, REPLAY_S, WORK, json, mo, os, subprocess, sys, time, traceback


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
def _(Path, mo, project_dir, pull_button, repo_url, subprocess, sys):
    mo.stop(not pull_button.value, mo.md("_Press **Clone / pull** to fetch the latest code._"))
    _proj = Path(project_dir.value)
    _root = _proj.parent
    if not (_root / ".git").exists():
        _out = subprocess.run(["git", "clone", repo_url.value, str(_root)], capture_output=True, text=True)
    else:
        _out = subprocess.run(["git", "-C", str(_root), "pull", "--ff-only"], capture_output=True, text=True)
    _head = subprocess.run(["git", "-C", str(_root), "log", "-1", "--oneline"], capture_output=True, text=True).stdout
    code_ready = (_proj / "trajectory" / "runner.py").exists()
    if code_ready and str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else
             f"**The pull ran, but `{_proj / 'trajectory'}` does not exist at HEAD above.** "
             "Either the Phase 5 code is not committed and pushed to GitHub yet, or the "
             "directory box points at the wrong folder."))
    return (code_ready,)


@app.cell
def _(mo, os, subprocess):
    _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True)
    gpu_line = _gpu.stdout.strip().splitlines()[0] if _gpu.returncode == 0 and _gpu.stdout.strip() else ""
    gpu_mem_gb = float(gpu_line.split(",")[-1]) / 1024 if gpu_line else 0.0
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # one replay ~ one CPU core and at most ~4 GB of GPU memory (MNIST depth 8 with the AGOP history)
    _rep_default = max(1, min(16, n_cpu - 1, int(0.6 * gpu_mem_gb // 4))) if gpu_mem_gb else 1
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo", full_width=True)
    rec_workers = mo.ui.number(start=1, stop=64, value=max(1, min(16, n_cpu - 1)), label="recorded stage: processes (CPU)")
    rep_workers = mo.ui.number(start=1, stop=32, value=_rep_default, label="replay stage: parallel replays (GPU)")
    include_dups = mo.ui.checkbox(value=False, label="include the gamma = 0 Adam+L2 runs (copies of their AdamW twins)")
    limit = mo.ui.number(start=0, stop=1000, value=0, label="only the first N runs, longest first (0 = all; use ~12 for a pilot)")
    pilot = mo.ui.checkbox(value=False, label="pilot: one run per (variant, optimizer), taken from F2's runs "
                                              "(12 runs; covers both MNIST networks and Adam+L2), then press 4")
    verify = mo.ui.checkbox(value=True, label="verify every replay against the recording (a diverging run fails)")
    allow_env = mo.ui.checkbox(value=False, label="allow replay in an environment that differs from the runs' "
                                                  "(the run must then still pass within tolerance, L1)")
    upload_every = mo.ui.number(start=1, stop=200, value=10, label="upload every N finished runs")
    preview_button = mo.ui.run_button(label="Preview")
    mo.vstack([mo.md(f"GPU: **{gpu_line or 'none found'}** ({gpu_mem_gb:.0f} GB); CPUs: **{n_cpu}**."),
               hf_token, hf_repo, rec_workers, rep_workers, include_dups, limit, pilot, verify, allow_env,
               upload_every, preview_button])
    return (allow_env, gpu_line, hf_repo, hf_token, include_dups, limit, n_cpu, pilot, preview_button, rec_workers,
            rep_workers, upload_every, verify)


@app.cell
def _(REPLAY_S, WORK, code_ready, hf_repo, hf_token, include_dups, limit, mo, pilot, preview_button, rep_workers):
    mo.stop(not preview_button.value, mo.md("_Press **Preview**._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))
    import pandas as _pd
    from tfs.storage.hub import HfRemote as _HfRemote
    from trajectory.corpus import load_corpus, pilot_runs, work_list

    _hf = _HfRemote(hf_repo.value, token=hf_token.value, create=False)
    table = load_corpus(_hf)
    if pilot.value:
        _f2 = _hf.download("phase4/agop_final/series.parquet", WORK / "f2_series.parquet")
        runs = pilot_runs(table, set(_pd.read_parquet(_f2, columns=["run_id"])["run_id"]))
    else:
        runs = work_list(table, include_duplicates=include_dups.value, limit=int(limit.value) or None)
    _sel = table[table["run_id"].isin({r["run_id"] for r in runs})]
    _counts = _pd.crosstab(_sel["variant"], _sel["category"], margins=True)
    _hours = sum(REPLAY_S.get(r["variant"], 1000) for r in runs) / 3600 / max(1, int(rep_workers.value))
    mo.vstack([
        mo.md(f"**{len(runs)} runs** ({int(table['duplicate_of'].notna().sum())} duplicates "
              f"{'included' if include_dups.value else 'skipped'}). Replay estimate with "
              f"{int(rep_workers.value)} parallel replays: **{_hours:.1f} h** (from F2's measured times, about +-30%)."),
        mo.ui.table(_counts.reset_index(), selection=None),
    ])
    return runs, table


@app.cell
def _(mo):
    rec_button = mo.ui.run_button(label="1. Recorded stage (CPU)")
    rep_button = mo.ui.run_button(label="2. Replay stage (GPU)")
    curves_button = mo.ui.run_button(label="3. Curves")
    check_button = mo.ui.run_button(label="4. Check against F2 (pilot)")
    mo.hstack([rec_button, rep_button, curves_button, check_button])
    return check_button, curves_button, rep_button, rec_button


@app.cell
def _(OUT_PREFIX, Path, WORK, allow_env, hf_repo, hf_token, json, n_cpu, os, project_dir, subprocess, time):
    def make_env(workers: int) -> dict:
        e = dict(os.environ)
        e["HF_TOKEN"] = hf_token.value  # workers read it from the environment; never written to disk
        n = str(max(1, n_cpu // max(1, workers)))
        e.update(OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, TFS_NUM_THREADS=n,
                 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        return e

    def gpu_state() -> str:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True)
        return q.stdout.strip().replace("\n", " | ") if q.returncode == 0 else "n/a"

    def out_remote():
        from tfs.storage import hub
        return hub.PrefixRemote(hub.HfRemote(hf_repo.value, token=hf_token.value, create=False), OUT_PREFIX)

    def run_stage(stage: str, runs: list, workers: int, device: str, verify: bool, upload_every: int):
        from trajectory import runner
        WORK.mkdir(parents=True, exist_ok=True)
        remote = out_remote()
        t0 = time.time()

        def log(msg):
            print(msg + (f"  [GPU: {gpu_state()}]" if stage == "replay" and "done," in msg else ""), flush=True)

        runner.resume(remote, WORK, stage, {r["run_id"] for r in runs}, log=log)
        status = runner.launch(stage, runs, WORK, Path(project_dir.value), workers=workers,
                               source={"kind": "hf", "repo": hf_repo.value}, device=device, verify=verify,
                               allow_env_mismatch=allow_env.value,
                               mem_fraction=min(0.5, max(0.08, 1.8 / workers)) if stage == "replay" else None,
                               out_remote=remote, upload_every=upload_every, env=make_env(workers), log=log)
        path = WORK / f"status_{stage}.json"
        path.write_text(json.dumps({"minutes": round((time.time() - t0) / 60, 1), "workers": workers,
                                    "device": device, "runs": len(runs), "status": status}, indent=2, default=str))
        remote.commit({f"status_{stage}.json": path}, message=f"phase5: {stage} status")
        return status

    return out_remote, run_stage


@app.cell
def _(mo, rec_button, rec_workers, run_stage, runs, traceback, upload_every):
    mo.stop(not rec_button.value, mo.md("_Press **1. Recorded stage** to run it._"))
    try:
        _st = run_stage("recorded", runs, int(rec_workers.value), "cpu", False, max(20, int(upload_every.value)))
        _bad = [s for s in _st if not s["ok"]]
        _view = mo.md(f"**Recorded stage:** {len(_st) - len(_bad)} ok, {len(_bad)} failed (press again to retry).")
        rec_result = mo.vstack([_view] + ([mo.ui.table(_bad, selection=None)] if _bad else []))
    except Exception:
        rec_result = mo.md(f"**Error:**\n```\n{traceback.format_exc()}\n```")
    rec_result
    return


@app.cell
def _(mo, rep_button, rep_workers, run_stage, runs, traceback, upload_every, verify):
    mo.stop(not rep_button.value, mo.md("_Press **2. Replay stage** to run it (GPU)._"))
    try:
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        if any(r["app"] == "mnist" for r in runs):
            from tfs.tasks.mnist import ensure_mnist as _ensure_mnist
            _ensure_mnist()  # once, before the workers start
        _st = run_stage("replay", runs, int(rep_workers.value), _device, verify.value, int(upload_every.value))
        _bad = [s for s in _st if not s["ok"]]
        _view = mo.md(f"**Replay stage ({_device}):** {len(_st) - len(_bad)} ok, {len(_bad)} failed "
                      "(press again to retry).")
        rep_result = mo.vstack([_view] + ([mo.ui.table(_bad, selection=None)] if _bad else []))
    except Exception:
        rep_result = mo.md(f"**Error:**\n```\n{traceback.format_exc()}\n```")
    rep_result
    return


@app.cell
def _(WORK, curves_button, mo, out_remote, runs, table, traceback):
    mo.stop(not curves_button.value, mo.md("_Press **3. Curves** after the stages you want._"))
    try:
        from trajectory import layout as _layout
        from trajectory import runner as _runner
        _remote = out_remote()
        _ids = {r["run_id"] for r in runs}
        for _stage in _layout.STAGES:
            _runner.resume(_remote, WORK, _stage, _ids, log=print)
        _tab = table[table["run_id"].isin(_ids)].reset_index(drop=True)
        _curves, _problems = _runner.build_curves(_tab, WORK)
        _tab.to_parquet(WORK / _layout.RUNS, index=False)
        _remote.commit({_layout.CURVES: WORK / _layout.CURVES, _layout.RUNS: WORK / _layout.RUNS},
                       message="phase5: curves and runs")
        _have = _curves.groupby("metric", observed=True)["run_id"].nunique().rename("runs").reset_index()
        curves_result = mo.vstack([
            mo.md(f"**Curves:** {_curves['run_id'].nunique()} runs, {len(_curves):,} rows; "
                  f"{len(_problems)} problem(s). Uploaded `curves.parquet` and `runs.parquet`."),
            mo.ui.table(_have, selection=None),
        ] + ([mo.ui.table(_problems, selection=None)] if _problems else []))
    except Exception:
        curves_result = mo.md(f"**Error:**\n```\n{traceback.format_exc()}\n```")
    curves_result
    return


@app.cell
def _(WORK, check_button, mo, out_remote, traceback):
    mo.stop(not check_button.value, mo.md("_After a pilot (stages 1-3), press **4. Check against F2**._"))
    try:
        import pandas as _pandas
        from trajectory.layout import CURVES as _CURVES
        from trajectory.validate import compare_with_f2
        _remote = out_remote()
        _f2 = WORK / "f2_series.parquet"
        if not _f2.exists():
            from tfs.storage import hub as _hub
            _hub.PrefixRemote(_remote.inner, "phase4/agop_final").download("series.parquet", _f2)
        _cmp = compare_with_f2(_pandas.read_parquet(WORK / _CURVES), _pandas.read_parquet(_f2))
        _cmp.to_csv(WORK / "check_f2.csv", index=False)
        _remote.commit({"check_f2.csv": WORK / "check_f2.csv"}, message="phase5: pilot check against F2")
        _worst = float(_cmp["max_rel_diff"].max()) if len(_cmp) else float("nan")
        check_result = mo.vstack([
            mo.md(f"**Check against F2:** {_cmp['run_id'].nunique()} runs; largest relative difference "
                  f"**{_worst:.2e}** (expected about 1e-6 or less: same weights, same formulas)."),
            mo.ui.table(_cmp, selection=None),
        ])
    except Exception:
        check_result = mo.md(f"**Error:**\n```\n{traceback.format_exc()}\n```")
    check_result
    return


if __name__ == "__main__":
    app.run()

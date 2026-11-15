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
"""Phase 4 study A: N1 (weight norm), R1 (effective rank of the activations) and B5
(HTSR alpha) as stopping rules (stopping/metric_study.py).

Modular, MNIST and teacher-student: whole hyperparameter cells (all 4 seeds). The three
signals come from data recorded during training, so there is no replay: each run's
telemetry, summaries and test curve are downloaded, the signals are computed once, and
every rule setting is scored offline. Settings are tuned on calibration runs (seeds 0-1;
also x+y, MNIST depth 3, teacher-student N 50 and 100) and scored on held-out runs (seeds
2-3; x*y and x/y; depth 8; N 400). CPU only.

Resumable: traces and curves upload to advay333/tfs under phase4/metrics_a/ as they
finish. Outputs there: rows.parquet, series.parquet, selection.csv, status.json, and one
folder per metric: report_N1/, report_R1/, report_B5/ (REPORT_auto.md, figures/, tables/).
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

    OUT_PREFIX = "phase4/metrics_a"

    mo.md(
        f"""
        # Phase 4, study A: N1, R1 and B5 as stopping rules

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token, check the options, press **Preview selection** and look at
           the run list and the time estimate.
        3. Press **Run**. Traces upload to `advay333/tfs` under `{OUT_PREFIX}/` as they finish,
           so an interrupted session resumes where it stopped (press Run again).
        4. Tell Claude "done". Everything Claude needs is in `{OUT_PREFIX}/`.

        No GPU is needed: the signals come from data recorded during training.
        """
    )
    return OUT_PREFIX, Path, json, mo, os, subprocess, sys, time, traceback


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
    code_ready = (_proj / "stopping" / "metric_study.py").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**stopping/metric_study.py not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(mo, os):
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    # each worker downloads a run (I/O) and fits power laws to its weight spectra (one core);
    # more than ~16 parallel downloads gains little and risks HF rate limits
    _default = max(1, min(16, n_cpu - 1))
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo", full_width=True)
    mod_cells = mo.ui.text(value="modular x+y:6, modular x*y:2, modular x/y:2",
                           label="modular: cells per (variant, optimizer); each cell = 4 seeds", full_width=True)
    mnist_cells = mo.ui.text(value="mnist d3:2, mnist d8:2", label="MNIST: cells per (variant, optimizer)",
                             full_width=True)
    ts_cells = mo.ui.text(value="teacher_student n50:2, teacher_student n100:2, teacher_student n400:2",
                          label="teacher-student: cells per (training-set size, optimizer)", full_width=True)
    workers = mo.ui.number(start=1, stop=64, value=_default, label="parallel worker processes")
    upload_every = mo.ui.number(start=1, stop=200, value=20, label="upload traces every N finished runs")
    preview_button = mo.ui.run_button(label="Preview selection")
    mo.vstack([mo.md(f"CPUs: **{n_cpu}**. Default workers: **{_default}**."), hf_token, hf_repo, mod_cells,
               mnist_cells, ts_cells, workers, upload_every, preview_button])
    return hf_repo, hf_token, mnist_cells, mod_cells, preview_button, ts_cells, upload_every, workers


@app.cell
def _(Path, code_ready, hf_repo, hf_token, mnist_cells, mo, mod_cells, preview_button, project_dir, sys, ts_cells,
      workers):
    mo.stop(not preview_button.value, mo.md("_Press **Preview selection**._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))
    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from stopping import metric_study as ms
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote

    def _parse(text):
        out = {}
        for part in text.split(","):
            if ":" in part:
                k, v = part.rsplit(":", 1)
                out[k.strip()] = int(v)
        return out

    _hf = _HfRemote(hf_repo.value, token=hf_token.value, create=False)
    _cells = {"modular": _parse(mod_cells.value), "mnist": _parse(mnist_cells.value),
              "teacher_student": _parse(ts_cells.value)}
    _rows = {a: _PrefixRemote(_hf, f"phase3/{a}").read_json("reports/corpus_runs.json") or []
             for a in ms.APPS if _cells[a]}
    selection = ms.select_runs(_rows, _cells)
    _notes = []
    for _a in ms.APPS:
        _sel = [r for r in selection if r["app"] == _a]
        if not _sel:
            continue
        _cats = {}
        for _r in _sel:
            if _r["seed"] == 0:
                _cats[_r["cell_category"]] = _cats.get(_r["cell_category"], 0) + 1
        _notes.append(f"- **{_a}**: {len(_rows.get(_a, []))} corpus runs; {len(_sel)} selected = "
                      f"{len(_sel) // 4} cells x 4 seeds; cells per seed-0 category {_cats}")
    # measured locally (download + signals + all rule settings): ~20 s modular, ~25 s
    # teacher-student, ~45 s MNIST depth 8; assume 40 s per run on average
    _min = len(selection) * 40 / int(workers.value) / 60
    _cols = ("app", "variant", "optim", "lr", "gamma", "alpha", "size", "seed", "cell_category", "category",
             "max_test", "t_eps", "run_id")
    mo.vstack([
        mo.md("\n".join(_notes) + f"\n\n**{len(selection)} runs.** Rough time: **{_min:.0f} min** with "
              f"{int(workers.value)} workers. Press **Run** below when this looks right."),
        mo.ui.table([{k: r.get(k) for k in _cols} for r in selection], selection=None, page_size=20),
    ])
    return ms, selection


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run")
    run_button
    return (run_button,)


@app.cell
def _(OUT_PREFIX, Path, hf_repo, hf_token, json, mo, ms, run_button, selection, time, traceback, upload_every,
      workers):
    mo.stop(not run_button.value, mo.md("_Press **Run** when ready._"))
    mo.stop(not selection, mo.md("**Preview a non-empty selection first.**"))

    from tfs.storage import hub as _hub

    _work = Path("/tmp/tfs_phase4_metrics_a")
    _work.mkdir(parents=True, exist_ok=True)
    _hf = _hub.HfRemote(hf_repo.value, token=hf_token.value, create=False)
    _out = _hub.PrefixRemote(_hf, OUT_PREFIX)
    errors, status = [], []
    _t0 = time.time()

    def _log(msg):
        print(msg, flush=True)

    try:
        _ids = {r["run_id"] for r in selection}
        ms.resume_from(_out, _work, _ids, log=_log)
        _log(f"--- {len(selection)} runs, {int(workers.value)} workers ---")
        status = ms.run_all(selection, _work, workers=int(workers.value), repo=hf_repo.value,
                             token=hf_token.value, out_remote=_out, upload_every=int(upload_every.value), log=_log)
        _meta = ms.build_meta(selection)
        _rows, _series = ms.load_analysis(_work, _meta["run_id"])
        _have = set(_rows["run_id"]) if len(_rows) else set()
        _log(f"analyzed {len(_have)} of {len(selection)} runs; building the reports ...")
        _meta = _meta[_meta["run_id"].isin(_have)].reset_index(drop=True)
        _files = ms.build_reports(_rows, _series, _meta, _work / "reports")
        for _name, _df in (("rows.parquet", _rows), ("series.parquet", _series)):
            _df.to_parquet(_work / _name, index=False)
            _files[_name] = _work / _name
        _meta.to_csv(_work / "selection.csv", index=False)
        _files["selection.csv"] = _work / "selection.csv"
        (_work / "status.json").write_text(json.dumps(
            {"minutes": round((time.time() - _t0) / 60, 1), "workers": int(workers.value),
             "runs": len(selection), "analyzed": len(_have), "status": status}, indent=2, default=str))
        _files["status.json"] = _work / "status.json"
        _out.commit(_files, message="metrics_a: analysis and per-metric reports")
        _log(f"uploaded {len(_files)} files to {OUT_PREFIX}/")
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **error:**\n```\n{errors[-1]}\n```" if errors else ""))]
    for _m in ("N1", "R1", "B5"):
        _rep = _work / "reports" / f"report_{_m}"
        if (_rep / "REPORT_auto.md").exists():
            _view.append(mo.md(_rep.joinpath("REPORT_auto.md").read_text().split("![")[0]))
            _view += [mo.image(src=p.read_bytes()) for p in sorted((_rep / "figures").glob("*.png"))]
    _failed = [s for s in status if not s.get("ok")]
    if _failed:
        _view.append(mo.md(f"**{len(_failed)} run(s) failed** (they are skipped; press Run again to retry):"))
        _view.append(mo.ui.table(_failed, selection=None))
    mo.vstack(_view)
    return errors, status


if __name__ == "__main__":
    app.run()

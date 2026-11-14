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
"""Phase 4 F2 experiment: the remaining-angle bound on real corpus runs (AGOP.md §4,
THEORY.md §4, stopping/agop_study.py).

Picks a spread-out set of Phase 3 runs per application (one seed-0 run per variant x
optimizer x category, plus gamma = 0 runs and extra seeds of a few grokkers), replays
each once and records the agop_path trace: pairwise distances between the unit-normalized
AGOP G, the feature matrix M = G^{1/2} and each layer's weights, at ~300 eval steps.
Every estimator setting (object, lag, window, noise floor, kappa estimator, threshold,
norm gate) is then evaluated offline from those traces and scored against the test curves.

Resumable: per-run traces are uploaded every few runs to advay333/tfs under
phase4/agop_study/, and a new session downloads them before continuing. Outputs:
signals/ (traces), curves/, plots/<app>/, settings_runs.csv, settings_summary.csv,
questions.csv, selection.csv, status.json and summary.md.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import subprocess
    import sys
    import time
    import traceback
    from pathlib import Path

    import marimo as mo

    OUT_PREFIX = "phase4/agop_study"

    mo.md(
        f"""
        # Phase 4 F2 experiment: remaining-angle bound on corpus runs

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token, choose the options, press **Preview selection** and check
           the run list and the time estimate.
        3. Press **Run**. Traces upload every few runs to `advay333/tfs` under
           `{OUT_PREFIX}/`, so an interrupted session resumes where it stopped.
        4. Tell Claude "done". Read `summary.md` there first.
        """
    )
    return OUT_PREFIX, Path, json, mo, subprocess, sys, time, traceback


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
    code_ready = (_proj / "stopping" / "agop_study.py").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**F2 study code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo", full_width=True)
    apps = mo.ui.multiselect(options=["mnist", "modular", "teacher_student"],
                             value=["mnist", "modular", "teacher_student"], label="applications")
    modular_variants = mo.ui.text(value="modular x+y", label="modular variants (comma separated; empty = all)",
                                  full_width=True)
    categories = mo.ui.multiselect(options=["grokker", "fast", "late_collapse", "joint_plateau", "never", "no_fit"],
                                   value=["grokker", "fast", "late_collapse"], label="categories (seed 0)")
    per_cell = mo.ui.number(start=1, stop=3, value=1, label="runs per (variant, optimizer, category)")
    gamma0 = mo.ui.checkbox(value=True, label="add one gamma = 0 run per (variant, optimizer)")
    seed_grokkers = mo.ui.number(start=0, stop=4, value=2, label="grokkers per app that also get seeds 1 and 2")
    max_runs = mo.ui.number(start=0, stop=500, value=0, label="max runs in total (0 = all selected)")
    n_eval = mo.ui.number(start=40, stop=600, value=300, label="eval steps per run (half log-spaced, half uniform)")
    weights = mo.ui.checkbox(value=True, label="also record per-layer weight-direction paths (THEORY.md §3)")
    upload_every = mo.ui.number(start=1, stop=50, value=5, label="upload traces every N runs")
    analysis_only = mo.ui.checkbox(value=False, label="analysis only: no replay, use the traces already uploaded")
    allow_env = mo.ui.checkbox(value=False, label="Allow replay in an environment that differs from the run's")
    preview_button = mo.ui.run_button(label="Preview selection")
    mo.vstack([hf_token, hf_repo, apps, modular_variants, categories, per_cell, gamma0, seed_grokkers, max_runs,
               n_eval, weights, upload_every, analysis_only, allow_env, preview_button])
    return (allow_env, analysis_only, apps, categories, gamma0, hf_repo, hf_token, max_runs, modular_variants,
            n_eval, per_cell, preview_button, seed_grokkers, upload_every, weights)


@app.cell
def _(Path, apps, categories, code_ready, gamma0, hf_repo, hf_token, max_runs, mo, modular_variants, per_cell,
      preview_button, project_dir, seed_grokkers, sys):
    mo.stop(not preview_button.value, mo.md("_Press **Preview selection**._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))
    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    from stopping.agop_study import pick_study_runs as _pick
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote

    _hf = _HfRemote(hf_repo.value, token=hf_token.value, create=False)
    _per_app, _notes = [], []
    for _app in apps.value:
        _rows = _PrefixRemote(_hf, f"phase3/{_app}").read_json("reports/corpus_runs.json") or []
        _vars = None
        if _app == "modular" and modular_variants.value.strip():
            _vars = [v.strip() for v in modular_variants.value.split(",") if v.strip()]
        _sel = _pick(_rows, _app, categories=categories.value, per_cell=int(per_cell.value), variants=_vars,
                     gamma0=gamma0.value, seed_grokkers=int(seed_grokkers.value))
        _notes.append(f"- **{_app}**: {len(_rows)} corpus runs, {len(_sel)} selected; variants present: "
                      f"{sorted({r['variant'] for r in _rows})}")
        _per_app.append(_sel)
    # interleave the applications, so that a truncated or interrupted run still covers all of them
    selection = [lst[i] for i in range(max((len(x) for x in _per_app), default=0)) for lst in _per_app if i < len(lst)]
    if int(max_runs.value) > 0:
        selection = selection[: int(max_runs.value)]
    _hours = sum(float(r.get("wall_s") or 0) for r in selection) / 3600
    _cols = ("app", "variant", "optim", "category", "role", "alpha", "gamma", "size", "seed", "max_test", "t_eps",
             "wall_s", "run_id")
    mo.vstack([
        mo.md("\n".join(_notes) + f"\n\n**{len(selection)} runs.** Rough replay time: about **{_hours:.1f} h** "
              "(the runs' own training wall time, measured with 16 runs sharing the GPU, so a single replay is "
              "probably faster). Press **Run** below when this looks right."),
        mo.ui.table([{k: r.get(k) for k in _cols} for r in selection], selection=None, page_size=25),
    ])
    return (selection,)


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run")
    run_button
    return (run_button,)


@app.cell
def _(OUT_PREFIX, Path, allow_env, analysis_only, hf_repo, hf_token, json, mo, n_eval, run_button, selection,
      time, traceback, upload_every, weights):
    mo.stop(not run_button.value, mo.md("_Press **Run** when ready._"))
    mo.stop(not selection, mo.md("**Preview a non-empty selection first.**"))

    import torch as _torch
    from stopping import agop_study as _st
    from stopping.signals import AgopPathConfig as _Cfg
    from stopping.store import TraceStore as _TraceStore
    from stopping.tfs_source import HfRunSource as _HfRunSource
    from tfs.storage import hub as _hub
    from tfs.training import determinism as _det

    _det.apply()
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    _work = Path("/tmp/tfs_phase4_agop_study")
    _work.mkdir(parents=True, exist_ok=True)
    _out = _hub.PrefixRemote(_hub.HfRemote(hf_repo.value, token=hf_token.value, create=False), OUT_PREFIX)
    _cfg = _Cfg(weights=weights.value)
    _store = _TraceStore(_work / "traces")
    errors, status, result = [], [], {}
    _t0 = time.time()

    def _log(msg):
        print(msg, flush=True)

    try:
        _st.resume_from_remote(_out, _store, _cfg, _work, log=_log)
        if not analysis_only.value:
            if any(r["app"] == "mnist" for r in selection):
                from tfs.tasks.mnist import ensure_mnist as _ensure_mnist
                _ensure_mnist()
            _sources = {a: _HfRunSource(hf_repo.value, f"phase3/{a}", work=_work / "runs", device=_device,
                                        token=hf_token.value, allow_env_mismatch=allow_env.value)
                        for a in sorted({r["app"] for r in selection})}
            _n = int(n_eval.value)

            def _log_and_free(msg):
                _log(msg)
                if _device == "cuda":
                    _torch.cuda.empty_cache()

            status = _st.run_study(_sources, selection, _store, _cfg, _work, out_remote=_out,
                                   eval_kw={"n_log": _n // 2, "n_uniform": _n - _n // 2},
                                   upload_every=int(upload_every.value), log=_log_and_free)
        _log("analyzing ...")
        result = _st.analyze_all(selection, _store, _cfg, _work, _work / "out", log=_log)
        _files = dict(result["files"])
        (_work / "out" / "status.json").write_text(json.dumps(
            {"device": _device, "minutes": round((time.time() - _t0) / 60, 1), "config": _cfg.to_dict(),
             "runs": status}, indent=2, default=str))
        _files["status.json"] = _work / "out" / "status.json"
        _out.commit(_files, message="agop study: analysis, plots, summary")
        _log(f"uploaded {len(_files)} files to {OUT_PREFIX}/")
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **error:**\n```\n{errors[-1]}\n```" if errors else ""))]
    if status:
        _view.append(mo.ui.table(status, selection=None))
    if result:
        _view.append(mo.md(result["markdown"]))
        _view += [mo.image(src=p.read_bytes()) for p in result["plots"][:12] if p.exists()]
    mo.vstack(_view)
    return errors, result, status


if __name__ == "__main__":
    app.run()

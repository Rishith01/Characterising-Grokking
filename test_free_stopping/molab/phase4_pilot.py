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
"""Phase 4 pilot: the built signals on a handful of MNIST corpus runs (DESIGN.md §13.6).

Picks up to N seed-0 runs of each provisional category from phase3/mnist (default 3, so
at most 18 runs), downloads each with only the parts the signals need, computes every
built signal (N1, R1, B0-B6; S1, S1b and F2 are not built yet), and plots each signal
under the run's train/test accuracy with every method's stop marked. Rule settings are
uncalibrated placeholders: this is for checking that the signals behave as METRICS.md
predicts, not for scoring (Phase 5).

The replay signals (B2, B3 replay form, B5 traps) replay each run from its keyframes,
which costs about one training run per run (~5 min for MNIST depth 8 on the GPU, run one
at a time). Untick them for a fast, telemetry-and-summaries-only pass.

Outputs go to advay333/tfs under phase4/pilot/mnist/: signals/ (the traces), plots/,
stops.csv and summary.md.
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

    DATASET = "mnist"

    mo.md(
        f"""
        # Phase 4 pilot: signals on **{DATASET}** runs

        1. Check the repo location and press **Clone / pull**.
        2. Enter the HF token, choose the options, and press **Run**. With replay signals on,
           expect ~1-1.5 h for 18 runs; without, a few minutes.
        3. Tell Claude "done". Plots and a stop table are uploaded to `advay333/tfs` under
           `phase4/pilot/{DATASET}/`.
        """
    )
    return DATASET, Path, json, mo, subprocess, sys, time, traceback


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
    code_ready = (_proj / "stopping" / "pilot.py").exists()
    mo.md(f"```\n{_out.stdout}{_out.stderr}\nHEAD: {_head}\n```\n\n"
          + ("**Code ready.**" if code_ready else "**Phase 4 code not found. Pull the latest commit.**"))
    return (code_ready,)


@app.cell
def _(mo):
    hf_token = mo.ui.text(kind="password", label="HF write token (not stored)", full_width=True)
    hf_repo = mo.ui.text(value="advay333/tfs", label="HF dataset repo", full_width=True)
    per_category = mo.ui.number(start=1, stop=10, value=3, label="runs per provisional category (seed 0)")
    with_replay = mo.ui.checkbox(value=True, label="Replay signals (B2 EB, B3 replay form, B5 traps)")
    n_eval = mo.ui.number(start=20, stop=600, value=200, label="eval steps per run (half log-spaced, half uniform)")
    allow_env = mo.ui.checkbox(value=False, label="Allow replay in an environment that differs from the run's")
    run_button = mo.ui.run_button(label="Run")
    mo.vstack([hf_token, hf_repo, per_category, with_replay, n_eval, allow_env, run_button])
    return allow_env, hf_repo, hf_token, n_eval, per_category, run_button, with_replay


@app.cell
def _(DATASET, Path, allow_env, code_ready, hf_repo, hf_token, json, mo, n_eval, per_category, project_dir,
      run_button, sys, time, traceback, with_replay):
    mo.stop(not run_button.value, mo.md("_Press **Run** when ready._"))
    mo.stop(not code_ready, mo.md("**Clone / pull the code first.**"))
    mo.stop(not hf_token.value, mo.md("**Enter the HF token.**"))

    _proj = Path(project_dir.value)
    if str(_proj) not in sys.path:
        sys.path.insert(0, str(_proj))
    import torch as _torch
    from stopping.methods import default_methods as _methods
    from stopping.pilot import pick_runs as _pick, run_pilot as _run_pilot, stops_markdown as _stops_md
    from stopping.store import TraceStore as _TraceStore
    from stopping.tfs_source import HfRunSource as _HfRunSource
    from tfs.evaluation.report import rows_to_markdown as _md
    from tfs.storage.hub import HfRemote as _HfRemote, PrefixRemote as _PrefixRemote
    from tfs.tasks.mnist import ensure_mnist as _ensure_mnist
    from tfs.training import determinism as _det

    _det.apply()
    _device = "cuda" if _torch.cuda.is_available() else "cpu"
    _work = Path(f"/tmp/tfs_phase4_pilot_{DATASET}")
    _work.mkdir(parents=True, exist_ok=True)  # kept across re-runs: cached traces are reused
    _ensure_mnist()
    _hf = _HfRemote(hf_repo.value, token=hf_token.value, create=False)
    _corpus = _PrefixRemote(_hf, f"phase3/{DATASET}")
    _out = _PrefixRemote(_hf, f"phase4/pilot/{DATASET}")
    errors, plot_paths, result = [], [], {}
    _t0 = time.time()

    def _log(msg):
        print(msg, flush=True)

    try:
        _rows = _corpus.read_json("reports/corpus_runs.json") or []
        _runs = _pick(_rows, per_category=int(per_category.value))
        _log("runs:\n" + _md([{k: r[k] for k in ("run_id", "category", "variant", "optim", "alpha", "gamma",
                                                 "max_test", "t_eps")} for r in _runs]))
        _source = _HfRunSource(hf_repo.value, f"phase3/{DATASET}", work=_work / "runs", device=_device,
                               token=hf_token.value, allow_env_mismatch=allow_env.value)
        _store = _TraceStore(_work / "traces")
        _n = int(n_eval.value)
        result = _run_pilot(_source, _runs, _methods(replay=with_replay.value), _store, _work / "out",
                            eval_kw={"n_log": _n // 2, "n_uniform": _n - _n // 2}, log=_log)
        plot_paths = result["plots"]
        _table = _stops_md(result["stops"])
        _log("\n## Stops: step (test acc at stop); rules are uncalibrated\n" + _table)

        _files = {f"plots/{p.name}": p for p in plot_paths}
        _files["stops.csv"] = _work / "out" / "stops.csv"
        _cfgs = sorted({c.key: c for m in _methods(replay=with_replay.value) for c in m.signals()}.values(),
                       key=lambda c: c.key)
        _files.update(_store.upload_files([c for c in _cfgs if _store.run_ids(c)]))
        (_work / "out" / "status.json").write_text(json.dumps(result["status"], indent=2))
        _files["status.json"] = _work / "out" / "status.json"
        (_work / "out" / "summary.md").write_text(
            f"# Phase 4 pilot: {DATASET}\n\n- device: {_device}; replay signals: {with_replay.value}; "
            f"eval steps per run: {_n}; minutes: {(time.time() - _t0) / 60:.0f}\n"
            f"- rule settings are uncalibrated placeholders (stopping/methods.py)\n\n"
            f"## Runs\n\n" + _md([{k: r[k] for k in ("run_id", "category", "variant", "optim", "alpha", "gamma",
                                                     "max_test", "t_eps")} for r in _runs])
            + "\n## Stops: step (test acc at stop)\n\n" + _table
            + "\n## Compute status\n\n" + _md(result["status"]))
        _files["summary.md"] = _work / "out" / "summary.md"
        _out.commit(_files, message=f"phase 4 pilot {DATASET}: signals, plots, stops")
        _log(f"uploaded {len(_files)} files to phase4/pilot/{DATASET}/")
    except Exception:
        errors.append(traceback.format_exc())
        _log(errors[-1])

    _view = [mo.md(f"## Done in {(time.time() - _t0) / 60:.0f} min"
                   + (f" — **error:**\n```\n{errors[-1]}\n```" if errors else ""))]
    if result:
        _view.append(mo.ui.table(result["status"], selection=None))
    _view += [mo.image(src=p.read_bytes()) for p in plot_paths if p.exists()]
    mo.vstack(_view)
    return errors, plot_paths, result


if __name__ == "__main__":
    app.run()

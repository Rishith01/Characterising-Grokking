# test_free_stopping

Causal, test-data-free stopping criteria that survive grokking.

**Results so far in one place: [SUMMARY.md](SUMMARY.md)** (F2, N1, R1, with the
expected questions). **New here? Start with [RAMPUP.md](RAMPUP.md)**, then [PLAN.md](PLAN.md) (what we run and
why) and [DESIGN.md](DESIGN.md) (how the code works). Metric definitions are in
[METRICS.md](METRICS.md); the AGOP background and the proposed F2 signal are in
[AGOP.md](AGOP.md).

## Status

- **Phase 0** (scaffold and infrastructure): done. G0 passed on Molab, bit-exact (L0) on
  CPU and on the Blackwell GPU.
- **Phase 1** (tasks, recording, determinism, reproductions): done, after the Phase 1b
  clean-up run. See [results/phase1/RESULTS.md](results/phase1/RESULTS.md).
- **Phase 2** (pilot sweep): done (2a, 2b, 2c; [results/phase2/RESULTS.md](results/phase2/RESULTS.md)).
  G2 passed and the Phase 3 grid is frozen (PLAN.md §3).
- **Phase 3** (corpus, 1,664 runs; one notebook per dataset): done: MNIST ✅ (256 runs),
  teacher–student ✅ (576 runs) and modular ✅ (832 runs) are trained and pass G3
  ([results/phase3/RESULTS.md](results/phase3/RESULTS.md)).
- **Phase 4** (signals): in progress. Core built in `stopping/` (DESIGN.md §13): N1, R1
  and baselines B0–B6, piloted on MNIST with `molab/phase4_pilot.py` on 2026-11-04
  ([results/phase4/RESULTS.md](results/phase4/RESULTS.md)). S1, S1b and F2 are not built
  yet; their open decisions are in METRICS.md §6 and AGOP.md §6. The F2 experiment
  (`molab/phase4_agop_study.py`, DESIGN.md §13.9) ran on 2026-11-14; the final F2 study
  (`molab/phase4_agop_final.py`, DESIGN.md §13.10) ran on 2026-11-15 and closes F2;
  see [report_AGOP/REPORT.md](report_AGOP/REPORT.md). Verdict: on modular, F2 halves the
  cost of training to the end and never cuts off a grokker, but only ties a tuned fixed
  fraction of the budget; on MNIST it loses to the fixed fraction, and even an oracle
  that knows the final AGOP does not help. Study A (2026-11-16) tested N1 and R1 ([report_N1](report_N1/REPORT.md),
  [report_R1](report_R1/REPORT.md)); B5, B2 and S1b come in study B. S1 and S1b remain unbuilt. The theory review of the metric set is in [THEORY.md](THEORY.md).
- **Phases 5–6** (calibration, evaluation, write-up): not started.

## Local setup (development, CPU)

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu   # Windows path
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
```

## Commands

| Command | Does |
|---|---|
| `python -m scripts.train configs/runs/dummy.yaml --runs-root runs` | Train one run locally |
| `python -m scripts.replay_check <run_dir> [--full]` | Verify deterministic replay (L0 / L1 / failed) |
| `python -m scripts.det_check <sweep.yaml> --device cuda --workers 8` | Gate G1-det: train each config twice, compare, replay |
| `python -m scripts.throughput --workers 8 16 32 --mps-workers 16` | GPU throughput at several concurrency levels (and MPS), with a bit-identity check |
| `HF_TOKEN=... python -m scripts.sweep <sweep.yaml> --workers 16 --repo advay333/tfs --prefix phase3` | Train, replay-check, summarize and upload a sweep; resumes from the remote manifest |
| `python -m scripts.env_lock --out configs/env_lock.txt` / `--check configs/env_lock.txt` | Record the environment, or compare the live one with the lock |
| `HF_TOKEN=... python -m scripts.sync push --repo advay333/<repo>` | Upload complete runs, verify remotely, delete local copies |
| `python -m scripts.sync pull --repo ... --app <app> --run-id <id>` | Download one run and check its checksums |
| `python -m scripts.sync status --repo ...` | Summarize the remote manifest |

## Molab

Every GPU or Hugging Face job is a marimo notebook in `molab/`. Clone this repo on Molab
and open the notebook. Each notebook asks for the HF token in a password field (it is
never stored) and has a **Clone / pull** button to fetch the latest code first.

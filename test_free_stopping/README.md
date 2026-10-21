# test_free_stopping

Causal, test-data-free stopping criteria that survive grokking. The plan is in
[PLAN.md](PLAN.md) and the architecture is in [DESIGN.md](DESIGN.md).

## Status

- **Phase 0** (scaffold and infrastructure): done. G0 passed on Molab, bit-exact (L0) on
  CPU and on the Blackwell GPU.
- **Phase 1** (tasks, recording, determinism, reproductions): run on Molab. GPU fully
  deterministic; open items (CPU drift, teacher–student accuracy, summary size, SGD
  learning rates, MNIST horizon, two-stage case) go to the Phase 1b clean-up run:
  [`molab/phase1b.py`](molab/phase1b.py).

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
| `HF_TOKEN=... python -m scripts.sync push --repo advay333/<repo>` | Upload complete runs, verify remotely, delete local copies |
| `python -m scripts.sync pull --repo ... --app <app> --run-id <id>` | Download one run and check its checksums |
| `python -m scripts.sync status --repo ...` | Summarize the remote manifest |

## Molab

Every GPU or Hugging Face job is a marimo notebook in `molab/`. Clone this repo on Molab
and open the notebook. Each notebook asks for the HF token in a password field (it is
never stored) and has a **Clone / pull** button to fetch the latest code first.

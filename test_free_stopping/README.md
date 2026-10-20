# test_free_stopping

Causal, test-data-free stopping criteria that survive grokking. The plan is in
[PLAN.md](PLAN.md) and the architecture is in [DESIGN.md](DESIGN.md).

## Status

**Phase 0 (scaffold and infrastructure).** Code and tests are done locally. The G0 gate
on Molab is pending: [`molab/phase0_g0.py`](molab/phase0_g0.py).

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
| `HF_TOKEN=... python -m scripts.sync push --repo advay333/<repo>` | Upload complete runs, verify remotely, delete local copies |
| `python -m scripts.sync pull --repo ... --app <app> --run-id <id>` | Download one run and check its checksums |
| `python -m scripts.sync status --repo ...` | Summarize the remote manifest |

## Molab

Every GPU or Hugging Face job is a marimo notebook in `molab/`. Clone this repo on Molab
and open the notebook. Each notebook asks for the HF token in a password field (it is
never stored) and has a **Clone / pull** button to fetch the latest code first.

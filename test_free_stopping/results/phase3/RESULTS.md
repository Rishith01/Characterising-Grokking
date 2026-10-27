# Phase 3 results (corpus)

One notebook per dataset (`molab/phase3_<dataset>.py`). Runs and reports live in the HF
dataset `advay333/tfs` under `phase3/<dataset>/` (each with its own `manifest.json`;
reports under `reports/`).

| Dataset | Runs | Status | G3 |
|---|---|---|---|
| MNIST | 256 | ✅ done 2026-10-22 | ✅ passes |
| modular | 832 | not started | |
| teacher–student | 576 | notebook ready (`molab/phase3_teacher_student.py`) | |

## MNIST (2026-10-22, 84 min incl. audit, MPS, 16 workers)

Figures (seed 0, one panel per optimizer): [`corpus_mnist_d3_seed0.png`](corpus_mnist_d3_seed0.png),
[`corpus_mnist_d8_seed0.png`](corpus_mnist_d8_seed0.png).

### Infrastructure: G3 passes

- Environment matched `configs/env_lock.txt`.
- **256/256 runs uploaded, all `complete`, none diverged, every one replay-checked at L0**
  (one random keyframe interval each).
- **Audit:** 13 random runs (5%) re-downloaded, all checksums matched, all **fully
  replayed at L0 with zero error** (10 keyframe intervals, 2,001 log steps each).
- **Cross-session reproducibility:** 17 corpus runs have exactly the configuration of a
  Phase 2b/2c pilot run (same run id). All 17 reproduce the pilot's max test accuracy,
  t_ε and final test accuracy to every recorded digit, in a different Molab session.
- Cost as projected: 77 min of training (projection: ~1.3 GPU-h); 21.8 GB (projection
  20.2 GB; depth-8 runs are 145 MB instead of 133 MB).

### What the corpus contains

Seeds agree closely: within a configuration, max test accuracy varies by 0.001–0.005
(s.d.) in almost every cell. The exceptions are the slow depth-8 SGD-family runs at
α = 4 (s.d. 0.03–0.15).

| Configuration | Behaviour (4 seeds) | Matches the pilot / literature? |
|---|---|---|
| depth 3, Adam family, α = 8 | **grokking**, t_ε ∝ 1/γ: AdamW 71k / 47k / 18k at γ = 0.01 / 0.03 / 0.1; γ = 0 groks late (80k) to a lower max (0.85) | ✅ Omnigrok Fig. 3d; Phase 1 (88k / 52k / 23k) |
| depth 3, Adam family, α = 1 | test peaks at step 100 (0.92), then decays: **late collapse** for γ ≤ 0.03 | ✅ the Phase 1 control |
| depth 3, SGD family, α = 1 | **joint plateau**: train and test rise together, fit at 3–17k | new at this size; expected for SGD at small init |
| depth 3, SGD family, α = 4 | grokking at small γ (31k–74k), fast at large γ; γ = 0 generalizes worse (0.78–0.81) | ✅ Phase 2c |
| depth 3, coupled γ = 3e-3 | cannot fit (train ≈ 0.98): `no_fit` | ✅ Phase 2b |
| depth 8, Adam family, α = 1 | fast (t_ε ≈ 0.7–1.4k, test 0.955) | ✅ Phase 2c |
| depth 8, Adam family, α = 8 | **joint plateau then creep**: train and test flat at 0.10 until ~2k steps, jump together, then test creeps to its max at 8k–25k | ✅ Phase 1 / 2c depth-8 behaviour |
| depth 8, Adam+L2, α = 1, γ = 1e-3 | collapses to the constant output (test 0.114): **never** | not tested in the pilot (only at α = 4); a plausible extra trap |
| depth 8, SGD, α = 2 | joint plateau (γ ≤ 1e-3); γ = 1e-2 peaks at 0.66 then collapses to the constant output | ✅ Phase 2c |
| depth 8, SGD family, α = 4 | slow learners still fitting at 1e5, or never / late collapse | ✅ Phase 2c |

Provisional categories (`evaluation/run_summary.provisional_category`):

| | fast | grokker | joint plateau | late collapse | never | no fit |
|---|---|---|---|---|---|---|
| MNIST depth 3 | 35 | 42 | 23 | 16 | 0 | 12 |
| MNIST depth 8 | 72 | 2 | 15 | 9 | 15 | 15 |

Grok-time (t_ε) spread: 2.9 decades (depth 3), 2.8 decades (depth 8).

**One labelling caveat, not a training problem.** The provisional rule calls a run a
grokker only if t_ε ≥ 10 × the step it fits the training set, and a joint plateau only
if t_ε ≤ 2 × that step. The depth-8 α = 8 Adam runs fit at ~2.4k and reach their max at
8–25k after a slow creep, so they land in "fast" although their curves are a joint
plateau. Phase 5's `GroundTruth` should define the categories from the curve shapes,
not from this ratio. Stopping rules are scored on regret, which does not use the
categories, so the corpus itself is unaffected.

**Verdict: as expected.** Every configuration behaves as the pilot predicted, and the
three traps a stopping rule must survive (grokking plateau, joint plateau, late
collapse) are all present with consistent seeds.

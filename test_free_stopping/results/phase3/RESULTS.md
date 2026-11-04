# Phase 3 results (corpus)

One notebook per dataset (`molab/phase3_<dataset>.py`). Runs and reports live in the HF
dataset `advay333/tfs` under `phase3/<dataset>/` (each with its own `manifest.json`;
reports under `reports/`).

| Dataset | Runs | Status | G3 |
|---|---|---|---|
| MNIST | 256 | ✅ done 2026-10-22 | ✅ passes |
| modular | 832 | ✅ done 2026-11-04 | ✅ passes |
| teacher–student | 576 | ✅ done 2026-10-28 | ✅ passes |

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

## Teacher–student (2026-10-28, 270 min incl. audit, MPS, 16 workers)

Figure (seed 0): [`corpus_teacher_student_seed0.png`](corpus_teacher_student_seed0.png).

### Infrastructure: G3 passes

- Environment matched the lock. **576/576 runs uploaded, all `complete`, none diverged,
  all replay-checked at L0.**
- **Audit:** 29 random runs (5%) re-downloaded and **fully replayed at L0 with zero error**
  (30 keyframe intervals, 6,001 log steps each).
- **Cross-session reproducibility:** the 8 corpus runs whose configuration equals a pilot
  run reproduce it exactly.
- Cost: 255 min of training (projection ~3.8 h; mean 396 s per run vs 384 s projected);
  6.3 GB (projection 5.9 GB).

### What the corpus contains

Seed spread is larger than on MNIST (median s.d. of max test accuracy 0.02, max 0.065),
as expected with N = 50–400 training points.

- **The LU picture holds.** At α = 2 with γ = 0, the student never generalizes at
  N = 50 and 100 (test ≈ 0) for every optimizer, but does at N = 400 (late, ~260k
  steps): more data moves the run past the critical data size. AdamW at α = 2 groks at
  t_ε ≈ 135k / 40–46k / 3–12k for γ = 0.03 / 0.1 / 1 (∝ 1/γ, Omnigrok Fig. 2c).
- **Training-set size sets the ceiling:** max test accuracy ≈ 0.55–0.6 at N = 50,
  0.8–0.9 at N = 100, 0.95–1.0 at N = 400.
- **Coupled weight decay at γ = 1e-4** (Adam+L2, SGD, SGD+momentum) stops the student
  fitting below θ: never-generalizing or late-collapse runs in every (α, N) cell, as in
  the pilot.
- **Plain SGD is slow:** median t_ε 160k–295k in most cells.
- **The 3e5 horizon was needed:** 203 of 576 runs reach their best test accuracy after
  step 1e5.

Provisional categories: grokker 244, never 144, fast 94, late collapse 51, joint
plateau 25, no fit 18. Grok-time spread 2.8 decades. 122 runs end with train accuracy
< 0.99 (the coupled-L2 floor and slow SGD).

**Two notes.**
- Many "grokkers" here are slow creeps rather than a plateau then a jump: the student
  fits the training set by ~350–600 steps and test accuracy rises gradually for tens of
  thousands of steps, which passes the provisional 10× rule. The clean plateau-then-jump
  runs are AdamW/Adam+L2 at α = 2 (as in Omnigrok). Same labelling caveat as MNIST;
  regret does not depend on it.
- Plain SGD at γ ≤ 1e-5 often peaks in the last ~10% of the horizon (t_ε 258k–295k), so
  some of those runs may still be improving at 3e5. For scoring this is harmless (the
  best test accuracy is taken within the horizon), but a rule that never stops scores
  well on them; this is worth keeping in mind in Phase 5.

**Verdict: as expected.** Every pattern the pilot and Omnigrok predict appears, with
consistent seeds and all three traps present.

## Modular (2026-11-04, 458 min incl. audit, MPS, 16 workers)

Figures (seed 0; x+y at train fraction 0.5): [`corpus_modular_xplusy_seed0.png`](corpus_modular_xplusy_seed0.png),
[`corpus_modular_xmuly_seed0.png`](corpus_modular_xmuly_seed0.png),
[`corpus_modular_xdivy_seed0.png`](corpus_modular_xdivy_seed0.png).

### Infrastructure: G3 passes

- Environment matched the lock. **832/832 runs uploaded, all `complete`, none diverged,
  all replay-checked at L0.**
- **Audit:** 42 random runs (5%) re-downloaded and **fully replayed at L0 with zero
  error** (10 keyframe intervals, 2,001 log steps each).
- **Cross-session reproducibility:** the 24 corpus runs whose configuration equals a
  Phase 2b/2c pilot run reproduce its max test accuracy, t_ε, final test accuracy and
  peak step exactly.
- **Internal check:** at γ = 0, AdamW and Adam+L2 are both plain Adam, and all 52 such
  pairs give identical results.
- Cost: 428 min of training and upload (projection ~5.4 GPU-h; 440 s per run against
  355–437 s projected, so x+y ran ~25% slower than its pilot estimate). 50.5 GB
  (projection ~51 GB).

### What the corpus contains

Seeds agree exactly in most cells (median s.d. of max test 0.000). The spread is large
only at the edges: train fraction 0.3 at the slowest decay (runs still rising at the
horizon) and plain SGD at γ = 1e-3.

| Configuration | Behaviour (4 seeds) | Matches the pilot / literature? |
|---|---|---|
| AdamW, γ = 0.1 / 0.3 / 1 (x+y, f 0.5, α = 1) | **grokking**, t_ε 23k / 8.1k / 2.4k: ∝ 1/γ. x/y 23k / 8.1k / 2.4k, x\*y 24k / 8.1k / 2.4k | ✅ Phase 2b/2c (27k / 9.2k / 2.6k), Omnigrok |
| SGD and SGD+momentum | the same law in λ = lr_eff·γ: λ = 1e-4 → 26k (SGD) / 33k (momentum, λ = 9e-5); λ = 3e-4 → 10.2k / 10.7k | ✅ Phase 2b |
| larger α | slower: AdamW γ = 0.1, f 0.5: t_ε 23k / 28k / 41k at α = 1 / 2 / 4 | ✅ the LU picture, Phase 2c |
| smaller train fraction | slower, and past the horizon at the smallest decay: AdamW γ = 0.3: 49k / 8k / 3k at f 0.3 / 0.5 / 0.7. At f 0.3 the slowest decay of each optimizer (AdamW 0.1, SGD 1e-4, momentum 3e-5) is still rising at 1e5 (max test 0.07–0.54, seeds split between grokker and never) | ✅ critical data size (Power et al., Omnigrok) |
| γ = 0, every optimizer | **never** generalizes (test ≈ 0.01, i.e. chance at p = 97), except a slow creep at f 0.7 (Adam: 0.36, SGD: 0.28 at α = 1) | ✅ Phase 2b |
| Adam+L2, γ > 0 | **fast** (t_ε 0.35–1.7k) | ✅ Phase 2b |
| coupled γ = 1e-3, f 0.7, α = 1 (Adam+L2 and SGD) | never fits the training set (train ≈ test ≈ 0.01): the decay wins from the start, the Type B attractor of METRICS.md §5. α ≥ 2 and smaller fractions fit fine | new, a plausible edge of the 2b result (Adam+L2 cannot fit at 1e-2) |
| plain SGD, lr 1, γ = 1e-3 | fits and generalizes (max test ≈ 1), then **train and test accuracy oscillate violently** for the rest of the run (final train 0.06–0.99): an edge-of-stability regime. These are 20 of the 22 "late collapse" runs | not in the pilot; a real trap (it is Type B in the sense that train accuracy falls too), but the final values depend on where the oscillation is at step 1e5 |
| AdamW γ ≥ 0.1, α ≥ 2 | sharp test-accuracy spikes after grokking (slingshot-like) | ✅ Thilak et al. (slingshot) |

Provisional categories: grokker 349, never 243, fast 206, late collapse 22, joint
plateau 11, no fit 1. Grokking time spans 2.0 decades (x\*y), 2.5 (x+y) and 2.2 (x/y).
33 runs end with train accuracy < 0.99: 28 SGD at γ = 1e-3, 4 Adam+L2 at γ = 1e-3, and
one AdamW at γ = 0.1 caught in a spike.

**Notes.**
- **Many "never" runs (243, 29%).** They are γ = 0 for every optimizer (208 runs; that
  level is in the grid on purpose as the no-grokking control), plus the slowest-decay
  cells at f 0.3 and the coupled γ = 1e-3, f 0.7, α = 1 no-fit cells.
- **Duplicates:** the 52 γ = 0 AdamW/Adam+L2 pairs are the same trajectory, so it is
  the same run counted twice. Phase 5 aggregates should count each such pair once, or
  report that they are duplicates (MNIST and teacher–student have the same pairs).
- **Labelling:** the 2 AdamW "late collapses" (γ = 0.1, f 0.3) are grokkers still
  rising at the horizon, where a spike puts the max at step 99.75k above the final
  value. Same provisional-category caveat as MNIST; regret does not depend on it.
- **Absence:** there is no Type A (norm-growth) late collapse on modular, unlike MNIST.
  The cross-entropy softmax-collapse version (Prieto et al.) does not show up within
  1e5 steps at these learning rates.

**Verdict: as expected.** G3 passes, every pilot prediction holds (1/λ law, α and data
size effects, γ = 0 never generalizing), and the corpus adds two traps the pilot did not
show: the no-fit decay attractor and the oscillating SGD regime.

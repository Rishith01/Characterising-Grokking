# Phase 5 plan: compute the new metrics on modular addition and MNIST

**Goal.** Compute every metric in [new_metrics.md](new_metrics.md) on every run in scope, on the common grid, then the per-run descriptors. Insight first; stopping rules later, as a separate step.

**Status.** Plan only; no code yet. The time estimates come from scaling earlier measured runs, not from benchmarks. Benchmarking is milestone M3.

---

## 0. Scope (decided 2026-11-21)

**In scope: 832 runs.**

| Variant | Runs | Settings × 4 seeds | Network | Training data |
|---|---|---|---|---|
| Modular addition, x+y mod 97 | 576 | 144 | MLP 194-512-512-97, full batch, cross-entropy | 30%, 50% or 70% of the 9,409 pairs |
| MNIST, 3-layer | 128 | 32 | MLP 784-200-200-10, batch 200, MSE | 1,000 images |
| MNIST, 8-layer | 128 | 32 | MLP 784-400×7-10, batch 200, MSE | 4,000 images |

- **Out of scope:** modular x×y and x÷y (256 runs) and teacher–student (576 runs).
- **Every run in scope replays at L0** (bit-identical) according to the Phase 3 manifest.
- **Duplicates.** At γ = 0, AdamW and Adam+L2 are the same algorithm, so each γ = 0 setting of the two optimizers is the same trajectory recorded twice:
  - x+y: 36 runs are copies;
  - MNIST: 8 copies on each network.
  - That leaves **780 distinct runs**. Analysis counts each pair once. The replay can skip the copies.

### 0.1 What kinds of runs we will see

These are the Phase 3 provisional labels, from `phase3/{modular,mnist}/reports/corpus_runs.json` on HF. They use the same rules as the earlier reports, applied in this order:
1. **never:** best test accuracy below 50%;
2. **late collapse:** final test accuracy more than 10 points below its best;
3. **no fit:** training accuracy never reaches 99%;
4. **grokker:** t_ε at least 10× the step at which the training set is fit, and at least 1,000 steps later;
5. **joint plateau:** fitting takes at least 2,000 steps and t_ε is within 2× of it;
6. **fast:** everything else.

**All runs in scope, by outcome:**

| | Fast | Grokker | Joint plateau | Late collapse | Never | No fit | Total |
|---|---|---|---|---|---|---|---|
| **x+y** | 144 | **230** | 8 | 16 | 178 | 0 | 576 |
| **MNIST 3-layer** | 35 | **42** | 23 | 16 | 0 | 12 | 128 |
| **MNIST 8-layer** | 72 | 2 | 15 | 9 | 15 | 15 | 128 |
| **Total** | 251 | **274** | 46 | 41 | 193 | 27 | 832 |

**By optimizer:**

| | Optimizer | Fast | Grokker | Joint plateau | Late collapse | Never | No fit |
|---|---|---|---|---|---|---|---|
| **x+y** | AdamW | 8 | 92 | 0 | 2 | 42 | 0 |
| | Adam+L2 | 88 | 16 | 0 | 0 | 40 | 0 |
| | SGD | 32 | 42 | 8 | 14 | 48 | 0 |
| | SGD+momentum | 16 | 80 | 0 | 0 | 48 | 0 |
| **MNIST 3-layer** | AdamW | 4 | 16 | 0 | 12 | 0 | 0 |
| | Adam+L2 | 15 | 13 | 0 | 4 | 0 | 0 |
| | SGD | 8 | 8 | 12 | 0 | 0 | 4 |
| | SGD+momentum | 8 | 5 | 11 | 0 | 0 | 8 |
| **MNIST 8-layer** | AdamW | 30 | 2 | 0 | 0 | 0 | 0 |
| | Adam+L2 | 24 | 0 | 4 | 0 | 4 | 0 |
| | SGD | 2 | 0 | 10 | 5 | 10 | 5 |
| | SGD+momentum | 16 | 0 | 1 | 4 | 1 | 10 |

**When grokkers finish learning** (t_ε as a share of the 100k-step budget):

| | ≤ 2% | 2–10% | 10–35% | 35–55% | > 55% | Median |
|---|---|---|---|---|---|---|
| x+y (230) | 16 | 59 | 93 | 50 | 12 | 15% |
| MNIST 3-layer (42) | 0 | 0 | 15 | 7 | 20 | 52% |
| MNIST 8-layer (2) | 0 | 0 | 2 | 0 | 0 | 26% |

Fast learners have a median t_ε of 1.0% (x+y), 2.4% (MNIST 3-layer) and 1.3% (MNIST 8-layer) of the budget.

**What the labels hide.** Read these before trusting a count:

- **x+y "never" (178) is mostly the no-decay control.**
  - 144 of the runs have γ = 0, every optimizer and every training fraction. They memorize and never generalize.
  - The other 34 are the slowest-decay settings at a training fraction of 30%, still rising at the end of the budget (26 runs), plus 8 runs at γ = 1e-3 with 70% data that never fit the training set.
- **x+y "late collapse" (16) are not collapses.**
  - 14 are plain SGD at γ = 1e-3, which generalize and then oscillate violently (edge of stability).
  - 2 are AdamW grokkers still rising at the end, where a spike sets the maximum.
- **MNIST 8-layer "fast" (72) includes about 26 mislabelled runs.** These are Adam runs at α = 8: train and test are flat together until about step 2k, jump together, then creep up until 8–25k. That is a joint plateau followed by a creep, but the label rule calls it fast. Phase 3 already flagged this.
- **MNIST grokking is a slow creep** (to about 91%), not a jump from chance to 100% as on modular. The MNIST 8-layer network has almost no grokkers (2).
- **Within a setting, seeds agree closely.** On x+y the median s.d. of best test accuracy across seeds is 0.000. So 576 runs carry about 144 independent behaviours, not 576. MNIST seeds spread more, but only in the slow 8-layer SGD settings.

**Where the grokkers are.** x+y holds 230 of the 274. They come from all four optimizers, mostly AdamW (92) and SGD+momentum (80), with t_ε spread over 2.5 decades. That spread is what the timing analyses need. The no-decay control (144 runs) is the matching "never generalizes" group.

### 0.2 Time for this scope

These are scaled from the measured F2 replay times on Molab (per run, with 14 workers sharing the machine): x+y 837 s, MNIST 3-layer 591 s, MNIST 8-layer 1,527 s.

**Total work:** x+y about 134 Molab-worker-hours, MNIST 3-layer about 21, MNIST 8-layer about 54. That is **about 210 worker-hours**, or about 195 with the 52 duplicate runs skipped.

| | Molab | Kaggle (2×T4) | Local (RTX 4060 Laptop) |
|---|---|---|---|
| No-replay stage (N, R.1, R.2, H.1, W.1, descriptors) | ~10 min | ~20–40 min | ~30–60 min (mostly downloading) |
| Replay: x+y (576 runs, GPU-bound) | ~8–10 h | ~20–40 h (2 T4s at an assumed 3–6 TFLOPS achieved each) | ~22–45 h (at an assumed 5–10 TFLOPS achieved) |
| Replay: MNIST (256 runs, CPU-bound) | ~4–5 h | about 75 worker-h. Mostly done on the 2 spare cores while x+y runs; ~3–15 h extra afterwards | ~8–13 h, alongside x+y |
| **Replay total (wall time)** | **~12–15 h: one 12-h session plus a short second one** (about 8–10 h with MPS) | **~25–45 h: 3–4 sessions, 1–1.5 weeks of free quota** | **~30–45 h: about 1.5–2 days non-stop**, plus 20–30% if it throttles |
| Download | ~55 GB, fast | ~55 GB, download and delete per run | ~55 GB: about 2.5 h at 50 Mbit/s |
| Replay reproduces the run? | yes, bit-identical (L0) | approximately at best (L1); untested | approximately at best (L1); untested |

- **Molab is about ±30%** (F2 measured these exact variants).
- **Kaggle and local could be off by 2×.** They rest on unmeasured GPU and CPU speeds, and the Kaggle hardware limits are from memory (§4.2).
- **This roughly halves the full-corpus estimate**, and removes teacher–student, which was the least certain part of it.

---

## 1. What has to be computed, and from what data

Every metric needs one of two kinds of data:
- **Recorded:** data already stored with each run.
- **Replay:** data that needs the weights at grid points, so the run must be re-trained from a keyframe.

The common grid (new_metrics.md §0) is all multiples of 500 steps, plus about 30 log-spaced points before step 5,000, snapped to multiples of 50. That gives about 230 points for a 100k-step run (modular, MNIST) and about 630 for a 300k-step teacher–student run.

| Metric | Recorded data (no replay) | Needs replay |
|---|---|---|
| N.1, N.2 (log‖w‖, per layer) | telemetry, every 50 steps | – |
| N.3 ρ̄\*, W.1 | telemetry (SGD, SGD+m, AdamW) | Adam+L2 only: Σ w²/(√v̂+ε) at each log step |
| R.1 activation erank | summaries `actspec/l{i}`, every 500 steps | early grid points (<500), forward pass only |
| R.2 weight erank, H.1 HTSR α | summaries `sv/l{i}`, every 500 steps | early grid points (<500) |
| AP.1, AO.1, AO.2, R.3 (AGOP) | – | every grid point: d_out backward passes + one eigendecomposition |
| S.1, S.2, W.2 (gradient statistics) | – | every grid point: per-example gradient sums |
| Outcome descriptors (t_fit, t_ε, best test acc) | oracle curve (`oracle.parquet`; evaluation only) | – |

Two conclusions follow.

1. **About half the metrics need no GPU at all.** These are N.1, N.2, N.3 (except Adam+L2), W.1, R.1, R.2 and H.1, plus the descriptors. Study A computed the same kind of data for 320 runs in 2.4 minutes. This part can run immediately, on any machine.
2. **The expensive part is one replay pass per run.** It computes AGOP, the gradient statistics, the early grid points and the Adam+L2 state *together*, so each run is replayed exactly once.

---

## 2. Code structure

### 2.1 Principle: extract raw data once, define metrics as cheap functions afterwards

The old studies mixed three things in one place: computing signals, turning them into stopping rules, and scoring the rules. Phase 5 splits the work into three stages with clear boundaries:

```
  Stage E: EXTRACT          Stage Q: QUANTITIES          Stage D: DESCRIPTORS
  (expensive, run once)     (cheap, re-runnable)         (cheap, re-runnable)

  recorded files ──┐
                   ├──► raw/<run>.npz ──► curves/<run>.parquet ──► descriptors.parquet
  replay (GPU) ────┘    (versioned)       (every metric ID          (1 row per run,
                                           on the common grid)       ~50 columns)
```

- **Stage E stores raw material, not metrics.** For example, it stores the matrix of pairwise distances between the AGOP feature matrices at all grid points, not "AO.2 with c = 0.05". Then every angle quantity, every lag and every variant can be computed later without replaying.
- **Stage Q holds the metric definitions.** Each metric ID in new_metrics.md is one pure function: raw arrays in, curve on the grid out. It does no I/O. Changing ε, c, the smoother or a definition means re-running Stage Q (minutes on a CPU), never the replay.
- **Stage D turns curves into the per-run table** that the analysis uses.

### 2.2 Package layout

A new package, `test_free_stopping/trajectory/`. The old `stopping/` package stays untouched, so the earlier reports remain reproducible.

**As built (2026-11-21), for weight norm, rank and AGOP (offline and online).** It differs from the planned layout below in these ways:

```
trajectory/
  grid.py  corpus.py  layout.py  rawfile.py
  extract/recorded.py            telemetry + summaries -> raw/recorded/<run_id>.npz
  extract/probes.py              AgopProbe, EarlySpectraProbe, AdamDecayProbe (one module, not a folder)
  extract/replay.py              one verified replay pass        -> raw/replay/<run_id>.npz
  quantities/{norm,rank,agop}.py N1-N3, R1-R3, AP1, AO1, AO2 on the grid
  runner.py  worker.py           queue, worker processes, upload, resume, build_curves
  validate.py                    pilot check against F2's series.parquet
molab/phase5_extract.py          buttons: recorded, replay, curves, check against F2; pilot option
tests/test_trajectory.py
```

- **No `cli.py` and no `reduce.py`.** The notebook calls `runner` directly, and each probe assembles its own arrays.
- **Not built yet:** the common smoother, Stage D descriptors, plots, HTSR (H.1), S1b (S.1, S.2) and the AdamW test (W.1, W.2).
- **Curves are the raw values on the grid,** with no smoothing applied.

```
trajectory/
  grid.py            common grid for a run (pure function of T and the log/summary schedule)
  corpus.py          list of runs from the HF manifest, metadata table, removal of the duplicate setting
  layout.py          output paths (raw/, curves/, descriptors) locally and on HF; one source of truth
  extract/
    recorded.py      Stage E from telemetry + summaries (no replay)
    replay.py        Stage E replay pass: restore at each keyframe, call probes at grid/log steps
    probes/          one small class per kind of measurement made on the live model
      agop.py          G at each grid point (kept in host RAM); eigenvalues of G per point (R.3);
                       chord² distance matrices between all pairs of grid points, for M̂ and Ĝ
      gradstats.py     per-layer Σ_k (ḡ+γw)²/σ̂², Σ_k ḡ²/σ̂², AdamW-null statistic, coordinate counts
      activations.py   activation covariance eigenvalues at early grid points
      weights.py       singular values at early grid points
      adam_state.py    Adam+L2: decay pull Σ_j w_j²/(√v̂_j+ε) at each log step (N.3);
                       AdamW: m̂ and v̂ statistics at grid points (W.2)
    reduce.py        assembles one run's raw file and writes it atomically
  quantities/        Stage Q; one module per family, functions named by metric ID
    norm.py          N1, N2, N3
    rank.py          R1, R2, R3
    agop.py          AP1, AO1, AO2
    htsr.py          H1
    gradstats.py     S1, S2, W2   (B = minibatch, n_eff or n; the two corrections)
    adamw.py         W1           (equivalence test, batch means)
    smooth.py        the single symmetric smoother
  descriptors.py     Stage D: value at start / t_fit / t_ε / T, t50, t90, still-moving flag
  runner.py          work queue, claiming, resuming, uploading; the same code on every platform
  cli.py             python -m trajectory {recorded|replay|quantities|descriptors} --workers N --shard i/k
molab/phase5_*.py    thin marimo notebooks that call cli
kaggle/phase5.ipynb  thin wrapper (only if Kaggle is used)
tests/test_trajectory_*.py
```

### 2.3 Design rules

- **Probes are independent.** A probe declares what it needs (grid steps or log steps; model, optimizer state, batch). The replay engine calls every registered probe at each requested step. Adding a metric later means adding a probe, without touching the replay loop. This is the same idea as the existing `Signal.requires`, minus the streaming and stopping logic.
- **Reuse, don't rewrite, the low-level pieces:**
  - `tfs.training.replay.ReplayEngine`, which already restores from any keyframe and verifies;
  - `RunReader` and `tfs.storage.hub`;
  - `stopping.signals._linalg`: the vmapped `input_agop`, erank and the matrix square root;
  - the claim / resume / kill-stale worker pattern in `stopping/agop_final.py`, moved into `runner.py`.
- **The unit of work is one run.**
  - Inside a run, replay restarts from each 10k-step keyframe in turn (the restart is free; `iterate` already supports it). On non-native hardware this limits drift to at most 10k steps; see §3.
  - Output is one raw file per run, uploaded as soon as the run finishes. That makes the job safe to kill at any point and resumable on any platform.
- **Everything is versioned.**
  - Each raw file records the probe versions, the code git hash, the platform, the GPU and the replay level (L0/L1).
  - Stage Q output records the quantity versions.
  - Stale files are recomputed, never silently mixed with new ones.
- **Gradient statistics without per-example gradients in memory.** For a linear layer the per-example gradient is an outer product, so Σᵢ gᵢ² = (a²)ᵀ(δ²). That is one extra matrix product per layer instead of n gradients (Goodfellow's trick). S.1, S.2 and W.2 then cost about one extra backward pass per grid point.
- **Store sums, apply corrections later.** The probe stores Σ_k ḡ_k²/σ̂_k² and the like, computed with the plain n. Stage Q then applies B = 200, n_eff or n and the small-sample correction. Changing a correction does not need a replay.
- **Causality is tested, not assumed.** For AO.1, AO.2 and W.1, a test truncates a run at step t and checks that every value at steps ≤ t is unchanged. Old `tests/` already contain this pattern.
- **Hindsight is explicit.** Quantities that use the end of the run (AP.1, t_ε, outcome labels) live in clearly named functions. Only Stage D and the analysis read `oracle.parquet`, the same quarantine as now.

### 2.4 Tests and validation, before any large run

1. **Unit tests with known answers.** Examples:
   - ρ̄\* on a synthetic pure-decay trajectory is 1;
   - the angle between diag(1,0) and diag(0,1) is π/2;
   - EB on simulated pure noise is about 0, and about 0 with the t correction at n = 50;
   - the equivalence test reproduces the table in new_metrics.md §7.3.
2. **Regression against earlier results.** The 224 F2 runs and the 320 study-A runs are a ready-made check:
   - AP.1 and the AO.2 angle at c = 0.02 must match the AGOP alignment and F2-C columns of `phase4/agop_final/series.parquet`;
   - N.1 and R.1 must match study A's traces.
3. **End-to-end on 2 tiny synthetic runs**, CPU only, in `pytest`. Existing fixtures in `tests/conftest.py` build such runs.

---

## 3. A constraint that decides the platform: replay must reproduce the run

The corpus was trained on Molab's RTX PRO 6000 Blackwell with a pinned stack (torch 2.11.0+cu130, driver 595.71). Every replay so far on that setup was **L0: bit-identical**.

**On any other GPU** (Kaggle's T4, the local RTX 4060), matrix kernels differ, so replay is **not bit-identical**.
- Tiny rounding differences can grow over many steps, especially in grokking runs, which are unstable around the jump.
- `ReplayEngine` refuses a different environment unless `allow_env_mismatch=True`. The run must then pass **L1**: keyframe weights within relative error 1e-6, train predictions agreeing on 99.9% of examples.
- Restarting from every 10k-step keyframe (§2.3) limits drift to 10k steps. Whether that is enough is unknown until it is measured.

**Two more risks off Molab:**
- **Kaggle:** its NVIDIA driver must support CUDA 13 to run the pinned torch build. If it doesn't, a cu12 build is needed: a further mismatch.
- **Local:** Windows rather than Linux. The driver (610.91) supports CUDA 13, but the local venv has CPU-only torch, so a CUDA build must be installed.

**Consequence.** Molab is the only platform where replay is known to reproduce the runs. Kaggle and local are usable only if a pilot (milestone M3) shows that nearly all runs pass L1. Any run that fails L1 is replayed again on Molab.

The no-replay stage has none of these problems and runs anywhere.

---

## 4. Time estimates (full 1,664-run corpus; kept for reference; the in-scope numbers are in §0.2)

### 4.1 What the estimates rest on

**Measured: the F2 replay pass on Molab** (`report_AGOP/detailed/data/status.json`).
- It ran 14 replays at once, with verification on.
- It computed the AGOP at 223 points per run and the angles between every pair of points.
- The times below are **seconds per run for one worker while 13 others shared the GPU and CPUs**. They are not solo times.

| Variant | Runs timed | Median seconds per run (1 of 14 workers) | Training time per run (16 workers, Phase 2) | Replay ÷ training |
|---|---|---|---|---|
| Modular x+y / x×y / x÷y | 46 / 8 / 11 | 837 / 944 / 1,045 | 355 / 437 / 363 | ~2.4 |
| MNIST 3-layer | 20 | 591 | 156 | ~3.8 |
| MNIST 8-layer | 25 | 1,527 | 442 | ~3.5 |
| Teacher–student | not replayed by F2 | **estimate ~1,350** | 384 | ~3.5 (assumed) |

**Why replay is 2.4–3.8× slower than training.**
- Replay re-checks the recorded telemetry, train predictions and fingerprints at every 50-step log step.
- It also runs the probes at each grid point.

**Teacher–student is pure estimate.** It has 300k steps, so about 6,000 log steps to check and about 630 grid points. MNIST's ratio is assumed.

**The new pass is about as heavy per run as F2.**
- It keeps the pairwise distance matrices for M̂ and Ĝ, as F2 did, but drops F2's per-layer weight directions and their pairwise angles.
- It adds the gradient statistics (about one extra backward pass per grid point, via the §2.3 trick) and the early-step probes.
- Estimate: ±20% of F2.

**Total work, in Molab-worker-hours** (each worker sharing the machine with 13 others):

| | Runs | Seconds per run | Molab-worker-hours | What limits the speed |
|---|---|---|---|---|
| Modular | 832 | ~880 | ~205 | GPU arithmetic: about 1.3 PFLOP per run |
| MNIST 3-layer | 128 | ~590 | ~21 | per-step overhead (CPU) |
| MNIST 8-layer | 128 | ~1,530 | ~54 | per-step overhead (CPU), plus some arithmetic |
| Teacher–student | 576 | ~1,350 (estimated) | ~215 | per-step overhead (CPU): 300k tiny steps |
| **Total** | **1,664** | | **~500 (±150)** | |

**How the other platforms are scaled.** The two kinds of runs scale differently.
- **Modular is GPU-bound.**
  - Its full batch of about 4,700 examples through 412k parameters is about 11.6 GFLOP per step. With the AGOP, that is about 1.3 PFLOP per run.
  - The F2 pass achieved about 22 TFLOPS in total on Molab. So modular time scales with the arithmetic speed each GPU actually achieves.
- **MNIST and teacher–student are CPU-bound.**
  - Each step is a tiny GPU kernel, so speed is set by how fast one CPU core can drive the step loop, and by how many processes run at once.
  - Every process, including a modular one, needs its own core.
  - Assumption: one uncontended process on Kaggle or on the laptop is 0.7–1.5× as fast as one contended Molab worker. This is not measured.

### 4.2 Estimates per platform

| | Molab | Kaggle (2×T4) | Local (RTX 4060 Laptop) |
|---|---|---|---|
| Hardware | RTX PRO 6000 Blackwell 96 GB, 20 CPUs, 12-h sessions (RAMPUP §7) | 2× T4 16 GB; **from memory, not checked:** 4 vCPUs, about 29 GB RAM, about 70 GB scratch disk, 12-h sessions, about 30 GPU-h/week free | RTX 4060 Laptop 8 GB, 16 CPU threads (checked), no session limit |
| Processes at once | 14–18 | at most 4 (one per vCPU): 2 drive modular on the GPUs, 2 run CPU-bound jobs | ~2 for modular + ~6 CPU-bound (8 GB of GPU memory and laptop heat limit it) |
| **No-replay stage, all 1,664 runs** | **~15 min** | ~30–60 min | ~1–2 h (mostly downloading telemetry and summaries) |
| **Replay: modular** | ~205 worker-h ÷ 14–18 ≈ 12–15 h | ~34 h (2 T4s at an assumed 4–5 TFLOPS achieved each) | ~45 h (at an assumed ~7 TFLOPS achieved) |
| **Replay: MNIST + teacher–student** | ~290 worker-h ÷ 14–18 ≈ 16–21 h | ~290 worker-h at 0.7–1.3× per process. About 70 h done on the 2 spare cores while modular runs, the rest on all 4 cores afterwards: **+40–80 h after modular finishes** | ~290 worker-h ÷ (6 processes × 1.0–1.5×) ≈ 30–50 h, **running alongside modular** |
| **Replay total (wall time)** | **~25–35 h: 3 Molab sessions.** MPS (2× in Phase 2, still bit-identical) or thinner checking could bring it to ~15–20 h | **~75–115 h: 7–10 sessions, about 3–4 weeks of free quota** | **~45–60 h: 2–3 days running non-stop**, plus 20–30% if the laptop throttles |
| Download of the corpus (~77 GB) | fast | fast; download and delete one run at a time | ~3–4 h at 50 Mbit/s; needs ~77 GB free, or delete per run |
| Replay verification | **L0, bit-identical** | L1 at best; pass rate unknown | L1 at best; pass rate unknown |
| Stages Q and D | minutes | minutes | minutes |
| Main risks | none known; it is the native platform | driver may not support CUDA 13; drift fails L1; weekly quota; only 4 cores | drift fails L1; setting up CUDA torch on Windows; heat over 2–3 days |

**These are planning numbers.**
- The Molab column is about ±30%: modular, MNIST and the time ratios are measured, and only teacher–student is estimated.
- The Kaggle and local columns rest on unmeasured assumptions (achieved TFLOPS, per-process step rate) and could be off by 2×.
- Milestone M3 replaces all of them with measured numbers.

### 4.3 Ways to cut the replay cost, if needed

- **Check less often during replay.**
  - Replay currently re-checks everything at every 50-step log step. That matters most for teacher–student, with about 6,000 checks per run.
  - Checking at keyframes plus every 10th log step keeps the guarantee that the run is reproduced, at a fraction of the cost.
  - This is likely the biggest single saving for the CPU-bound runs. It needs measuring in M3.
- **MPS on Molab.** It doubled training throughput in Phase 2 and stayed bit-identical.
- **AGOP on a stratified subset of modular cells**, if AGOP scope is cut (the decision from new_metrics.md §10). The modular replay itself is still needed for S.1, S.2 and W.2, so this saves only the AGOP share, roughly a third of modular's time.
- **One seed per setting** for a first look cuts every replay number by 4×. It is a preview only, since seed spread is itself a descriptor.

---

## 5. Milestones

| # | Milestone | Where | Done when |
|---|---|---|---|
| M0 | Scaffold the `trajectory/` package: grid, corpus table, layout, runner, CLI; unit tests | local, CPU | `pytest` green; the runner resumes a killed fake job |
| M1 | **No-replay stage on all 1,664 runs:** N.1–N.3 (not Adam+L2), W.1, R.1 (≥500), R.2, H.1; Stages Q and D for these; first look at H1 (partly), H3, H7 | Molab (15 min) or local | descriptors table for all runs; N.1 and R.1 match study A on its 320 runs |
| M2 | Replay probes (AGOP, gradient statistics, early points, Adam state) on CPU fixtures; GPU check on 10 runs | local, then Molab | AP.1 and the AO angle match F2's `series.parquet` on the same runs |
| M3 | **Pilot and benchmark: 12 runs** (4 modular, 4 MNIST, 4 teacher–student) on each platform you want to use | each candidate platform | measured seconds per run; L1 pass rate off Molab; §4 updated with real numbers |
| M4 | Full replay pass | chosen platform(s); runs failing L1 go to Molab | a raw file on HF for every run, each with its replay level |
| M5 | Stages Q and D for every metric; test H1–H8 (old_results_summary.md §6) | anywhere, CPU | descriptors table complete; first insight report |

M1 needs no GPU and gives results within a day of M0. It is worth doing before any replay work.

---

## 6. Decisions for you

1. **Platform for the replay.** Recommendation: **Molab** (about 3 sessions, bit-identical). Kaggle or local only as overflow, and only if the M3 pilot shows they pass L1.
2. **AGOP scope.** All 1,664 runs (about 25–35 h of replay on Molab), or a stratified subset of modular cells (saves roughly a third of modular's ~12–15 h; the replay is still needed for the gradient statistics).
3. **Package name and location.** Proposed: `test_free_stopping/trajectory/`, a new package, leaving `stopping/` frozen. The alternative is extending `stopping/`, which keeps one package but tangles the old rule-scoring code with the new pipeline.
4. **Kaggle credentials, if Kaggle is used.** Your rule is that the HF token is typed into a password field in each notebook and never stored. On Kaggle that means `getpass` in the notebook, not Kaggle Secrets, which would store it.

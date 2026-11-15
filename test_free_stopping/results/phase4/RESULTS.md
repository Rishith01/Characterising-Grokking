# Phase 4 results

Contents: the MNIST signal pilot (below), a toy check of online AGOP estimators
([section](#agop-online-estimators-toy-check-2026-11-04)), and the F2 experiment on
corpus runs ([last section](#f2-experiment-remaining-angle-bound-on-61-corpus-runs-2026-11-15)).

## Pilot: MNIST, 18 runs (2026-11-04)

**Run.** `molab/phase4_pilot.py` on Molab (cuda), with the replay signals turned on and
200 eval steps per run. It took 61 min. The outputs are on HF at `advay333/tfs`,
`phase4/pilot/mnist/`: `plots/`, `stops.csv`, `signals/` (packed traces), `status.json`
and `summary.md`. The runs are 3 seed-0 runs per provisional category: grokker,
joint_plateau, fast, late_collapse, never and no_fit, spread over variant and optimizer.

**Verdict: the signals are computed correctly.** The stop table comes from
**uncalibrated placeholder rules** (`stopping/methods.py`), so it is a sanity check, not a
score.

### Integrity checks (all pass)

| Check | Result |
|---|---|
| Runs computed | 18/18 ok, 11 signals each, 118–355 s per run (depth 8 is the slowest) |
| Rows per cadence | log 2,001 (B0, B1, N1); B4 2,000 (from step 50); summary 201 (R1, B3 summary form, B5 α); eval 153 (B2, B3 replay form, traps; the 200 requested steps minus duplicates where the log-spaced points snap to the same step) |
| NaN | none in any trace. The B3 summary form is defined, as expected, since n_train ≥ 1,000 > the 256-example probe batch |
| Step-0 values across runs | identical for runs with the same depth and α, whatever the optimizer, as they should be (same init) |
| Init scaling | ‖w‖ at step 0 is exactly α × 11.68 at depth 3 (α = 1, 4, 8 → 11.68, 46.73, 93.45) and α × 30.65 at depth 8 |
| HTSR α at step 0 | 11.55 for every depth-3 run, whatever α. Correct: a power-law exponent does not depend on scale |
| B3 replay vs summary form | Theory gives E‖g₁−g₂‖² = 2 trΣ/B for two disjoint batches of the training size B = 200. The stored value comes from 256-sample probe batches, with the finite-population factor and the biased 1/K estimate (K = 8). Predicted ratio 1.40 (n = 1,000) and 1.25 (n = 4,000); measured medians **1.40 and 1.21** |

### Signals against METRICS.md's predictions

| Prediction (METRICS.md) | Seen | Verdict |
|---|---|---|
| N1: the norm shrinks on grokkers as test accuracy rises (Omnigrok) | depth-3 α = 8: 93 → 8 (Adam+L2) and 93 → 32 (AdamW); the test rise coincides with the fall | ✅ |
| Late collapse Type A: the norm grows and train accuracy stays at 1 | Adam+L2 γ = 0: ‖w‖ 12 → 50, train accuracy 1.0, test 0.92 → 0.67, validation loss rises; AdamW γ = 0.01: 12 → 35 | ✅ |
| Late collapse Type B: the norm collapses toward 0 at depth 8 (constant output) | Adam+L2 depth 8 α = 1 γ = 1e-3: ‖w‖ 31 → 0.32, test 0.11; SGD depth 8 γ = 0.1: 123 → 45, test peaks at 0.55 then falls | ✅ |
| R1: erank falls together with generalization (Deep Grokking) | AdamW grokker: erank rises to 80 during memorization, then falls to 24 exactly as test accuracy rises. Depth-3 runs end at erank 10–24 (from 60) | ✅ (and the rise during memorization is a new detail) |
| B1, B4 stop on the plateau | B4 stops at about 2k on all 3 grokkers (test 0.24 / 0.28 / 0.77) and is the most premature method (13/18 runs). B1 stops at 26–32k, near or just before t_ε | ✅ B4 clearly; B1 only mildly |
| B4/B1 cannot see Type A | train accuracy and label changes are flat at 1 / 0 throughout | ✅ |
| B5: HTSR α heads toward 2 as a layer trains; α < 2 on anti-grokking | depth 3: α falls from 11.5 to 1.2–2.2; the Type A run ends at 1.78. Depth-8 SGD at lr 1e-4 stays at 13.7 (weights barely move) | ✅ |
| **B2 (Mahsereci EB) stops on the plateau** | **Not on Adam-family runs:** EB stays at −0.5 to −17 throughout and never crosses 0 (8/9 Adam runs never stop). On SGD runs EB sits near 0 and does stop, early (e.g. step 1,350 at test 0.70 on the SGD grokker) | ⚠️ partly wrong (below) |

**Why B2 behaves differently from the prediction.** With weight decay the optimizer
settles where the data gradient balances the decay, not where g = 0. The mean data
gradient there is coherent (≈ −γw for coupled decay; for Adam, the preconditioner
amplifies it per coordinate), so it does not look like noise to the test. On the Adam
runs with γ = 0, the loss keeps falling and the norm keeps growing (Type A), so the
gradient is not noise either. Under SGD at lr 0.01–0.03, the minibatch noise swamps both
effects, so EB ≈ 0. This is the argument for S1b (testing ∇J = g + γw instead of g), now
seen in data. METRICS.md's B2 entry should say "stops on the plateau under SGD; on
Adam-family runs it tends not to stop at all".

### Other observations

- **Correlation traps appear early and on healthy runs.** On depth 3 there are 1–6
  traps, often from step 250. On depth 8 Adam+L2 there are 20–44 per layer, including
  the α = 8 run that holds test accuracy at 0.95. So "traps present" is not by itself a
  collapse signal here, and the "α or traps" variant (stop when traps appear) stops the
  AdamW grokker early (17.5k, test 0.83). Treat traps as a feature for Phase 5, not as a
  stop trigger. The count uses the theoretical MP edge from the randomized matrix's
  variance; WeightWatcher fits the bulk instead, which may count fewer.
- **B6 never stops on 8/18 runs**: validation loss keeps creeping down. This is the
  placeholder rule (plateau with threshold 0), not the signal.
- Stop-table statistics under the placeholder rules, for orientation only: median
  regret ≤ 0.01 for most methods, then R1 0.019, B4 0.065, B2 0.139. B5 α and B6 never
  stop on 8/18 runs, and B2 on 10/18.

### Fixed after the pilot

- `stopping/pilot.py` `stops_markdown` cut run ids to 10 characters, so every row read
  "mnist__n10". It now shows the run's config part. Only the stop table in `summary.md`
  was affected; `stops.csv` has the full ids.

## AGOP online estimators: toy check (2026-11-04)

Not a corpus run. [`agop_toy_sim.py`](agop_toy_sim.py) (numpy, ~1 min) compares two ways
of estimating AGOP alignment online on synthetic trace-normalized AGOP paths
(d = 30, full rank → rank 3, 10 seeds), where the true alignment is known:
- the team's online-alignment proposal (`BTP_Proof_AGOP_online_variant.pdf`, chord model
  + Aitken), bare and gated by its D_chord test;
- the remaining-angle bound proposed for F2.

Premature stops (true alignment < 0.95 at the stop), out of 10:

| Path | Proposal, bare | Proposal + D_chord gate | Remaining-angle bound |
|---|---|---|---|
| exponential (the proposal's own model), 5% noise | 2 (at s = 0.947) | 2 (at s = 0.947) | 0 |
| sigmoid (grokking-like), clean | 0 | 0 | 0 |
| sigmoid, off-chord detour | **9** (s down to 0.51) | 0 | 0 |
| sigmoid, 5% noise | **10** (s ≈ 0.15) | 0 | 0 |
| sigmoid, 15% noise | **10** (s ≈ 0.15) | 0 | 0 |

**Reading.** The bare proposal stops on the plateau: its three-point extrapolation is
arbitrary when the signal is flat plus noise. With its own gate, it is safe in the toy.
The bound is never premature, and stops 1.4k–5k steps after the true crossing.
Full discussion: [AGOP.md](../../AGOP.md) §3–§5. Real AGOP traces are needed before
choosing (AGOP.md §6).

## F2 experiment: remaining-angle bound on 61 corpus runs (2026-11-15)

**Run.** `molab/phase4_agop_study.py` (DESIGN.md §13.9), cuda, 227 min, 61/61 runs ok:
MNIST 26, modular x+y 17, teacher–student 18 (28 grokkers, 18 fast, 10 late collapse,
5 never; seed 0 plus seeds 1–2 of six grokkers). ~300 eval steps per run. Outputs on HF
at `advay333/tfs`, `phase4/agop_study/` (traces, curves, plots, `settings_summary.csv`,
`questions.csv`, `summary.md`). 324 estimator settings scored offline per run.

**Verdict: the remaining-angle bound fails as a stopping rule.** No setting is both rarely
premature and stopping in most runs, and none beats simply stopping at the horizon.

| | premature | never stops | mean regret (never = horizon) |
|---|---|---|---|
| Stop at the horizon (B0 at T) | 0 | – | **0.042** (median 0.002) |
| Best F2 setting by regret (W, lag 0.15, w 3, ε 0.03) | 0.08 | 0.82 | 0.050 |
| Best F2 settings that stop in ≥ 80% of runs | 0.43–0.62 | 0.12–0.20 | 0.23–0.26 |
| Default (M, lag 0.25, w 3, floor, ratio), ε 0.1, patience = ×4 in t, norm gate | 0.10 | 0.85 | 0.076 |

Lowering the premature rate always pushed the never-stop rate toward 80–90%. Longer
patience measured in log-time (R̂ < ε held over a factor 2 or 4 in t) and the norm gate
both trade premature for never-stop the same way.

**Answers to the questions.**
- **Q1 (do the features move on the plateau?): yes.** On grokkers a median 20% of the
  whole M path is travelled between memorization and t_ε, at a near-constant speed
  (median κ·t = 0.29). The premise of F2 holds, and on the plateau itself R̂ is large or
  ∞, as designed (MNIST AdamW α = 8: ω ≈ 5e-5 rad/step, flat from 2k to 30k steps).
- **Q3 (does R̂ dip before t_ε?): yes, almost always. This is the failure.** R̂ fell
  below 0.1 before t_ε in 42/54 runs (24/28 grokkers on the plateau). It is the
  two-timescale case predicted in THEORY.md §4.3: right after memorization the feature
  speed falls fast (κ·t ≈ 3.5 on MNIST), so the exponential extrapolation says "almost
  done" (R̂ ≈ 0.03 rad) while 1.5 rad of feature motion is still to come.
- **The norm gate does not catch it** because at the end of memorization the norm has
  not started shrinking yet (Omnigrok's shrinkage starts later, ~5k steps on MNIST α = 8),
  so the norm also looks settled.
- **Bound validity:** at the stop, R̂ was below the true remaining angle in 95% of stops.
  The exponential-tail assumption does not hold at the points where the rule fires.
- **Q2 (tail shape):** neither exponential nor power law is clearly better (coefficient of
  variation 0.46 for κ vs 0.44 for κ·t in the last 30% of runs).
- **Q4 (noise floor):** small on MNIST (late scatter/lagged angle 0.18), but large on the
  full-batch tasks: modular 0.90, teacher–student 0.59. That is Adam-family and
  edge-of-stability jitter (THEORY.md §4.2): on modular the features never settle to a
  point, so the true remaining angle stays at 0.3–0.5 rad to the end.
- **Q6 (G vs M):** no real difference; per-layer weight directions (W) were slightly
  better than either (fewer premature stops, more never-stops).
- **Q8 (chord model):** a median 86% of post-memorization rows have D_chord < 0.05, so
  the team's straight-line assumption is closer to true than expected on modular and
  teacher–student, but not on MNIST grokkers (D_chord 0.05–0.17).

**Caveat on the baseline.** Stopping at the horizon is hard to beat on this corpus: most
runs hold their best test accuracy to the end (median horizon regret 0.002). Only late
collapse (mean horizon regret 0.24) rewards stopping early, and that is where every
stationarity signal is late by construction (METRICS.md §5).

**What this means for F2.** The AGOP speed is a valid "not done yet" signal on the
plateau (Q1), but extrapolating it into a remaining distance is not safe on real runs:
the speed's history does not predict its future at the memorization–plateau handoff.
Options are in the discussion with the user (2026-11-15).

### Follow-up (2026-11-15): late-penalized score and settling tests, re-scored locally

Same 61 traces, no new replay (`analyze_all` on the downloaded traces; the notebook's
"analysis only" option reproduces it and uploads). Two changes:
- **Score** (PLAN.md §5, the user's decision): cost_λ = regret + λ·max(0, τ − t_ε)/T,
  never-stop scored at the horizon T.
- **Two settling tests instead of extrapolation** (AGOP.md §5b): the MSD exponent α of
  the feature path over a trailing window [t/f, t] (≈2 directed, ≈0 confined), and a
  Geweke-style Z (early vs late segment means, over their standard errors).

Mean over 61 runs (λ = 0.1 is the headline):

| | premature | never | regret | cost λ=0.03 | **cost λ=0.1** | cost λ=0.3 |
|---|---|---|---|---|---|---|
| Oracle stop at t_ε (floor) | 0 | 0 | 0.008 | 0.008 | **0.008** | 0.008 |
| Horizon (never stop) | 0 | 1 | 0.042 | 0.066 | **0.120** | 0.276 |
| Best bound (W, lag 0.15, ε 0.03) | 0.08 | 0.82 | 0.050 | 0.070 | **0.117** | 0.250 |
| Best MSD (G, f 2, α ≤ 0.5) | 0.21 | 0.52 | 0.060 | 0.072 | **0.101** | 0.183 |
| Best Geweke (M, f 4, Z ≤ 5) | 0.15 | 0.75 | 0.047 | 0.064 | **0.104** | 0.219 |

Per application, the best of each family at λ = 0.1 (horizon → best):
MNIST 0.130 → 0.097 (Geweke W); modular 0.102 → 0.052 (bound M, ε 0.3, norm gate) and
0.066 (Geweke M); teacher–student 0.123 → 0.079 (MSD M).

**Reading.**
- The settling tests beat the bound and the horizon, but only modestly: they close
  ~15–20% of the gap between the horizon (0.120) and the oracle (0.008) overall, and up
  to ~50% on modular. Premature stops drop for Geweke, not for MSD.
- **Why the gain is small: the features keep drifting after test accuracy has
  settled.** On MNIST, t_ε comes at a median 21% of the horizon, but the M path is still
  directed (α > 1) in 88% of the eval steps after t_ε, and the Geweke Z at the last step
  is 76 (jitter gives ~1). 27/61 runs are still directed at the very last step. On
  modular and teacher–student the features do settle (directed in 25% of post-t_ε
  steps). So "the features have stopped moving" is a much later event than "the test
  accuracy has stopped improving" in the Omnigrok-style MNIST runs. Plausible mechanism:
  the norm keeps shrinking and the features keep sharpening after accuracy saturates.
  Any F2 rule that waits for the features to settle is late there by construction.

### Do the features settle when test accuracy does? Per dataset (2026-11-15)

From the same 61 traces. "Angle to final" = angle between M = G^{1/2} at a step and M at
the last step; alignment = its cosine (Mallinar's measure). The **F2 oracle** stops at
the first step after which M stays within δ of its final value: no online F2 rule can do
better, so it bounds what F2 can achieve on each dataset. Costs at λ = 0.1, means.

| | MNIST (26) | modular (17) | teacher–student (18) |
|---|---|---|---|
| t_ε as a fraction of the horizon (median) | 0.21 | 0.02 | 0.07 |
| Angle memorization → final (median, rad) | 1.36 | 1.08 | **0.012** |
| Angle t_ε → final (median, rad) | **1.11** | 0.67 | 0.003 |
| Alignment at t_ε (median) | **0.45** | 0.78 | 1.00 |
| Share of the memorization→final angle still ahead at t_ε | 85% | 62% | – |
| Runs with angle t_ε → final > 0.3 rad | 77% | 94% | 0% |
| Change in log‖w‖ after t_ε (median) | −0.20 | −0.05 | −0.04 |
| erank(G) at the end / at t_ε (median) | **0.31** | 1.19 | 1.00 |
| Cost: oracle stop at t_ε / horizon | 0.009 / 0.130 | 0.006 / 0.102 | 0.009 / 0.123 |
| Cost: F2 oracle, δ = 0.05 / 0.1 / 0.2 / 0.3 | 0.128 / 0.140 / 0.133 / 0.131 | 0.068 / 0.056 / **0.052** / 0.069 | 0.63–0.81 (stops at step 0) |
| Best online F2 rule found | 0.097 (Geweke W) | **0.052** (bound M ε 0.3 + norm gate) | 0.079 (MSD M) |

**Three different situations.**
- **MNIST: the features keep changing long after accuracy settles, and F2 cannot work
  there even in principle.** At t_ε the features are only 0.45-aligned with their final
  state. After t_ε the norm keeps shrinking (Adam+L2 and SGD-family: log‖w‖ −0.6 to −0.8)
  and the AGOP compresses to ~1/3 of its effective rank, without changing predictions
  (Spearman 0.59 between post-t_ε feature angle and post-t_ε norm change, MNIST +
  modular). Even the F2 oracle stops at 84–100% of the horizon and does not beat
  training to the end.
- **Modular: F2 works as well as it can.** The features also move after t_ε (alignment
  0.78), but much of that is a persistent oscillation, not a drift: the same runs
  oscillate in test accuracy (Adam / edge-of-stability, Q4 floor ratio 0.90), so M never
  converges to a point. The F2 oracle halves the horizon's cost (0.102 → 0.052, 6%
  premature), and the best online rule already reaches it (0.052). Caveat: that rule
  was chosen among ~400 settings on these 17 runs, so the number is optimistic until it
  is calibrated on seeds {0, 1} and scored on {2, 3} (PLAN §5).
- **Teacher–student: the input AGOP carries no signal.** It is 5 × 5 and moves only
  0.012 rad over the whole run: the teacher uses all five input directions, so learning
  does not rotate the input-space features. Every F2 variant is uninformative there (the
  oracle "settles" at step 0); the 0.079 of the best rule is threshold luck.

## F2 final study: 224 runs, held-out calibration (2026-11-15)

Full report: [report_AGOP/REPORT.md](../../report_AGOP/REPORT.md) (plain-language, ten
sections, 13 figures; every number in `report_AGOP/tables/summary.csv` and
`tables/numbers.txt`). Earlier technical write-up: `report_AGOP/detailed/`. This summary
supersedes the shorter one written before the report was checked number by number.

**Setup.** 224 corpus runs (modular 160 = 40 settings x 4 seeds; MNIST 64 = 16 x 4),
replayed with verification (188 replayed, 36 reused from phase4/agop_study), 223 eval
points each. Five online F2 variants (F2-A bound, F2-B rotation over a lag f*t, F2-C
rotation over a fixed lag c*T, F2-D MSD exponent, F2-E Geweke; 954 settings), a "tuned
pick" across all of them, fixed fraction of the budget, never stop, two final-AGOP oracles
and the perfect stop at t_eps. Score = regret + lambda * (stop - t_eps)+ / T, lambda in
{0.03, 0.1 (main), 0.3}. Each rule tuned on one set of runs, scored on held-out runs:
Test A seeds 0-1 -> 2-3; Test B modular x+y -> x*y, x/y and MNIST d3 -> d8. 95% CIs:
percentile cluster bootstrap over settings (2,000 resamples, seed 0), paired for
differences; verdict "better/worse" only when the paired CI excludes 0.

**Findings (lambda 0.1 unless noted).**
- **Modular.** F2-B and F2-C beat never stopping on both tests (F2-C: 0.073 vs 0.128 and
  0.068 vs 0.138, better in 16/16 Test B settings). No variant is clearly better than
  the tuned fixed fraction (all CIs include 0); the tuned pick ties on both tests.
- **Safety.** The fixed fraction (40%, tuned on x+y) stops 52% of the x*y, x/y grokkers
  before they generalize; F2-B, C, D stop none, F2-A 1 of 31, F2-E 9 of 31. Caveat: F2-B
  and F2-D never stop on ~3/4 of Test-B grokkers (slow post-jump drift), so part of their
  safety is not stopping at all; F2-C stops all but 3 of 31 grokkers, every one after the
  jump.
- **Modular oracle.** Tuned oracles tie the fixed fraction, but the best oracle threshold
  in hindsight beats the best fixed fraction (0.044 vs 0.076; 0.035 vs 0.046), and at
  lambda 0.3 both tuned oracles are clearly better on Test A. So the features hold
  timing information beyond a budget fraction; the online rules could not exploit it
  with this much tuning data. (Earlier text saying "no feature rule could beat fixed"
  was wrong.)
- **Lambda.** At 0.03 the tuned fixed fraction becomes 100% (= never stop) and F2-A/F2-B
  (Test A) and F2-B/F2-C (Test B) beat it. At 0.3, F2-C tuned on x+y is clearly worse on
  Test B (+0.173, 23% premature, 48% of grokkers cut): safety depends on a cautious
  threshold.
- **MNIST.** The fixed fraction (0.034, 0.039) beats every variant on Test A and A/B/C
  on Test B (D, E tie). Only F2-D beats never stopping. The oracle is no better than
  never stopping even with the best threshold in hindsight (0.082, 0.091). Reason:
  alignment with the final M at t_eps is 0.44 (d3) / 0.27 (d8) vs 0.80 on modular; after
  t_eps the weight norm falls to 0.53 (d3) / 0.78 (d8) and the AGOP effective rank to
  ~0.27, while test accuracy stays flat.
- **Measurement.** Spearman(F2-C angle over 0.02T on M, true remaining angle): within a
  run (median) 0.79 modular, 0.81 MNIST d3, 0.20 MNIST d8; pooled 0.84 / 0.78 / 0.87.
  F2-A bound within a run: 0.37 / -0.01 / -0.17.
- **Tuning transfer.** Fixed fraction and never stop transfer almost exactly; angle
  thresholds do not transfer to the 8-layer MNIST network (F2-A, F2-C, both oracles).

**Conclusion.** F2 is closed. AGOP settling is a real, grokking-safe stopping signal where
feature learning and generalization finish together (modular), but on average no better
than a tuned budget fraction; it does not work on MNIST, and no feature-settling rule can
there. Keep the plain rotation over a window (F2-B budget-free, F2-C); drop extrapolation.

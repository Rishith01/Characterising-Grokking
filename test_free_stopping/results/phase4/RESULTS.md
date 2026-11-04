# Phase 4 results

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

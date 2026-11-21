# What the three earlier studies found, and what survives

This file condenses the three metric studies already run: weight norm (N1), activation effective rank (R1) and AGOP feature motion (F2). It separates the findings that still stand from the parts that depend on the old tune-and-score method, which [new_metrics.md](new_metrics.md) replaces.

**Sources** (nothing was re-run for this summary):
- [report_N1/REPORT.md](report_N1/REPORT.md) and its `tables/numbers.txt`;
- [report_R1/REPORT.md](report_R1/REPORT.md) and its `tables/numbers.txt`;
- [report_AGOP/REPORT.md](report_AGOP/REPORT.md) and its `tables/`;
- the section "Re-score on the F2 grid" in [results/phase4/RESULTS.md](results/phase4/RESULTS.md). It corrects several claims in the reports that were never fixed in the reports themselves.

---

## 1. What was run

| Study | Metric | Runs | Tasks | Data source |
|---|---|---|---|---|
| F2 (2026-11-15) | Angle between AGOP feature matrices over time; 5 online variants, 954 rule settings | 224 | modular, MNIST | replay from 10k-step keyframes, 223 evaluation points per run |
| Study A, N1 (2026-11-16) | Change of log‖w‖ over a lag; 140 rule settings | 320 | modular, MNIST, teacher–student | 50-step telemetry |
| Study A, R1 (2026-11-16) | Change of log(effective rank of hidden activations); 210 rule settings | 320 (the same runs) | modular, MNIST, teacher–student | 500-step summaries |

**The runs** were chosen as whole settings with all 4 seeds, from 40 modular, 16 MNIST and 24 teacher–student settings. They are **stratified by outcome**: within each task variant and optimizer, settings were picked from each outcome type seen on seed 0. So the outcome counts below describe this sample, not how common each outcome is in the 1,664-run corpus.

**The method** in all three studies:
1. Turn the metric into a stopping rule.
2. Tune its settings on half the runs.
3. Score the tuned rule on the other half with cost = (accuracy lost) + λ·(budget wasted after t_ε).
4. Compare it with a tuned "stop at a fixed fraction of the budget".

t_ε is the first step at which test accuracy comes within 1 point of its best.

---

## 2. What to discard

These parts are artifacts of the scoring method. They should not be quoted as findings.

1. **Every "better / tie / worse" verdict against the fixed fraction.**
   - At the main λ = 0.1, every metric **ties** the tuned fixed fraction on every modular test.
   - The verdicts change with λ, and λ is arbitrary.
   - The confidence intervals are wide (about ±0.03–0.04), because there are only 8–40 independent settings per test.
2. **R1's "clearly better on new modular operations".** It does not survive:
   - The win came from a patience mismatch: the fixed fraction was applied without the 3-check patience that the F2 study used.
   - Scored against the fixed fraction exactly as the F2 study stopped it, the comparison is a tie: R1 0.065 vs 0.090, difference −0.026 [−0.066, +0.011].
3. **The "decision grid" explanation** of why "stop at 40%" scored 0.126 in study A but 0.090 in F2. The actual cause is patience (point 2), not the grid.
4. **Tuned thresholds and lags as facts about networks.** Values like "|Δlog‖w‖| ≤ 0.003 over 5–10% of T" are whatever minimized an arbitrary cost on about 80 runs.
5. **Scores compared across reports.** Within a report the comparison is fair. Across reports it is not, because patience was handled differently.
6. **The F2 variants built on statistical tests and extrapolation** (F2-A bound, F2-D MSD, F2-E Geweke). Already dropped in [new_metrics.md](new_metrics.md); see §3.3 for why.

**Wrong statements still present in the source files**, listed in RESULTS.md and not yet corrected:
- SUMMARY.md §0 point 4, §5, §6 point 10, and §7 Q4 and Q9;
- report_R1's "clearly better" claims (summary paragraph, table 9.1, §9.2, conclusion 1);
- the "Comparing with the F2 report" paragraphs in report_N1 §9.3 and report_R1 §9.2.

---

## 3. What survives: descriptive findings

These findings come from curves lined up on each run's t_ε, or from the run outcomes themselves. They don't depend on λ, on tuning, or on any stopping rule, so they stand.

**Caveats that apply to all of them.**
- **They use hindsight.** Curves are aligned on t_ε, which is measured on the test set.
- **Most are medians over runs that learned:** 123 modular, 32 MNIST 3-layer and 20 MNIST 8-layer runs, plus 80 teacher–student runs in study A.
- **The sample is stratified by outcome,** so these are not corpus-wide estimates. They are the hypotheses to re-test on the full corpus.

### 3.1 The runs themselves

| | Modular (160) | MNIST (64) | Teacher–student (96) |
|---|---|---|---|
| Learn fast | 50 (median t_ε 2% of budget) | 29 (1.6%) | 27 (3%) |
| Grok | 61 (median 38%) | 19 (40%; 17 on the 3-layer net) | 50 (17%, spread 2–87%) |
| Never learn | 37 (32 of them have γ = 0) | 12 (all 8-layer; 8 are SGD at lr 1e-4) | 16 |
| Other (stall-then-rise, collapse, never fit) | 12 | 4 | 3 |

- **Modular grok times cluster.** 31 of the 61 grokkers have t_ε at 35–55% of the budget. The x×y and x÷y runs grok later than x+y.
- **Grokking looks different across tasks.** On modular it is a jump from chance to 100%. On MNIST it is a slow creep to about 91%. On teacher–student it is a gradual rise whose final level grows with N: median best accuracy 0.55, 0.80 and 0.97 for N = 50, 100 and 400.
- **No weight decay mostly means no generalization on modular** within the 100k-step budget.

### 3.2 Weight norm (from N1)

Each entry is the median norm relative to its value at t_ε:

| | At 3·t_ε | At end | \|Δlog‖w‖\| over last 10% of budget |
|---|---|---|---|
| Modular | 0.89 | 0.89 | 0.000 |
| MNIST 3-layer | 0.62 | 0.62 | 0.009 |
| MNIST 8-layer | 0.96 | 0.78 | 0.016 |
| Teacher–student | 0.97 | 0.98 | 0.003 |

**Findings.**
- **N-a. Modular: the norm first grows, then shrinks through the jump, then stops.**
  - During memorization the norm *grows*: in the median modular run it goes from 39 at the start to 57 at t_ε.
  - It then shrinks through the jump and is flat afterwards.
  - This matches the Omnigrok picture, with one addition the picture doesn't emphasize: a growth phase before the decay phase. That growth phase is why "the norm stopped shrinking" fires on the memorization plateau.
- **N-b. MNIST: the norm keeps shrinking long after test accuracy has finished,** while accuracy doesn't change.
- **N-c. Teacher–student: a slow downward drift over the whole run,** with no feature at t_ε.
- **N-d. With γ = 0 the norm can sit still from the first steps.** A flat norm then says nothing about learning. This is the cause of the 8-layer MNIST failure: all 12 runs stopped before t_ε had γ = 0, and were stopped at step 2,000.
- **N-e. After the jump, some modular grokkers keep drifting in norm, or climb back up.** These are AdamW, SGD and SGD+momentum runs, none with Adam+L2. A strict flatness rule never fired on 21–41% of modular runs. The evidence is anecdotal (one example figure, plus the never-stop counts), but it is exactly what N.3 / W should measure.

### 3.3 Feature motion, AGOP (from F2)

| | Modular (123) | MNIST 3-layer (32) | MNIST 8-layer (20) |
|---|---|---|---|
| Alignment cos(M_{t_ε}, M_T) (median) | **0.80** | **0.44** | **0.27** |
| AGOP effective rank, end ÷ t_ε | 1.03 | 0.27 | 0.26 |
| Weight norm, end ÷ t_ε (F2 replay) | 0.88 | 0.53 | 0.78 |

**Findings.**
- **A-a. Modular: features and accuracy finish together.**
  - At t_ε the features are already 0.80 aligned with their final form; the alignment climbs while test accuracy is still near chance. That is the paper's "hidden progress", reproduced on MLPs and across four optimizers.
  - The features finish settling within a few multiples of t_ε, and the AGOP rank does not change afterwards.
- **A-b. MNIST: accuracy finishes long before the features do.**
  - After t_ε the AGOP concentrates onto about a quarter as many directions, and the weights shrink.
  - Test accuracy doesn't change during this. The network keeps simplifying under weight decay in a way that neither helps nor hurts the test score.
- **A-c. Knowing the final features does not tell you when MNIST accuracy finished.**
  - Even a stopping rule that knows M_T (the oracle) is no better than training to the end on MNIST.
  - This finding uses the old score, but its conclusion doesn't depend on λ: on MNIST the timing of feature settling and the timing of accuracy are unrelated.
- **A-d. The online angle tracks the true remaining change.** This is measured as the within-run rank correlation between the constant-lag online angle (F2-C, now AO.2) and the true angle still to go:

  | | Modular | MNIST 3-layer | MNIST 8-layer |
  |---|---|---|---|
  | Constant-lag angle (F2-C) | 0.79 | 0.81 | 0.20 |
  | Extrapolated bound (F2-A) | 0.37 | −0.01 | −0.17 |

  - Plain online angles are faithful on modular and on the 3-layer MNIST network.
  - Extrapolation is not, because feature motion has **two timescales**: a fast slowdown right after memorization, then a slow drift that carries much of the remaining change.
  - On the 8-layer network, within-run tracking is weak (0.20), even though the pooled correlation is 0.87.
- **A-e. Most modular grokkers keep a slow residual drift after settling.** Strict angle thresholds never fired on 24 of 31 grokkers of the new operations (F2-B). This mirrors N-e.
- **A-f. On modular, the features carry timing information beyond "a fraction of the budget has passed."** The best oracle threshold in hindsight scored 0.044 against 0.076 for the best fixed fraction.
  - This is suggestive only. It is a hindsight pick under the old score, and with the threshold tuned honestly the oracle only tied.

### 3.4 Activation effective rank (from R1)

Each entry is the median mean-over-layers rank relative to its value at t_ε:

| | At 3·t_ε | At end | \|Δlog rank\| over last 10% of budget |
|---|---|---|---|
| Modular | 0.46 | 0.44 | 0.006 |
| MNIST 3-layer | 0.54 | 0.45 | 0.010 |
| MNIST 8-layer | 0.81 | 0.76 | 0.011 |
| Teacher–student | 1.00 | 1.00 | 0.001 |

**Findings.**
- **R-a. Modular: the representation halves its rank through and just after the jump,** then settles.
- **R-b. Modular: the last hidden layer's rank often collapses *before* the jump.**
  - At the end of memorization it falls several-fold (it starts at about 5× its value at t_ε). It then sits flat through the plateau while test accuracy is still at chance.
  - This is the most interesting new observation from these studies. Compression of the last layer precedes generalization rather than coinciding with it. It is also why last-layer rules stopped on the plateau.
  - The evidence is a median curve and example runs. It has not been quantified as a timing (for example, log(t_collapse/t_ε) per run).
- **R-c. MNIST: the rank first rises, then falls through and after t_ε.**
  - It rises early in 75% of 3-layer runs and 90% of 8-layer runs (the network spreads out to memorize), then compresses.
  - Like the norm and the AGOP, it keeps compressing after accuracy is done.
- **R-d. Teacher–student: the rank is pinned at about 5** (the input dimension). The median mean rank is 5.6 at the start, 5.2 at t_ε and 5.1 at the end, out of 100 units. This is a structural limit: R1 cannot see compression where there is no room for it.

### 3.5 Across the three metrics

This is the main thing worth carrying forward.

| | Modular | MNIST | Teacher–student |
|---|---|---|---|
| Weight norm | grows while memorizing, shrinks through the jump, flat after | keeps shrinking after t_ε | slow drift throughout |
| Activation rank | halves through the jump (last layer can collapse earlier), flat after | rises, then keeps falling after t_ε | flat at about the input dimension |
| AGOP features | 0.80 aligned at t_ε, settle with the jump | 0.44 / 0.27 aligned at t_ε, keep simplifying | not measured |
| **Summary** | **all three settle together with test accuracy** | **all three keep moving long after test accuracy has finished** | **little moves; nothing marks t_ε** |

1. **Modular arithmetic is the clean case.** Norm, rank and features change during the jump and stop soon after. A rule that waits for any of them to stop does not cut off a grokker:
   - on new seeds, the tuned N1 and F2 cut off 0 of 30 grokkers; R1 cut off 0–3 of 30, depending on version and grid (its last-layer rule stops on the plateau, R-b);
   - on new operations, N1 and F2 cut off at most 1 of 31; R1 cut off 0 of 31 on study A's grid and 5 of 31 on the F2 grid, a patience artifact.
   - The fixed fraction cut off 16 of 31 on new operations.
   That safety is the only stopping result that does not depend on λ. Even it is partly a property of a strict threshold: at λ = 0.3 the tuned F2-C cut off 48% of new-operation grokkers.
2. **On MNIST, post-learning simplification is decoupled from test accuracy.** Under weight decay the network keeps shrinking its weights, compressing its activations and concentrating its AGOP long after test accuracy is flat. None of these signals marks "learning is done". This is a real finding about the training trajectory, not a failure of one metric.
3. **Residual drift after grokking is common.** All three signals keep a slow drift on many modular grokkers after the jump (N-e, A-e). Whether that drift is noise, a slow approach to a fixed point, or real continued change is exactly what N.3 / W.1 and AO.2 are designed to answer.
4. **Weight decay is what drives every signal.** With γ = 0 the norm can be flat from the start, and modular runs mostly never generalize. Stage 1 analysis should split every result by γ = 0 vs γ > 0.

---

## 4. Stopping results, compressed

For the record only. This is the one table where all metrics were checked at the same steps with the same patience, from the "Re-score on the F2 grid" section of RESULTS.md. λ = 0.1; each rule was re-tuned; the verdict is against the tuned fixed fraction.

| Test | Fixed fraction | F2 | N1 | R1 |
|---|---|---|---|---|
| Modular, new seeds | 0.076 | 0.073 tie | 0.080 tie | 0.088 tie |
| Modular, new ops | 0.090 | 0.086 tie | 0.093 tie | 0.148 tie (patience artifact) |
| MNIST, new seeds | 0.034 | 0.070 **worse** | 0.044 tie | 0.041 tie |
| MNIST, 8-layer | 0.039 | 0.119 **worse** | 0.251 **worse** (γ = 0 runs) | 0.058 tie |

Never stopping scores 0.10–0.14 and a perfect stop about 0.01.

**Reading it.**
- No metric beats a well-chosen fixed fraction.
- Two metrics fail outright on MNIST, for the descriptive reasons in §3: A-b for F2 and N-d for N1.
- With this power the comparison could not detect a small real advantage anyway.

---

## 5. Inconsistencies to resolve in the new pipeline

1. **Two values for the same MNIST 3-layer norm ratio.** Weight norm at the end ÷ at t_ε is 0.62 in study A (50-step telemetry) and 0.53 in F2 (replay). The two studies may compute the norm or the reference point differently. The common-grid pipeline should give one number.
2. **Two grids, different patience semantics.** R1's 500-step data read on F2's denser grid let "3 checks in a row" rest on 1–2 actual measurements. The common grid in [new_metrics.md](new_metrics.md) §0 removes this.
3. **Duplicate setting.** One modular x+y γ = 0 setting appears in the selection as both AdamW and Adam+L2; the runs are identical. That is 1 of 40 modular settings. Deduplicate it in the full-corpus analysis.

---

## 6. Hypotheses to carry into the full-corpus analysis

Each item below is a claim from §3, restated so the new metrics can confirm or refute it on all 1,664 runs.

| # | Claim | Tested with |
|---|---|---|
| H1 | On modular, norm, rank and features all settle within a small multiple of t_ε | t₉₀ of N.1, R.1 and AP.1 relative to t_ε |
| H2 | On modular, the last hidden layer's rank collapses before the jump, at the end of memorization | per-layer R.1: log(t_collapse / t_fit) and log(t_collapse / t_ε) |
| H3 | On modular, the norm grows during memorization and shrinks during the plateau; the plateau is pure decay | N.1 shape; N.3 ≈ 1 on the plateau, across optimizers and γ |
| H4 | On MNIST, norm, rank and AGOP keep compressing after t_ε with no effect on accuracy | "still moving at T" flag; AP.1 at t_ε; R.3 ratio after t_ε |
| H5 | AGOP alignment at t_ε is high on modular (hidden progress) and low on MNIST | AP.1 at t_ε, split by task and optimizer |
| H6 | Grokkers keep a slow residual drift after the jump | W.1 settled or not; AO.2 level after t_ε |
| H7 | Without weight decay, none of the signals marks learning | every descriptor split by γ = 0 vs γ > 0 |
| H8 | Constant-lag online angles track the true remaining feature change; extrapolation does not | AO.1 and AO.2 against AP.1, within-run correlation |

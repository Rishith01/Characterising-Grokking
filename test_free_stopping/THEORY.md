# Theory review of the metric set: what is grounded, and what AdamW needs

**What this is.** A theoretician's pass over the candidate metrics (METRICS.md §3,
AGOP.md §4), written 2026-11-14 at the user's request. Question asked: are the
self-derived parts (the online AGOP signal, and the "step-based" handling of AdamW)
hand-wavy, or do they rest on published results? And does each metric treat AdamW (and
Adam+L2) as what it is, instead of as SGD? **Status: recommendations; nothing here is
built or decided.**

**Citations.** † = cited from memory, not yet checked against the paper (same convention
as METRICS.md). None of the papers introduced here is among the local PDFs.

**Verdict in five lines.**
1. **F2's remaining-angle bound is not hand-wavy.** It is the classical Banach
   fixed-point *a posteriori* error estimate, with an estimated contraction rate, applied
   on the sphere. That is a textbook stopping test for iterative solvers. Cite it as such.
   It is optimizer-agnostic, but it has one unaddressed failure mode (§4.3) and needs a
   norm companion.
2. **ρ (step-based) is close but has two errors**: it drops the second-order term
   ‖Δw‖², which is *not* negligible under AdamW, and its normalizer lr_eff·γ is wrong for
   Adam+L2. Both fixes are exact and turn ρ into a published quantity: Yaida's
   fluctuation–dissipation relation (§2).
3. **τ cannot work as defined under any noisy or Adam-type optimizer.** Its stationary
   value is not 0 but is fixed by the norm balance (Kosson et al.'s rotational
   equilibrium). Replace it with a windowed version (§3).
4. **S1b is correct for SGD, SGD+momentum and Adam+L2, and wrong for AdamW.** The
   AdamW-correct null hypothesis can be derived (§5), and at our scales it reduces to B2
   plus a negligible correction.
5. **N1 and R1 are fine as they are**; nothing in them is optimizer-specific.

---

## 0. The criterion: what "handling AdamW properly" means

Every candidate except R1 is a **stationarity** signal: it asks whether training has
reached the point where it will stop moving. Training stops where the **update map**
w_{t+1} = Φ(w_t, optimizer state, batch) stops moving w. That set depends on the
optimizer:

| Optimizer | Step Δw | Fixed-point condition | Same as ∇J = 0? |
|---|---|---|---|
| SGD + L2 | −lr(g + γw) | g + γw = 0 | yes |
| SGD + momentum + L2 | −lr·(EMA of g + γw) | g + γw = 0 | yes |
| Adam + L2 | −lr·P(m̂ of g + γw), P = diag(1/(√v̂+ε)) ≻ 0 | P(g + γw) = 0 ⇔ g + γw = 0 | **yes**: a positive preconditioner does not change the zero set |
| AdamW | −lr(m̂/(√v̂+ε) + γw) | m̂/(√v̂+ε) = −γw | **no** |

So "treating AdamW like SGD" means using any condition of the form g + γw ≈ 0. Three
kinds of metric are legitimate for every optimizer:
1. **Gradient conditions**, but only with the right zero set (the table above).
2. **Distributional stationarity of the iterate**: E[O(w_{t+1})] = E[O(w_t)] for some
   observable O. This needs no formula for the step, so it holds for any optimizer
   (Yaida 2019, §2).
3. **Geometric convergence of a path** measured from the iterates themselves (Banach's
   a posteriori estimate, §4). Also optimizer-free.

One more thing that changes with the optimizer is the **time unit** (lr spans 3e-4 to
1.0, and Adam has no single effective lr). A signal must either be dimensionless or be
normalized by a rate that the optimizer actually applies. Below, each metric is checked
on both counts.

---

## 1. Verdict table

| Metric | Grounded in | Correct for AdamW? | Correct for Adam+L2? | Change |
|---|---|---|---|---|
| N1 | Omnigrok | yes (it is just ‖w‖) | yes | none; note it equals ρ* × decay rate (§2.4) |
| S1 ρ (step form) | first-order norm dynamics | **no**: drops ‖Δw‖² | **no**: wrong normalizer | use ρ* (§2) |
| S1 τ | none (normalization is a choice) | **no**: floor ≠ 0 | no | replace by a windowed direction signal (§3) |
| S1b | Mahsereci 2017 + correct fixed point | **no** | yes | AdamW null hypothesis from §5 |
| F2 R̂ | Banach a posteriori bound + NFA | yes | yes, with one fix to AGOP.md §1.4 | cite Banach; add floor on all Adam runs; AND with norm (§4) |
| R1 | Roy–Vetterli; Galanti †; Deep Grokking | yes (activation space) | yes | rate-normalize like F2 (§6) |

---

## 2. ρ: from "step-based" to the exact, published version

### 2.1 What is wrong with the current step-based form

METRICS §3 defines ρ = −⟨Δw, w⟩ / (lr_eff·γ·‖w‖²) and says the second-order term of
‖w_{t+1}‖² = ‖w_t‖² + 2⟨Δw, w⟩ + ‖Δw‖² is negligible. Two problems.

**(a) The second-order term is not negligible under AdamW.** Adam's step has a
per-coordinate size of about lr·c, with c ∈ (0, 1] roughly the update's RMS per
coordinate, *whatever the gradient's size* (that is the point of normalizing by √v̂). So
‖Δw‖² ≈ lr²·P·c², which does not shrink as the gradient shrinks. Compared with the decay
term:

  ‖Δw‖² / (2·lr·γ·‖w‖²) ≈ lr·P·c² / (2γ‖w‖²)

- MNIST depth 3, AdamW (lr 1e-3, γ = 0.01, P = 2e5, ‖w‖ ≈ 20): **≈ 25·c²**, i.e.
  between 0.25 and 25 for c between 0.1 and 1. It is the same order as the term kept, or
  larger.
- MNIST depth 3, SGD (lr 0.03, γ = 1e-3, ‖w‖ ≈ 10): ≈ 0.15·‖g‖², small unless the
  minibatch gradient is large. So the approximation is fine for SGD and breaks for AdamW,
  which is exactly the "AdamW treated like SGD" failure.

Consequence: at a true steady state, E‖w_{t+1}‖² = E‖w_t‖², so
−2⟨Δw, w⟩ = ‖Δw‖² on average. The first-order ρ then settles at **‖Δw‖²/(2·lr·γ·‖w‖²)
> 0, not at 0**, and under AdamW that value can be order 1. A rule waiting for ρ → 0
would never stop, or the threshold would silently absorb an optimizer-dependent offset.

**(b) lr_eff·γ is not the decay rate under Adam+L2.** The decay term goes through the
preconditioner: its step is −lr·P·γw, so its radial rate is lr·γ·Σ_k w_k²/(√v̂_k+ε) /
‖w‖², which is larger than lr·γ by the typical 1/√v̂ (often 10²–10⁴). The Phase 3 grid
already shows this: Adam+L2 needs γ = 1e-6 to 1e-4 where AdamW uses 0.01 to 1. Dividing
by lr·γ makes ρ for Adam+L2 a huge number that does not mean "units of pure decay".

### 2.2 The fix: ρ*

Split each step into a decay part d_t and the rest. Let **D_t = −⟨d_t, w_t⟩ > 0**, the
radial pull of the decay alone in this step. Define

> **ρ*_t = −(‖w_{t+1}‖² − ‖w_t‖²) / (2·D_t) = −(⟨Δw, w⟩ + ½‖Δw‖²) / D_t**

D_t per optimizer:

| Optimizer | D_t | From |
|---|---|---|
| SGD + L2 | lr·γ·‖w‖² | telemetry |
| SGD + momentum | ≈ lr_eff·γ·‖w‖² (steady state) | telemetry |
| AdamW | lr·γ·‖w‖² (**exact**: PyTorch applies w ← (1 − lr·γ)w) | telemetry |
| Adam + L2 | ≈ lr·γ·Σ_k w_k²/(√v̂_k + ε) (m̂ is linear in its inputs, and the EMA of γw ≈ γw because w moves slowly) | **replay** (needs v̂); cheap, no extra backward pass |

Everything else is already logged: ⟨Δw, w⟩ = `update_radial`·`w_norm` and
‖Δw‖ = `update_norm` (tfs/training/loop.py:107–111). So ρ* costs nothing for SGD, SGD+m
and AdamW, and one pass over the optimizer state at eval steps for Adam+L2. No corpus
re-run is needed.

**Meaning, now exact for every optimizer:**
- ρ* = 1: the norm shrinks exactly as fast as decay alone would make it (the data part
  has no net radial effect, second order included). The Omnigrok plateau.
- ρ* = 0: the norm is stationary.
- ρ* < 0: the norm grows (the Type A rule of METRICS §5 carries over unchanged).

### 2.3 Why ρ* is a published quantity, not a self-derived one

ρ* = 0 in expectation is **Yaida's first fluctuation–dissipation relation**: at a
stationary state of the iterates, E‖w_{t+1}‖² = E‖w_t‖², which for SGD reads
⟨w·∇J_B⟩ = (lr/2)·⟨‖∇J_B‖²⟩. The relation comes from stationarity of the iterate
distribution alone, so it holds for any update rule; the optimizer only changes what Δw
is. ρ* is that relation divided by the decay's radial pull.
- Yaida, *Fluctuation-dissipation relations for stochastic gradient descent*, ICLR 2019,
  arXiv:1810.00004 †.
- Lang, Xiao, Zhang, *Using Statistics to Automate Stochastic Optimization* (SASA),
  NeurIPS 2019, arXiv:1909.09785 †: uses exactly this relation, with a batch-means
  t-test, as an automatic stationarity test that decides when to drop the learning rate.
  **This is a published, statistically calibrated stationarity rule of the same family**,
  and a better template for ρ*'s threshold than a hand-picked one: test whether the
  window mean of the per-step residual (⟨Δw, w⟩ + ½‖Δw‖²)/D is 0 within its confidence
  interval.
- The equilibrium norm under AdamW, where the decay balances the second-order growth:
  Kosson, Messmer, Jaggi, *Rotational Equilibrium: How Weight Decay Balances Learning
  Across Neural Networks*, ICML 2024, arXiv:2305.17212 †. For SGD with normalization
  layers: van Laarhoven 2017 †, Wan et al. 2021 (*Spherical Motion Dynamics*) †.
- The "ρ = 1 on the plateau" reading is still Omnigrok's mechanism.

### 2.4 N1 and ρ* are the same signal

The relative change of ‖w‖ over a window is d log‖w‖ = −(D/‖w‖²)·ρ*. So ρ* *is* N1's
log-rate, measured in units of the decay rate. Keep N1 as the free, unnormalized
reference; ρ* is its correctly normalized form, and the one that can share one threshold
across optimizers.

**γ = 0.** D = 0 and ρ* is undefined. The clean fallback is not ‖g‖ or ‖Δw‖ but the
dimensionless remaining-log-norm bound of §4.4, which needs no γ.

---

## 3. τ: why it cannot work, and the replacement

At stationarity (ρ* = 0) the step still has size: ‖Δw‖² = −2⟨Δw, w⟩ on average, so
‖Δw_⊥‖ is pinned by the radial balance, not by whether features are still forming. In
the scale-invariant idealization where the data part of the step is orthogonal to w
(Kosson et al.'s setting), the angle turned per step settles at √(2·lr·γ) for AdamW, so

  τ_equilibrium = √(2 / (lr·γ)) ≈ 450 for MNIST AdamW (lr 1e-3, γ 0.01).

Our MLPs are not scale-invariant, so the number differs, but the conclusion does not: a
**one-step** tangential size cannot tell coherent rotation (features forming) from
equilibrium jitter (noise rotating the weights back and forth). Both give large ‖Δw_⊥‖.
Under full-batch AdamW the same holds, because Adam oscillates around a minimum at
constant lr instead of converging to it (Cohen et al., *Adaptive Gradient Methods at the
Edge of Stability*, arXiv:2207.14484 †).

**What does distinguish them is persistence over a window:** coherent motion adds up
linearly in the number of steps, jitter adds up as √(steps) or saturates. So the
tangential test must compare directions K steps apart, not single steps. That is exactly
what F2's remaining-angle bound does in feature space.

**Replacement (recommended): the same R̂ machinery applied to each layer's weight
direction ŵ_l.** Angle between window-mean ŵ_l at eval steps, its decay rate, R̂_w,l =
ω/κ. Needs w at eval steps, which replay already produces (no Jacobians, so far cheaper
than F2). This makes "is the direction still moving?" and "are the features still
moving?" two instances of one theory instead of two ad hoc numbers. The 8 random
projections in `fingerprints.parquet` would give a free log-cadence estimate of
‖w_t − w_s‖, but with ~±50% error from 8 projections; use them only as a cross-check.

---

## 4. F2, the remaining-angle bound: grounding and gaps

### 4.1 It is the Banach a posteriori estimate

For an iteration x_{n+1} = T(x_n) with d(x_{n+1}, x_n) ≤ q·d(x_n, x_{n−1}) for all
future n (q < 1), the contraction-mapping theorem gives

  d(x_n, x*) ≤ q/(1 − q) · d(x_n, x_{n−1}).

Proof: the triangle inequality along the remaining path and a geometric series. AGOP.md
§4 rederives exactly this: the triangle inequality (step 4) plus a geometric tail
(step 6). With q = e^{−κΔ} and d(x_n, x_{n−1}) = ωΔ, q/(1 − q)·ωΔ = ω/κ − ωΔ/2 + O(Δ²),
so R̂ = ω/κ is the continuous-time version, slightly conservative. Estimating q from
successive differences and stopping when the bound is below tolerance is the standard
termination test for fixed-point and Newton iterations:
- Banach 1922 (any functional-analysis text states the a posteriori form).
- Kelley, *Iterative Methods for Linear and Nonlinear Equations*, SIAM 1995 †, and
  Deuflhard, *Newton Methods for Nonlinear Problems*, Springer 2004 †: termination
  tests of the form ‖s_n‖·q̂/(1 − q̂) ≤ tol.

The geometry is also rigorous: the sphere of unit-Frobenius matrices with the angle
metric is a complete metric space, so the triangle inequality and the bound hold
exactly. The bound also survives coarse sampling: the angles between sampled points are a
lower bound on arc length, but the triangle inequality holds for the sampled polygon just
as well. **The team's Aitken extrapolation and R̂ come from the same classical family**;
the difference is that R̂ uses only the size of the remaining path, not its direction.

So the right way to write it up: "the Banach a posteriori bound with an estimated
contraction rate, on the angular metric of the normalized feature matrix (NFA:
Radhakrishnan et al. 2024 †; feature matrix and alignment: Mallinar et al.)". Not novel
mathematics, and that is a strength.

### 4.2 AdamW: what it needs

R̂ uses only the iterates, so no gradient formula enters. Three optimizer-specific points:
1. **Noise floor on full-batch Adam runs too.** AGOP.md §4.5 subtracts a floor on MNIST
   only. Full-batch Adam also does not converge to a point at constant lr (§3, Cohen et
   al. †): its per-coordinate step stays near lr. So modular and teacher–student AdamW
   and Adam+L2 runs need the same floor estimate, or R̂ will read the oscillation as
   unfinished motion and stop late or never.
2. **AGOP.md §1.4 is wrong for Adam+L2.** "Pure weight decay does not move the
   normalized AGOP" holds for SGD and AdamW, whose decay shrinks every coordinate by the
   same factor. Under Adam+L2 the decay is preconditioned per coordinate, so it is not a
   uniform rescaling and it does rotate ŵ and M̂. Harmless for R̂ (it measures any
   rotation), but the sentence should be restricted.
3. **The tail model** (linear convergence near a stable fixed point) is the
   linearization of a smooth map. Adam's map with small ε is close to sign-like, so the
   tail may be a floor plus oscillation rather than an exponential. R̂ fails safe there
   (R̂ = ∞), so the risk is never-stopping, not premature stopping. Check κ̂ stability on
   real traces (the proposal's D_exp test, AGOP.md §4.6).

### 4.3 The gap that matters: two timescales

The Banach bound requires the contraction to hold **for all future steps**. Its
dangerous case is not a flat plateau (κ̂ ≈ 0 gives R̂ = ∞, safe) but a **fast transient
that ends on a slow drift**:
- During memorization the features move fast (the pilot's R1: erank rises to 80).
- When memorization ends, that speed falls quickly, so κ̂ is large and **R̂ = ω/κ̂ dips
  toward 0 right at the start of the plateau**, which is where B1 and B4 fail.
- R̂ recovers to ∞ only once the trailing window sees the speed flatten onto the slow
  plateau drift.

The toy (AGOP.md §5) cannot show this, because its path starts at the memorized state:
there is no memorization transient. Whether the dip crosses ε_R is an empirical question
for the real traces, and it is the first thing to look at.

**Principled defence: the norm.** M̂ is normalized, so R̂ is blind to the norm by
construction. But on the Omnigrok plateau the norm is precisely what is still moving
(ρ* ≈ 1). The state is (direction, norm), and the Banach hypothesis must hold for the
whole state. So the stop rule should require **both** coordinates to have settled:

> stop when R̂_features < ε_R **and** the norm has settled (|ρ*| < ε_ρ, or R̂_norm < ε_n
> from §4.4), each for the patience window.

This is not a heuristic AND-gate; it is the bound applied to the full state that the
normalization split in two.

### 4.4 The same bound for the norm (and the γ = 0 fallback)

Apply the same estimate to n_t = log‖w_t‖: speed |ṅ|, decay rate κ_n of that speed,
**R̂_norm = |ṅ|/κ_n**, the remaining change in log-norm (a dimensionless number: 0.05
means "at most about 5% more norm change"). No lr, no γ, so it is defined at γ = 0,
which settles S1's open γ = 0 fallback (METRICS §6.5). Under cross-entropy at γ = 0 the
norm grows like log t, κ_n → 0 and R̂_norm stays large: the rule does not stop on a
Type A drift, which is the right behaviour.

---

## 5. S1b under AdamW: the correct null hypothesis

S1b tests g + γw against minibatch noise. For SGD, SGD+m and Adam+L2 this is the right
zero set (§0), so S1b stands as written. For AdamW it is not.

**Derivation (noise-dominated regime, leading order).** Let ḡ_k be the mean data
gradient on coordinate k, σ_k² the per-example variance, B the batch size, and
z_k = √B·ḡ_k/σ_k its size in standard errors. Adam's moments track E[m̂_k] ≈ ḡ_k and
E[v̂_k] ≈ ḡ_k² + σ_k²/B, so the mean update is

  E[u_k] ≈ ḡ_k / √(ḡ_k² + σ_k²/B) = z_k / √(1 + z_k²).

AdamW's fixed point is E[u_k] = −γw_k, so **z_k = z*_k := −γw_k / √(1 − γ²w_k²)**. The
evidence statistic with the right null is

> **EB_AdamW = 1 − (1/P)·Σ_k (z_k − z*_k)²** (same masking and per-example pass as S1b).

At our scales γ|w_k| is tiny (typical |w_k| ≈ ‖w‖/√P ≈ 0.04; γ ≤ 0.1 on MNIST), so
z*_k ≈ −γw_k contributes ~1e-5 to EB. **For AdamW the right test is essentially B2, the
test on g alone, not on g + γw.** Adding γw, as S1b does, adds a term of size √B·γ|w_k|/σ_k
per coordinate that is real but not part of AdamW's fixed point, so S1b on AdamW stays
negative forever.

Grounding: Mahsereci et al. 2017 for the test; Xie & Li 2024 † for AdamW's fixed point
m̂/√v̂ = −γw; Malladi et al., *On the SDEs and Scaling Rules for Adaptive Gradient
Algorithms*, NeurIPS 2022, arXiv:2205.10287 †, for the noise-dominated approximation
E[u] ≈ ḡ/σ (the same assumption behind Adam's square-root lr scaling rule).

**This predicts something checkable on the existing pilot.** The pilot explanation for
B2 never stopping on Adam-family runs ("the mean data gradient balances the decay") is
right for Adam+L2 but, by this derivation, not for AdamW: at an AdamW stationary state,
mean z_k² → 1 and B2's EB → 0. So on the pilot's AdamW runs, either EB rises toward 0 at
the end, or those runs were not stationary yet (the γ = 0.01 Type A run is not: its norm
grows). Checking this on the stored traces tests §5 at no cost.

**Caveats.** Leading order only: E[m̂/√v̂] ≈ E[m̂]/√E[v̂] ignores the correlation of m̂ and
v̂. Coordinates with √v̂ ≲ ε must be masked. In full-batch AdamW, σ = 0 and the update is
sign-like, so the test has no meaning there; this is the same full-batch caveat S1b
already has (use it on MNIST only).

---

## 6. N1 and R1

- **N1**: the norm is optimizer-agnostic. Keep it as the free reference; ρ* is its
  normalized form (§2.4).
- **R1**: erank of activation covariances lives in function space, so it does not care
  which optimizer produced the weights. Grounded in Roy & Vetterli 2007, Galanti et al. †
  and Deep Grokking. Its rate needs the same normalization as F2; use the same
  dimensionless "remaining change" construction on log erank, which also removes the lr
  dependence METRICS §3 flags.
- Note that R1 shares F2's two-timescale risk: in the pilot, erank *rises* during
  memorization and then falls. A rule on R1 must not read the end of the rise as
  settling.

---

## 7. Recommended set

| Role | Signal | Status |
|---|---|---|
| Norm (radial) | **ρ*** (§2.2), with SASA-style t-test on its window mean; N1 kept as raw reference | replaces S1 ρ |
| Direction (weights) | **R̂_w per layer** (§3) | replaces S1 τ |
| Features | **F2 = R̂ on G^{1/2}** (AGOP.md §4), floor on all Adam runs | keep, cite Banach |
| Statistical | **S1b with the optimizer's own null**: g + γw for SGD, SGD+m, Adam+L2; EB_AdamW (§5) for AdamW. MNIST only | fix AdamW branch |
| Representation | **R1**, rate in remaining-change form | keep |
| Combined rule | features settled **and** norm settled (§4.3) | new, to calibrate in Phase 5 |
| γ = 0 | R̂_norm (§4.4) instead of ρ* | settles METRICS §6.5 |

All of these reduce to three published tools: Yaida's stationarity relation (norm),
Banach's a posteriori bound (direction and features), and Mahsereci's evidence test
(noise), each written with the fixed point of the optimizer actually used.

## 8. Checks on existing data (no new training)

1. **Second-order term.** On the pilot telemetry, plot ‖Δw‖²/(2·|⟨Δw, w⟩|) for AdamW
   and SGD runs. Prediction: order 1 or more under AdamW, small under SGD (§2.1a).
2. **ρ* on Adam+L2 grokkers** (replay for v̂). Prediction: ρ* ≈ 1 on the plateau,
   → 0 as the norm levels off, on the same scale as SGD and AdamW.
3. **B2's EB on AdamW pilot runs, last 10% of training.** Prediction: → 0 where the norm
   has settled (§5).
4. **R̂ on real MNIST AGOP traces at the end of memorization.** The key risk (§4.3): does
   R̂ dip below a plausible ε_R there? If yes, the AND-gate with the norm is required,
   not optional.

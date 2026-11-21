# New metric list

This file lists the metrics we agreed on in the redesign discussion. It replaces the old rule grids (about 900 AGOP settings and about 300 rank settings) with a short list of **quantities**: curves measured along each training run.

**Status.** This is the design only. No code has been written for it, and no numbers here have been measured. The numbers marked *back-of-envelope* still need checking on the existing telemetry.

---

## 0. Ground rules that apply to every metric

1. **A metric is a curve, not a stopping rule.**
   - Each metric produces a value at every grid point of every run.
   - Stopping rules (thresholds, patience, lags) are a later, separate step. They are judged by how well they track the curves, not by a stopping score.
2. **One main definition, fixed in advance.**
   - Each metric has exactly one main definition.
   - It has at most one or two *robustness variants*. A variant is used only to check that a conclusion survives a different definition. It is never a candidate to pick from.
3. **Every metric uses the same grid.**
   - All metrics are evaluated at every 500 steps.
   - About 30 extra log-spaced points are added early in the run, snapped to multiples of 50. That makes about 230 points per run.
   - Points not already logged are filled in by deterministic replay from keyframes.
4. **Every metric uses the same smoother.** One symmetric smoother is applied to every quantity, so no metric gets an advantage from its smoothing.
5. **Layers are reported separately, never averaged.** Differences between layers are findings. The mean across layers appears only in summary figures.
6. **The metrics apply to all four optimizers** (AdamW, Adam+L2, SGD+momentum, SGD), unless stated otherwise.

**Notation.**

| Symbol | Meaning |
|---|---|
| w | all weights of the network, flattened into one vector of P numbers |
| w_l | the weights of layer l |
| ‖·‖ | Euclidean length (for a matrix, the Frobenius norm: the square root of the sum of squared entries) |
| lr | learning rate |
| γ | weight-decay coefficient |
| g | gradient of the data loss (no decay term) |
| ḡ | that gradient averaged over the batch (or over the whole training set) |
| t | current step |
| T | the run's total step budget, fixed before training |

---

## 1. Weight norm (N)

| ID | Quantity | Data needed |
|---|---|---|
| **N.1** | log‖w‖, all layers together (main) | telemetry |
| **N.2** | log‖w_l‖ for each layer separately | telemetry |
| **N.3** | ρ̄\*: how fast the norm is changing, in units of what weight decay alone would do (§1.2) | telemetry. Adam+L2 also needs √v̂ from replay |

N.2 shows which layers do the shrinking.

### 1.1 Background: what AdamW does

> **Plain-language note: Adam.** At each step, for each weight separately:
> 1. m ← β₁·m + (1 − β₁)·g. A running average of the gradient (the "momentum"), with β₁ = 0.9.
> 2. v ← β₂·v + (1 − β₂)·g². A running average of the squared gradient, with β₂ = 0.999.
> 3. m̂ = m/(1 − β₁ᵗ) and v̂ = v/(1 − β₂ᵗ). These bias corrections matter only in the first few hundred steps.
> 4. w ← w − lr·m̂/(√v̂ + ε).
>
> Step 4 divides each weight's gradient by that weight's own typical gradient size √v̂. So every weight moves by roughly lr per step, whether its gradient is large or tiny.
>
> **Two ways to add weight decay to Adam:**
> - **Adam + L2 (coupled).** Add γw to the gradient before step 1. The decay is then also divided by √v̂, so weights with tiny gradients get hugely magnified decay. This is why Adam+L2 needs a much smaller γ (1e-6 to 1e-4).
> - **AdamW (decoupled).** First shrink every weight by the same factor: w ← (1 − lr·γ)·w. Then do the ordinary Adam step on g alone. PyTorch's AdamW does exactly this (`param.mul_(1 - lr*weight_decay)`, then the Adam update).
>
> **Numbers.** With lr = 1e-3 and γ = 0.01 the factor is 0.99999 per step. Decay alone would shrink the weights to e⁻¹ ≈ 0.37 of their size every 100k steps.

### 1.2 N.3: definition and derivation

**Step 1: the exact change in one step.** Write w_{k+1} = w_k + Δw_k and expand the square:

> ‖w_{k+1}‖² = ‖w_k‖² + 2⟨Δw_k, w_k⟩ + ‖Δw_k‖²

Call c_k = ‖w_{k+1}‖² − ‖w_k‖² the actual change in squared norm. It is computed directly from two norms, so nothing is approximated.

**Step 2: what decay alone would do in the same step.** Split the step into a decay part d_k and everything else. The *inward pull* of decay is D_k = −⟨d_k, w_k⟩. Decay alone would change ‖w‖² by about −2D_k.

| Optimizer (PyTorch form) | Decay part of the update | D_k |
|---|---|---|
| SGD | w −= lr·(g + γw) | lr·γ·‖w_k‖² |
| SGD + momentum μ = 0.9 | buf = μ·buf + (g + γw); w −= lr·buf | ≈ lr·γ·‖w_k‖²/(1−μ), valid for windows much longer than 1/(1−μ) = 10 steps |
| AdamW | w ← (1 − lr·γ)·w | lr·γ·‖w_k‖², **exact** |
| Adam + L2 | γw passes through the 1/(√v̂ + ε) scaling | ≈ lr·γ·Σ_j w_{kj}²/(√v̂_j + ε), which needs v̂ from replay |

For AdamW: (1 − lr·γ)²·‖w‖² ≈ ‖w‖² − 2·lr·γ·‖w‖². The dropped term, (lr·γ)²·‖w‖² ≈ 1e-10·‖w‖², is negligible.

**Step 3: the ratio in one step.** ρ_k = c_k / (−2D_k), the actual change divided by the change decay alone would cause.

**Step 4: average over a window, because one step is too noisy.** Over steps s, …, t−1, add up the numerators and add up the denominators:

> ρ̄\* = (Σ_k c_k) / (−2·Σ_k D_k)

**Step 5: the numerator telescopes.** Each intermediate norm appears once with + and once with −, so they cancel:

> (‖w_{s+1}‖² − ‖w_s‖²) + (‖w_{s+2}‖² − ‖w_{s+1}‖²) + … + (‖w_t‖² − ‖w_{t−1}‖²) = ‖w_t‖² − ‖w_s‖²

That gives the definition:

> **ρ̄\*(s, t) = (‖w_s‖² − ‖w_t‖²) / (2·Σ_{k=s}^{t−1} D_k)**

In words: **how much the squared norm actually fell, divided by how much decay alone would have made it fall.** It equals a weighted average of the per-step ρ_k, with weights D_k.

**What the values mean** (the same for every optimizer):

| ρ̄\* | Meaning |
|---|---|
| 1 | The data has no net effect on the norm: pure decay. This is the Omnigrok plateau. |
| 0 | The norm is stationary. This is Yaida's fluctuation–dissipation condition, E‖w_{k+1}‖² = E‖w_k‖². |
| < 0 | The norm is growing (Type A late collapse). |

**Worked example** (AdamW, lr = 1e-3, γ = 0.01, ‖w‖² ≈ 400):
- Each step D_k = 1e-5 × 400 = 0.004. Over 1,000 steps, 2·ΣD ≈ 8.
- If ‖w‖² fell from 400 to 392: ρ̄\* = 8/8 = **1** (plateau).
- If it stayed at 400: ρ̄\* = **0** (settled).
- If it rose to 404: ρ̄\* = −4/8 = **−0.5** (growing).

**Computing it from what is logged.**
- ‖w‖ is logged every 50 steps, so the numerator is exact.
- ‖w‖ changes by at most a relative 5e-4 over 50 steps, so the denominator is a trapezoid sum, ΣD ≈ lr·γ·50·Σ(logged ‖w‖²), times 1/(1−μ) for momentum. The error is negligible.
- On the grid, ρ̄\* is computed over each grid interval.

**Why the exact form matters.** Study A's N1 dropped the ‖Δw‖² term. Under Adam, ‖Δw‖² ≈ lr²·P·c², which is the same order as 2D_k (THEORY §2.1). At stationarity it equals the outward push that decay balances. So the old version settled at an offset that depends on the optimizer, not at 0. The telescoped form makes no such approximation, because it uses actual norms.

**When N.3 is needed, and when it is not.**
- **Not needed for timing questions** (when does the norm settle relative to t_ε?). N.1's t₉₀ is already dimensionless.
- **Needed for mechanism questions with a predicted value.** Is the plateau pure decay (ρ̄\* ≈ 1 across optimizers and γ)? Does the test-accuracy jump happen when ρ̄\* reaches 0? Is late collapse ρ̄\* < 0? N.1 cannot answer these, because its rate is in units of lr·γ, which differ by orders of magnitude between runs.
- **Needed as the statistic of the AdamW test W** (§7).
- **Undefined at γ = 0.** Use N.1 there.

---

## 2. Rank (R)

| ID | Quantity | Data needed |
|---|---|---|
| **R.1** | Effective rank of the activation covariance (centred, on the training set), each hidden layer (main) | summaries. Early grid points need replay, forward pass only |
| **R.2** | Effective rank of each weight matrix, from its singular values | summaries |
| **R.3** | Effective rank of the AGOP feature matrix M = G^{1/2} (§3) | free once the AGOP is computed |

**Effective rank.** Normalize the eigenvalues (or singular values) to p_i = λ_i/Σλ, and take exp(−Σ p_i log p_i). It counts roughly how many directions carry the variance.

- **R.1 variant:** the participation ratio 1/Σp_i² instead of the entropy form.
- **R.2:** Galanti et al.'s low-rank claim is about weight matrices, so R.2 is the better-grounded test of it.
- **R.3:** the team's ŝ_spiked estimator is a function of this rank, so it is covered here.

---

## 3. AGOP, as in the paper (AP): the hindsight reference

**Definitions.**
- **AGOP:** G_t = (1/n)·Σᵢ J(xᵢ)ᵀ J(xᵢ), computed over the training set at step t. J(x) is the Jacobian of the network's outputs with respect to its input. G is a d × d matrix, symmetric and positive semidefinite (PSD).
- **Feature matrix:** M_t = G_t^{1/2}. This is the paper's choice: its RFM uses s = ½, and the neural feature ansatz says WᵀW ∝ G^{1/2}.
- **Direction:** M̂_t = M_t / ‖M_t‖_F.
  - Only the direction counts. Scaling every weight by c scales G by c^{2D} in a bias-free ReLU net, so ‖M‖ mostly copies the weight norm, which N already measures.
- **Angle between two such matrices:** θ(A, B) = arccos⟨Â, B̂⟩_F.

> **Plain-language note: why θ is between 0° and 90°.** ⟨A, B⟩_F means: flatten both matrices into vectors of d² numbers and take the ordinary dot product. For symmetric matrices this equals tr(AB). The angle is then defined exactly as for vectors.
>
> If both matrices are PSD, the dot product can never be negative:
> 1. Write B = Σᵢ λᵢ vᵢvᵢᵀ, with every λᵢ ≥ 0 (B is PSD).
> 2. Then tr(AB) = Σᵢ λᵢ·(vᵢᵀ A vᵢ).
> 3. Each vᵢᵀ A vᵢ ≥ 0 (A is PSD), so the sum is ≥ 0.
>
> A non-negative cosine means θ ≤ 90°. Example: A = diag(1, 0) and B = diag(0, 1) use completely different input directions, and their angle is exactly 90°. That is the maximum; two AGOPs can never point "opposite" to each other.
>
> This matters only for interpretation: θ always runs from 0° (same features) to 90° (unrelated features), so angles are comparable across runs and tasks.
>
> θ is the distance along the unit sphere, so it is a true distance and obeys the triangle inequality. §4 uses this.

| ID | Quantity | Applies to |
|---|---|---|
| **AP.1** | AGOP alignment cos θ(M_t, M_T), with M_T at the last step (paper eq. 8) | all tasks |

- **Variant:** G instead of G^{1/2}. The paper uses G^{1/2}; the team's proposal uses G.
- **Always:** mask MNIST's dead pixels.
- **AP.1 uses hindsight** (it needs M_T). It is a reference curve for insight and for judging the online versions, never a stopping signal.
- **Circulant deviation is removed** at the user's request.

---

## 4. AGOP, online (AO): causal estimators

These use only the past. Each is judged by how well it tracks AP.1: correlation within a run, and error at t_ε.

| ID | Quantity | Lag |
|---|---|---|
| **AO.1** | θ(M̂_t, M̂_{t/2}): the angle the features have turned since half the current time | proportional to t (log time) |
| **AO.2** | θ(M̂_t, M̂_{t−cT}): the angle turned over the last c·T steps | constant. c = 0.05 (variant 0.02). Undefined for t < cT |

Both are plain angles, not divided by the lag, so they are dimensionless.

**Why these are linked to the paper's measure.** Apply the triangle inequality to M_t, M_s and the final M_T:

> |θ(M_t, M_T) − θ(M_s, M_T)| ≤ θ(M_t, M_s)

So the online angle is a causal **upper bound on how much the paper's alignment can have changed over the window**. This is exact; it needs no assumption. It errs only one way:
- Jitter, or motion not toward M_T, makes the online angle *larger* than the true change. It never makes it smaller.
- So a small online angle guarantees that AP.1 did not move in that window.

**What no online angle can do.** The angle still to go, θ(M_t, M_T), depends on the path ahead, which cannot be observed. The Banach bound (F2-A) and the proposal's Aitken step tried to predict it by assuming the speed decays geometrically. That failed for a known reason: there are two timescales, a fast memorization phase and then a slow drift. They are dropped, not simplified.

**Why two lags.** They are two time resolutions, and they disagree on purpose.
- **AO.1** is scale-free in time. Multiplying lr by k rescales time, and t/2 still maps to t/2. Grok times span 2–3 decades across the corpus, and log time treats each decade alike. Its window grows with t, so late in a run it averages over a long stretch. That also improves signal over noise: a drift grows with the window, while jitter levels off.
- **AO.2** resolves uniform time. Late in a run it sees recent motion that AO.1's long window dilutes. It tracked the true feature change best in F2 (variant F2-C: within-run rank correlation 0.79 on modular, 0.81 on MNIST 3-layer).
- Using T is not cheating: T is fixed before training starts.

**Jitter level.** It comes free as AO.2 at its shortest lag, one grid step. On MNIST and on every Adam run this noise floor is part of the definition (THEORY §4.2).

**Dropped online variants.**

| Dropped | Reason |
|---|---|
| MSD exponent (F2-D), Geweke (F2-E) | Both tried to tell drift from jitter. The bound above already makes jitter fail safe. Directedness can be measured in Stage 1 with hindsight: the ratio of the AP.1 change to the AO angle over the same window. |
| ŝ_proj (the team's proposal) | It is θ to a fixed anchor at memorization: the same function with a third reference point. Mention it in the write-up; don't compute it. |
| Banach bound (F2-A), Aitken extrapolation | Closed negative results. Keep the existing figures. |

**What complexity is left** is computation only: d_out backward passes per grid point, plus one d × d eigendecomposition to get G^{1/2}. AP.1, AO.1, AO.2 and R.3 all come from the same AGOP trace.

---

## 5. HTSR α (H)

| ID | Quantity | Data needed |
|---|---|---|
| **H.1** | Power-law exponent α of the eigenvalues of WᵀW, each eligible layer. Maximum-likelihood fit, with the tail cutoff x_min chosen by KS distance | summaries |

- **Always reported with α:** fit quality (KS distance, number of tail points). Poor fits are masked.
- **Variant:** a fixed tail fraction instead of the KS-chosen cutoff. Jumps in the cutoff are the usual cause of jumps in α.
- **Eligible layers:** those with enough eigenvalues. That excludes the output layers; on teacher–student almost nothing qualifies.
- **Correlation traps are out.** They need replay, and the pilot found them early on healthy runs.

---

## 6. S1b for SGD, SGD+momentum and Adam+L2 (S)

### 6.1 What S1b tests

S1b asks whether the step still has a **consistent push** in some direction, or is only **random wobble** around a stationary point.

For each weight coordinate k:

> z_k = (ḡ_k + γ·w_k) / (σ̂_k/√B)

- The numerator is the mean regularized gradient.
- The denominator is its standard error. σ̂_k is the spread of the per-example gradients and B is the number of examples averaged.

If there is no consistent push, each z_k² averages 1, so Σ_k z_k² ≈ P, give or take √(2P). The statistic is

> EB = 1 − (1/P)·Σ_k z_k²

EB > 0 means the gradient is smaller than its own noise: stop.

| ID | Quantity | Data needed |
|---|---|---|
| **S.1** | EB on ḡ + γw (main), over all parameters and for each layer. Coordinates with σ̂ ≈ 0 are masked | replay (per-example gradients) |
| **S.2** | EB on g alone (= B2) | the same replay pass, so free |

- **The gap between S.1 and S.2** is the part of the signal that comes from weight decay.
- **Adam+L2 is valid here:** a positive per-coordinate scaling does not change where the gradient is zero.
- **AdamW is not covered by S1b.** See §7.

### 6.2 Which B to use on each task

| Task | Training | B |
|---|---|---|
| MNIST | minibatch of 200 | 200, the step's own minibatch |
| Modular | full batch | n_eff = n/(1 − n/N), the finite-population correction (§6.3) |
| Teacher–student | full batch | n, with the small-sample correction (§6.3) |

**Why S1b still makes sense on full-batch tasks.** Full-batch training has no minibatch randomness. Instead, treat the training set as a random sample of all the task's possible examples. Then the training-set gradient ḡ estimates the "true" gradient over all examples, with standard error σ/√n. The question becomes: is the regularized gradient bigger than its own uncertainty, given the finite data?

- This is a principled tolerance.
- It no longer describes the optimizer's noise, because full-batch GD is deterministic.

### 6.3 The two corrections

The formula σ/√n assumes the examples are drawn independently from an **infinite** population, and that σ is **known**. Each full-batch task breaks one of these assumptions.

> **Plain-language note: modular, the population is finite.** There are only N = 97² = 9,409 possible pairs (x, y), or 97 × 96 = 9,312 for x/y. The training set is drawn from them without repeats.
>
> The extreme case shows the problem. If you trained on all 9,409 pairs, your average *would be* the true average, with zero uncertainty, yet σ/√n would still report some.
>
> **The fix** is the standard finite-population correction: the variance of the sample mean is (σ²/n)·(1 − n/N). At a training fraction of 0.5 the variance is halved, which is the same as having n_eff = n/(1 − n/N) ≈ 2n independent examples. Using n instead would overstate the uncertainty, and the test would say "stop" too easily.

> **Plain-language note: teacher–student, σ is estimated from few examples.** The inputs are Gaussian, so the population is infinite. But n can be as small as 50, and σ̂_k is estimated from those same 50 examples.
>
> A σ estimated from 50 numbers is sometimes too small by chance, which makes z_k² too large on average. Under "no push", z_k follows a t distribution with n − 1 degrees of freedom, so E[z_k²] = (n−1)/(n−3), not 1:
> - n = 50: 49/47 ≈ 1.043
> - n = 400: about 1.005
>
> Without a fix, Σz_k² ≈ 1.043·P even at a perfect stationary point, so EB sits near −0.043 and never fires.
>
> **The fix:** replace the 1 with the correct null average:
> EB = (n−1)/(n−3) − (1/P)·Σ_k z_k²

Both corrections assume the coordinates are roughly independent of each other. S1b already makes that approximation.

**Caveat.** At γ = 0 on teacher–student, both the gradient and σ̂ go to 0 as the fit becomes exact, so z_k is 0/0 there. Flag those points; don't interpret them.

---

## 7. AdamW: the S1b equivalent (W)

### 7.1 Why S1b cannot see the AdamW plateau

> **Plain-language note: the one-step signal is far too small.**
> - **The wobble.** Adam divides each coordinate by its own typical size √v̂. So in every coordinate the random part of the step is about lr × 1, always.
> - **The push.** AdamW applies decay outside that division. In coordinate k it is lr·γ·w_k. A typical weight is |w_k| ≈ ‖w‖/√P = 20/√200,000 ≈ 0.045, so at γ = 0.01 the push is about **0.00045** in wobble units (0.0045 at γ = 0.1).
> - **Summed over all coordinates,** the push adds Σ(γ·w_k)² = γ²‖w‖² = (0.01 × 20)² = **0.04** to Σz_k². That sum is about P = 200,000 and wobbles by √(2P) ≈ **630** (MNIST 3-layer).
>
> 0.04 against 630 is invisible: like weighing 0.04 g on a scale whose reading jumps by 630 g. So any per-coordinate test at one step, whatever its null, reduces to B2.
>
> **Why SGD doesn't have this problem.** For SGD the wobble is in gradient units, σ_k/√B. Once the training set is memorized, the per-example gradients shrink, so σ_k shrinks, while γ·w_k does not. Measured against a shrinking wobble, the decay becomes large and the plateau shows. AdamW's normalization keeps the wobble at about 1, which hides the decay.

### 7.2 Why adding up steps works, but only along ŵ

> **Plain-language note: the biased coin.** A coin that lands heads 50.05% of the time looks fair on one flip. Over K flips, the excess of heads grows like K while the random fluctuation grows like √K, so with enough flips the bias always shows. Adding up K steps works the same way: the push grows like K, the wobble like √K.
>
> **Option 1: every coordinate separately.** The excess of Σz² over P becomes K × 0.04. Beating 630 at 2σ needs K × 0.04 > 2 × 630, so **K ≈ 31,500 steps**. Far too slow.
>
> **Option 2: one number per step, the component of the step along ŵ = w/‖w‖.** This is the only part of the step that changes the norm.
> - **Push:** all P decay pushes point along w, so they add up coherently: Σ_k γ·w_k·ŵ_k = γ‖w‖ = 0.01 × 20 = **0.2** per step (in units of lr).
> - **Wobble:** the random parts (about 1 each, random signs) are combined with weights ŵ_k whose squares sum to 1. So the wobble along ŵ is also about **1**, not √P.
> - **Over K steps:** 0.2K > 2√K gives √K > 10, so **K ≈ 100 steps**.
>
> Option 2 keeps the one direction where decay acts and throws away P − 1 directions of pure wobble, a gain of about √P.
>
> **Caveats (back-of-envelope).** "Wobble ≈ 1 per coordinate" could be anywhere from 0.2 to 1. Steps are assumed roughly independent over long windows; momentum doesn't change the √K growth there. THEORY §5b estimated "a few thousand" steps. Read the result as "hundreds to thousands, not tens of thousands", and check it on the telemetry.

The radial statistic is exactly ρ̄\* (§1.2). So the AdamW test is a test on ρ̄\*.

### 7.3 The test

| ID | Quantity | Role |
|---|---|---|
| **W.1** | Equivalence test on ρ̄\* over the window [t/2, t] (main) | the AdamW stationarity signal |
| **W.2** | EB with AdamW's own null (THEORY §5, Option A) | control: predicted to match B2 |

**W.1 procedure.**
1. Split the window [t/2, t] into m = 10 equal pieces. The pieces are built from the 50-step telemetry, so the window must be at least 500 steps long (t ≥ 1,000).
2. Compute ρ̄\* in each piece with the formula of §1.2. That gives 10 numbers.
3. Form the 95% interval: mean ± 2.26 × sd/√10. The 2.26 is the t-value for 9 degrees of freedom.
4. **"Norm settled" if and only if the whole interval lies inside (−ε, ε)**, with ε = 0.1. That means "the norm moves at less than 10% of the rate decay alone would cause".

> **Plain-language note: why an equivalence test.** The usual test assumes "no drift" (ρ̄\* = 0) and concludes "settled" whenever the data cannot reject that. It fails in two opposite ways:
> 1. **Noisy or short window.** The norm is still drifting but the wobble is large, so "no drift" can't be rejected and you wrongly say "settled". Absence of evidence is not evidence of absence.
> 2. **No noise at all** (full batch: modular, teacher–student). Even a negligible drift, say 0.1% of decay's pull, is "significant", so you never say "settled".
>
> The equivalence test reverses the burden of proof: say "settled" only when you are confident the drift is small.
>
> | 95% interval for ρ̄\* | Verdict | Why |
> |---|---|---|
> | [0.7, 1.3] | not settled | clearly on the plateau |
> | [−0.6, 0.8] | not settled | too noisy to tell (fixes failure 1) |
> | [−0.03, 0.05] | settled | confidently small |
> | [0.001, 0.002] (full batch) | settled | tiny drift is accepted (fixes failure 2) |
>
> **Why pieces and not single steps.** Consecutive steps are correlated through momentum, so the per-step spread would give an interval that is too narrow. Pieces hundreds of steps long are close to independent. This batch-means approach follows SASA (Lang et al. 2019; cited from memory, not yet verified).

**What W.1 can and cannot see.**
- It tests the norm only, not the whole weight vector, so it is blind to rotation at constant norm. The online AGOP (§4) covers that.
- It is a statistically calibrated version of N.3, not new information.
- It works for every optimizer with γ > 0, so it can also run on SGD-family runs as a cross-check against S.1.
- It is undefined at γ = 0.

**W.2** costs almost nothing extra: it uses the same replay pass as S.2. If it matches B2 as predicted, that confirms §7.1.

---

## 8. Summary table

| Family | ID | Quantity | Variants | Hindsight? |
|---|---|---|---|---|
| Weight norm | N.1 | log‖w‖ | – | no |
| | N.2 | log‖w_l‖ per layer | – | no |
| | N.3 | ρ̄\* (windowed, telescoped) | – | no |
| Rank | R.1 | activation effective rank per hidden layer | participation ratio | no |
| | R.2 | weight-matrix effective rank | – | no |
| | R.3 | effective rank of G^{1/2} | – | no |
| AGOP (paper) | AP.1 | cos θ(M_t, M_T) | G instead of G^{1/2} | **yes** (reference only) |
| AGOP (online) | AO.1 | θ(M̂_t, M̂_{t/2}) | – | no |
| | AO.2 | θ(M̂_t, M̂_{t−cT}), c = 0.05 | c = 0.02 | no |
| HTSR | H.1 | α per eligible layer (MLE, KS x_min) + fit quality | fixed tail fraction | no |
| S1b (non-AdamW) | S.1 | EB on ḡ + γw; B = 200 / n_eff / n with corrections | – | no |
| | S.2 | EB on g (B2) | – | no |
| AdamW | W.1 | equivalence test on ρ̄\*, [t/2, t], m = 10, ε = 0.1 | – | no |
| | W.2 | EB with AdamW's null (control) | – | no |

## 9. Cost

| Cost | Metrics |
|---|---|
| Free (telemetry and summaries, all 1,664 runs) | N.1, N.2, N.3 (not Adam+L2), R.1 (500-step points), R.2, H.1, W.1 |
| Cheap replay | early R.1 points (forward pass); √v̂ for N.3 on Adam+L2 |
| Moderate replay | S.1, S.2, W.2 (per-example gradients) |
| Expensive replay | one AGOP trace per run (d_out backward passes per grid point). AP.1, AO.1, AO.2 and R.3 all come from it. On all runs it costs about 7× the F2 study's replay |

## 10. Open decisions

1. **AGOP scope:** all 1,664 runs, or a larger stratified subset. Decide once the F2 replay wall time is known.
2. **c for AO.2:** 0.05 (variant 0.02) is proposed.
3. **Free checks before building anything:**
   - B2's EB on the pilot's AdamW runs. It should rise toward 0 where the norm has settled.
   - ρ̄\* on study A's telemetry, to check the K estimate in §7.2 and see whether the plateau value is near 1.

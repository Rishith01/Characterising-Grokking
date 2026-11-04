# Metrics: candidates and baselines

**Status (2026-11-03): proposed, not confirmed; partly built.** Built in `stopping/`
(DESIGN.md §13): N1, R1 and B0–B6 (B3 in both forms, B5 with traps). Held back: S1, S1b,
F2. This document details the reduced
metric set proposed in [PLAN.md](PLAN.md) §5 ("Proposed reduced set"). The user has not
confirmed it yet. Phase 4 builds whatever is confirmed. Open decisions are collected in
§5 of this document.

Each metric produces a **signal**: a time series computed causally from a run's stored
data or from replay. A shared rule (causal smoothing, then a threshold with a direction,
then a patience window; [DESIGN.md](DESIGN.md) §7.3) turns a signal into a stopping
step. Rule hyperparameters are calibrated on seeds {0, 1} and scored on seeds {2, 3}.
Test data is never seen by any signal; it only scores the stop (regret).

**Citations.** † means cited from memory and not yet checked against the paper; verify
the claim and the arXiv number before relying on it. Only Omnigrok and Deep Grokking are
among the local PDFs (repo root).

---

## 1. Shared notation and shapes

| | Network (each arrow is one linear layer) | Parameters P | Training points n | Batch B | Loss |
|---|---|---|---|---|---|
| Modular (p = 97) | 194 → 512 → 512 → 97, ReLU | 412,257 | 4,704 (x/y: 4,656) | full batch | cross-entropy |
| MNIST depth 3 | 784 → 200 → 200 → 10, ReLU | 199,210 | 1,000 | 200 | MSE, one-hot targets |
| MNIST depth 8 | 784 → 400 (×7) → 10, ReLU | 1,284,410 | 4,000 | 200 | MSE, one-hot targets |
| Teacher–student | 5 → 100 → 100 → 5, tanh | 11,205 | 50 / 100 / 400 | full batch | MSE |

Modular inputs are two concatenated one-hot vectors (a in the first 97 slots, b in the
last 97).

**Why MNIST uses MSE.** It copies Omnigrok's MNIST setup (§3 of the paper: depth-3,
width-200 ReLU MLP, AdamW, 1k samples, large init, "MSE loss with one-hot targets").
Omnigrok's Appendix E answers the obvious objection. With cross-entropy, MNIST still
groks, but less cleanly:
- the Goldilocks zone is broader, so it needs α ≈ 100 instead of about 8;
- test accuracy first stalls at 30–40% before the late rise.

The choice is deliberate and should be stated in the write-up.

**Notation.**
- w_l is layer l's parameters (weight and bias) flattened, length P_l. w is all of them,
  length P.
- g = ∇L(w) is the data-loss gradient. On MNIST it is a minibatch gradient (noisy); on
  modular and teacher–student it is the full-batch gradient.
- γ is the weight decay. The regularized objective is J(w) = L(w) + (γ/2)‖w‖², so
  ∇J = g + γw.
- Δw = w_{t+1} − w_t is the step actually taken. lr is the learning rate.
  lr_eff = lr / (1 − β) for SGD with momentum β, and lr otherwise.
- ŵ = w / ‖w‖ is the unit vector along w.

**Data sources** (DESIGN.md §5):

| Source | Cadence | Holds |
|---|---|---|
| Telemetry | every 50 steps | per layer: `w_norm`, `g_norm` (data term), `reg_g_norm` (coupled L2 term), `update_norm`, `update_radial`, `update_tangential`, `cos_g_w`; plus `train_loss`, `train_acc`, `lr` |
| Train predictions | every 50 steps | argmax on the training set (int16); teacher–student stores per-sample RMS error (float16) |
| Dense summaries | every 500 steps | `sv/l{i}`: singular values of W_l. `actspec/l{i}`: eigenvalues of the centered training-set activation covariance of hidden layer i. `gnoise/l{i}`: [‖mean g‖², tr Cov(g), mean ‖g‖²] over 8 fixed probe minibatches of 256 |
| Replay | any log step (default eval grid: union of 200 log-spaced and 400 uniform steps) | exact weights and optimizer state, so anything: per-example gradients, Jacobians, activations |

Replay only moves forward from the last keyframe at or before t, so it never uses
future information.

---

## 2. Why these signals

On a grokking plateau, the training loss is flat while generalization has not happened
yet. Rules that watch train loss, train predictions or the data gradient (B1, B2, B4)
are therefore expected to stop on the plateau. Each candidate below measures something
that **keeps moving during the plateau and goes quiet only once the network settles**:
- the weight norm (N1, S1 radial);
- the direction of the weights (S1 tangential);
- the regularized gradient against noise (S1b);
- the learned input features (F2);
- the dimension of the representation (R1).

**Shared weakness: all five are stationarity signals.** On a **late-collapse** run, test
accuracy peaks *before* anything settles, so every candidate stops late by construction
and pays regret. This is a known limitation, not a bug to fix per metric.

---

## 3. Candidates

### N1: weight norm (reference)

**Computation.**
- ‖w‖ = √(Σ_l ‖w_l‖²), from telemetry `l{i}_w_norm`, every 50 steps.
- Rules use its trend, not its level: the relative change over a window,
  (‖w‖_t − ‖w‖_{t−Δ}) / ‖w‖_t.

**Intuition.** In Omnigrok, grokking is the weight norm shrinking under weight decay,
from the large norm set by the init scale α down to the Goldilocks zone where
generalizing solutions live. The plateau *is* that shrinking, and generalization arrives
as the norm levels off.

**Caveats.**
- Its level means nothing; it is scale-dependent across tasks and α.
- Its rate of change depends on lr and γ (see the normalization note under F2).
- At γ = 0 there is no shrinking mechanism, so N1 may drift or grow.
- It is the crudest version of S1's radial part, and is kept as a basic reference.

**Source.** Liu, Michaud, Tegmark, *Omnigrok: Grokking Beyond Algorithmic Data*, ICLR
2023, arXiv:2210.01117 (local PDF).

### S1: fixed-point residual, split into radial ρ and tangential τ

**Idea.** With weight decay, training does not converge to ∇L = 0. It converges to the
optimizer's fixed point, where the data gradient and the decay balance. S1 measures the
distance from that balance, split into a part along w (does the norm still change?) and
a part across w (does the direction still change?).

**Computation, per layer l, every 50 steps, from telemetry.**

*Recommended form: from the step actually taken, the same for every optimizer:*
- **ρ_l = −⟨Δw_l, w_l⟩ / (lr_eff · γ · ‖w_l‖²)** = −`update_radial` / (lr_eff · γ · `w_norm`)
- **τ_l = ‖Δw_l,⊥‖ / (lr_eff · γ · ‖w_l‖)** = `update_tangential` / (lr_eff · γ · `w_norm`)

*Gradient form, which is the same thing for plain SGD (and SGD + momentum at steady state):*
- r = g + γw. Its part along ŵ is ⟨r, ŵ⟩ = ‖g‖ cos θ + γ‖w‖, where θ is the angle
  between g and w. Dividing by γ‖w‖ gives
  **ρ = 1 + ⟨g, w⟩ / (γ‖w‖²)** = 1 + `g_norm` · `cos_g_w` / (γ · `w_norm`).
- γw points exactly along w, so it drops out of the part across w:
  **τ = ‖g‖ sin θ / (γ‖w‖)** = `g_norm` · √(1 − `cos_g_w`²) / (γ · `w_norm`).

*Layer aggregation (open):* use all parameters together, the maximum over layers, or a
single layer.

**Derivation of ρ.** Under gradient flow, ẇ = −lr·(g + γw). Using the chain rule twice:

d log‖w‖/dt = (1/‖w‖)·d‖w‖/dt = (1/‖w‖²)·d(½‖w‖²)/dt = (1/‖w‖²)·⟨w, ẇ⟩
= −lr·(⟨g,w⟩/‖w‖² + γ) = **−lr·γ·ρ**

So ρ is the norm's log-decay rate measured in units of the pure weight-decay rate lr·γ.

**This only holds for (momentum) SGD.** Adam's per-coordinate preconditioner means the
step is not −lr·(g + γw), and AdamW decouples the decay. The step-based form measures
⟨Δw, w⟩ directly, so it keeps the meaning "per-step change in log‖w‖ over lr_eff·γ" for
every optimizer. For a finite step, ‖w_{t+1}‖² = ‖w_t‖² + 2⟨Δw, w_t⟩ + ‖Δw‖², and the
second-order term is negligible at our step sizes.

**Intuition: ρ.**
- **ρ ≈ 1:** the data gradient has no radial pull, so the norm decays at the pure
  weight-decay rate. This is the Omnigrok plateau: training data is memorized, and the
  norm is still sliding toward the Goldilocks zone. Train loss is flat but ρ is not 0,
  so a rule waiting for ρ → 0 does not stop here.
- **ρ ≈ 0:** the data gradient pushes outward exactly as hard as the decay pulls inward,
  so the norm has stopped changing.
- **ρ < 0:** the norm is growing.

**Intuition: why τ is needed.** ⟨g, w⟩ only sees the part of g along w. ρ = 0 only
requires that radial part to equal −γ‖w‖, and the part across w can still be any size.
So the norm can be balanced while the direction is still rotating, i.e. features are
still forming.

Structurally, for a bias-free ReLU net of depth D, scaling w by c scales the output by
c^D, and Euler's identity gives ⟨∇_w f, w⟩ = D·f. So the radial part of g only changes
the **output scale**, and everything that changes the **shape of the function** lives
in the tangential part. Biases make this approximate. The fixed point needs both ρ ≈ 0
and τ ≈ 0.

**Caveats.**
- **At γ = 0, both are undefined** (division by zero). A fallback is needed, for
  example normalizing by ‖g‖ or ‖Δw‖ instead.
- **Noise floor on MNIST.** g is a minibatch gradient, so τ (and ρ) never reach 0. They
  settle at a noise level set by the batch size. S1b handles this properly; for S1
  itself the rule threshold must sit above the floor.
- The normalizations (by lr_eff·γ·‖w‖) are chosen to give ρ its rate meaning. τ's
  normalization is a choice, not a derivation.
- Layer aggregation is undecided.
- ρ can be non-monotone (overshoot, slingshot spikes), so it needs smoothing and a
  patience window.

**Sources.** There is no single paper that proposes S1 as a stopping signal; it is
constructed here from:
- first-order stationarity of the regularized objective, ∇J = 0 (standard optimization);
- the norm-shrinking picture of grokking: Omnigrok, arXiv:2210.01117. Omnigrok does
  not define ρ; the derivation above is ours;
- the equilibrium between weight norm and weight decay, and the radial/tangential
  split, from the scale-invariance literature:
  - van Laarhoven 2017, *L2 Regularization versus Batch and Weight Normalization*,
    arXiv:1706.05350 †
  - Wan et al. 2021, *Spherical Motion Dynamics* †
  - Kosson et al. 2024, *Rotational Equilibrium: How Weight Decay Balances Learning
    Across Neural Networks* (covers AdamW) †
- AdamW's fixed point: Xie & Li, *Implicit Bias of AdamW: ℓ∞ Norm Constrained
  Optimization*, ICML 2024, arXiv:2404.04454 †.

### S1b: evidence test on the regularized gradient

**The original metric (B2).** Mahsereci et al. 2017 test whether the minibatch gradient
is distinguishable from sampling noise.
- **Model:** by the central limit theorem, ∇L_S ~ N(∇L_D, Σ/B), with Σ diagonal.
- **Null hypothesis:** the expected gradient is zero, i.e. training is at a stationary
  point and what remains is noise.
- **Statistic:** compare the log-likelihood of the observed gradient under the null with
  its expected value under the null, normalized by the number of parameters. This gives
  - **EB = 1 − (B/P) · Σ_k ĝ_k² / σ̂_k²**, stop when EB > 0, after smoothing (EB is
    noisy step to step).

**Our change (S1b):** apply the same test to ∇J = g + γw instead of g.

**Computation, per eval step, during replay.**
1. Per-example data-loss gradients g_i, for the B examples of the batch (MNIST: the
   minibatch that step t uses, which is a pure function of t; full-batch tasks: all n),
   with `torch.func.vmap(torch.func.grad(...))`. The full matrix is B × P, too big for
   modular (4,704 × 412k ≈ 1.9e9 numbers), so accumulate only Σ_i g_i and Σ_i g_i² per
   coordinate, in chunks of examples.
2. ḡ_k = (1/B) Σ_i g_ik; σ̂_k² = (1/(B−1)) Σ_i (g_ik − ḡ_k)².
3. ĝ_k = ḡ_k + γ w_k. γw is deterministic, so it adds nothing to the variance.
4. Mask coordinates with σ̂_k² ≈ 0 (dead ReLU units, unused one-hot input rows) and let
   P be the number of coordinates that remain.
5. **EB = 1 − (B/P) · Σ_k ĝ_k² / σ̂_k²**. Report it overall and per layer.

**Why the B.** ĝ_k is a mean of B per-example gradients, so its standard error is
√(σ_k²/B). z_k = ĝ_k / √(σ̂_k²/B) counts standard errors from zero, and z_k² =
B·ĝ_k²/σ̂_k². Under the null, ĝ_k ~ N(0, σ_k²/B), so z_k² ~ χ²₁ with mean 1. The sum
over P coordinates is then about P, so EB = 1 − (1/P)·Σ z_k² ≈ 0.
- **EB ≪ 0:** a real gradient signal remains, so keep training.
- **EB > 0:** the gradient is no bigger than noise, so stop.

**Intuition: why ∇J and not g.** On a grokking plateau the data gradient g is nearly
pure noise around the memorized solution, so B2 says "stop" on the plateau (the
predicted failure). But γw is large and coherent there, so EB on ∇J stays strongly
negative until the norm settles. It asks S1's question ("is the regularized objective
stationary?") with a statistical test instead of a hand-picked threshold, and it handles
the MNIST noise floor that S1 struggles with.

**Caveats.**
- **Full-batch tasks** (modular, teacher–student) have no minibatch noise. With B = n,
  the per-example spread acts as fictitious noise, so the test is a heuristic there,
  not the paper's statistical test. Options (open): use S1b only on MNIST, or evaluate
  it at a nominal batch size.
- At γ = 0, S1b is exactly B2.
- Diagonal covariance is an approximation; correlated coordinates make the χ²
  calibration off.
- Cost: one per-example-gradient pass per eval step. The most expensive candidate after
  F2 for MNIST depth 8.

**Sources.**
- Mahsereci, Balles, Lassner, Hennig, *Early Stopping without a Validation Set*, 2017,
  arXiv:1703.09580 (the EB criterion; no regularizer in the paper).
- Applying it to ∇J is our extension.

### F2: change in the input AGOP

**Computation, per eval step, during replay.**
1. f: ℝ^{d_in} → ℝ^{d_out} is the network output (logits for modular). J(x) ∈
   ℝ^{d_out × d_in} is its input Jacobian at x.
2. **AGOP:** M = (1/n) · Σ_{i=1..n} J(x_i)ᵀ J(x_i), over the training set. M is
   d_in × d_in: 194 × 194 (modular), 784 × 784 (MNIST), 5 × 5 (teacher–student). For
   modular it splits into an a-block, a b-block and their cross terms.
3. **Efficient form.** M = (1/n) · Σ_c Σ_i ∇_x f_c(x_i) ∇_x f_c(x_i)ᵀ. Example i's
   output depends only on x_i, so backpropagating Σ_i f_c(x_i) to the input matrix X
   gives all n per-example input gradients for output c in one backward pass. So M costs
   **d_out backward passes**: 97 (modular), 10 (MNIST), 5 (teacher–student). With
   G_c ∈ ℝ^{n × d_in} the result for output c, M = (1/n) Σ_c G_cᵀ G_c.
4. **Change between eval steps t and t − Δ.** Two forms:
   - raw: F2 = ‖M_t − M_{t−Δ}‖_F / ‖M_t‖_F, or the alignment version
     1 − ⟨M_t, M_{t−Δ}⟩_F / (‖M_t‖_F ‖M_{t−Δ}‖_F), which ignores rescaling;
   - **self-normalized (proposed): F2 = (‖ΔM‖_F/‖M‖_F) / (‖Δw‖/‖w‖)**, the relative
     feature change per relative weight change over the same interval.

**Intuition.** The neural feature ansatz says the first layer's W₁ᵀW₁ tracks a power of
the AGOP, so M is the feature matrix the network has learned. In grokking, generalization
coincides with M converging to the structured matrix the task needs: for modular
arithmetic, a circulant / Fourier structure; for teacher–student, the teacher's input
subspace. On the plateau, train loss is flat but **M is still rotating** while those
features form. F2 → 0 means the features are done. It tracks the mechanism directly
instead of going through the norm.

**Caveats.**
- **Learning-rate normalization.** The raw form scales roughly with the step size over Δ
  steps. Our learning rates span orders of magnitude between the SGD and Adam families,
  so one threshold would mean different things in different runs. Fixes:
  - (a) divide by lr_eff·γ·Δ (change per unit of weight-decay time). Breaks at γ = 0 and
    needs an effective lr for Adam.
  - (b) **self-normalize by weight motion** (proposed). It does not depend on lr, the
    optimizer or γ, so γ = 0 works. Weakness: when the weights barely move it becomes
    0/0 and noisy, so it needs a floor on ‖Δw‖/‖w‖.
- **MNIST jitter.** Minibatch noise makes M jitter, so F2 has a noise floor. A reference
  is ‖ΔM‖ between two nearby checkpoints.
- The eval grid is not uniform, so Δ varies. Rules must use per-interval rates, not raw
  differences.
- Only the input AGOP is proposed. Per-layer AGOPs are possible but cost more.

**Sources.**
- Radhakrishnan, Beaglehole, Pandit, Belkin, *Mechanism for feature learning in neural
  networks and backpropagation-free machine learning models*, Science 2024,
  arXiv:2212.13881 † (neural feature ansatz).
- Mallinar et al., arXiv:2407.20199 (AGOP and grokking in RFM and MLPs).

### R1: effective rank of the activation covariance

**Computation, per hidden layer l, every 500 steps, from summaries (no replay).**
1. H_l ∈ ℝ^{n × d_l}: layer l's activations on the training set. d_l = 512 (modular),
   200 or 400 (MNIST), 100 (teacher–student).
2. Center the columns; the covariance C_l = (1/n) H̃ᵀH̃ is d_l × d_l.
3. Its eigenvalues λ_1 … λ_{d_l} are stored as `actspec/l{i}` (computed as squared
   singular values of H̃ divided by n, in float64).
4. p_k = λ_k / Σ_j λ_j.
5. **erank_l = exp(−Σ_k p_k log p_k)** (exponential of the spectral entropy). Lies in
   [1, d_l] and roughly counts the directions the layer actually uses.
6. Rules use the relative change of erank over a window. A finer time resolution is
   possible from replay (one forward pass on the training set per eval step).

**Intuition.** Weight decay has a low-rank bias: it removes directions the fit does not
need. A memorizing solution spreads energy over many directions; a generalizing one is
compact (modular arithmetic needs only a few Fourier frequencies). Deep Grokking reports
rank falling together with generalization. On the plateau erank is still falling; a rule
waits for it to flatten. It uses training data only.

**Caveats.**
- Same lr-normalization question as F2 for rate-based rules.
- 500-step resolution may be coarse for fast runs (t_ε of ~1k on MNIST depth-8 α = 1).
- erank depends on the activation; teacher–student uses tanh, whose saturation changes
  the spectrum.
- Not monotone in general: rank can drop then rise (e.g. during late collapse).
- Layer aggregation is undecided.

**Sources.**
- Roy & Vetterli, *The effective rank: a measure of effective dimensionality*, EUSIPCO
  2007 (definition).
- Galanti et al., *SGD and Weight Decay Secretly Minimize the Rank*, arXiv:2206.05794 †.
- Fan et al., *Deep Grokking*, arXiv:2405.19454 (local PDF).

---

## 4. Baselines

| ID | Method | Data | Expected behaviour on grokking |
|---|---|---|---|
| B0 | Fixed budget | none | ignores the dynamics |
| B1 | Training-loss patience | telemetry | stops on the plateau |
| B2 | Mahsereci EB on g | replay | stops on the plateau |
| B3 | Gradient disparity | summaries (cheap) or replay | unclear |
| B4 | Label Wave | train predictions | stops on the plateau |
| B5 | HTSR α and correlation traps | summaries | main competitor |
| B6 | Validation-loss patience | **held-out data** | oracle reference only |

### B0: fixed budget
Stop at T_fixed, a single step calibrated on seeds {0, 1} per dataset (e.g. the step
that minimizes mean calibration regret). The sanity floor: any signal that does not beat
it is useless.

### B1: training-loss patience
Stop when the causally smoothed (EMA) train loss has not improved by a relative δ for
k consecutive log steps (δ and k calibrated). Telemetry, every 50 steps. This is the
textbook rule without a validation set. **Expected to stop on the plateau**, since train
loss is flat there by definition.

### B2: Mahsereci EB (original)
Exactly S1b with γ = 0, i.e. the test on the data gradient g alone, with smoothing and
stop at EB > 0. Same computation, cost and full-batch caveat as S1b.
**Expected to stop on the plateau.** Pilot (2026-11-04): true under SGD, but on
Adam-family runs EB stays negative and B2 never stops; the mean data gradient balances
the decay and is not noise ([results](results/phase4/RESULTS.md)). Source: arXiv:1703.09580.

### B3: gradient disparity
**Original.** Forouzesh & Thiran (arXiv:2107.06665): draw two minibatches, compute
their gradients g₁ and g₂, and use D = ‖g₁ − g₂‖. It rises when the network starts
fitting batch-specific noise (overfitting), and they stop when it has increased for
k consecutive checks.

**Cheap form (proposed).** The summaries already store, per layer, tr Cov(g) over 8
fixed probe minibatches of 256 (`gnoise/l{i}`, second entry). For two independent
batches, E‖g₁ − g₂‖² = 2·tr Cov(g), so √(2·tr Cov(g)) is the expected disparity, every
500 steps, with no replay. The replay form (two fixed minibatches per eval step) is
kept for checking.

**Caveats.**
- For full-batch training the probe batches are still well-defined (256-sample
  subsets), but they don't match the training noise.
- Its behaviour on grokking is not predicted by the paper.

### B4: Label Wave
**Original.** Yuan et al. (ICLR 2024, arXiv:2502.07551, as listed in RAMPUP): track the
**prediction changes** on the training set, i.e. the number of training examples whose
predicted label changed since the previous check. Stop at the first local minimum of
that (smoothed) count, which in their noisy-label setting marks the switch from fitting
clean labels to memorizing noisy ones.

**Ours.**
- From stored train predictions every 50 steps: PC_t = #{i : ŷ_i(t) ≠ ŷ_i(t − 50)}.
  Causal smoothing, then stop at the first trailing local minimum (confirmed by a
  patience window).
- Teacher–student has no labels; use the change in per-sample RMS error instead, i.e.
  the count of samples crossing the θ threshold, or the mean |Δ error|.

**Expected to stop on the plateau**, since train predictions are frozen after
memorization.

### B5: HTSR α and correlation traps (main competitor)
**Computation, per layer, every 500 steps, from `sv/l{i}`.**
1. Eigenvalues of W_lᵀW_l are the squared singular values s_k².
2. Fit a power law p(λ) ∝ λ^{−α} to the tail of this empirical spectral density, by
   maximum likelihood with the x_min chosen by Kolmogorov–Smirnov distance (the
   WeightWatcher method). A layer with α ≈ 2 is "well trained"; α ≫ 2 is under-trained
   (random-like); α < 2 is over-correlated.
3. **Correlation traps:** eigenvalues of the *element-wise randomized* W_l that stick
   out beyond the Marchenko–Pastur bulk. These signal a pathological, "over-trained"
   layer. Computing them needs the full W_l, so it comes from replay or keyframes, not
   the stored singular values.
4. Rule: stop when the mean α over layers enters a band around 2 and stays there (band
   calibrated), and, as a variant, never stop while correlation traps are present.

**Why it is the main competitor.** Prakash & Martin apply HTSR to grokking and
"anti-grokking" (late collapse) and argue α and correlation traps track both, without
test data. If any existing test-free method handles our traps, it is this one.

**Caveats.**
- Power-law fits are unstable on small layers: teacher–student has 100 × 100 matrices,
  and the 97- and 10-row output layers barely have a tail. Restrict to hidden layers
  of width ≥ 200, or report fit quality.
- The papers' thresholds come from larger models.

**Sources.**
- Martin & Mahoney, *Implicit Self-Regularization in Deep Neural Networks*,
  arXiv:1810.01075 †.
- Prakash & Martin, arXiv:2506.04434 and arXiv:2602.02859.

### B6: validation-loss patience (oracle reference)
Classic early stopping on held-out loss: stop when the smoothed validation loss has not
improved for k checks.
- It uses test-side data, so it is **reported separately and never counted as a
  contender**. It shows what a validation set would buy.
- It is computed from the oracle stream, which signals never see.
- Note that on a grokking run it also tends to stop on the plateau, because validation
  loss often *rises* during memorization before it falls.

---

## 5. Late collapse: what it is and what it means for the metrics

Discussed with the user 2026-10-28. Provisional label: final test accuracy is more than
0.1 below the run's best. Counts: MNIST 25 (16 at depth 3, 9 at depth 8), teacher–student
51; modular not yet trained.

### The frame: Omnigrok's LU picture

Omnigrok plots error against the weight norm ‖w‖:
- **Training error is an "L":** low at any norm above some minimum.
- **Test error is a "U":** low only in a band, the **Goldilocks zone**.

Grokking is entering the U from the right: a large init memorizes, then weight decay
shrinks the norm into the zone. **Late collapse is leaving the U**, and our corpus shows
it on both sides. The two types call for different detectors.

| | Type A: drift out on the right | Type B: collapse out on the left |
|---|---|---|
| Where we see it | MNIST depth 3, Adam family, α = 1, γ ≤ 0.03: test peaks at 0.92 around step 100, then decays (γ = 0.1 does not) | MNIST depth 8, SGD, α = 2, γ = 1e-2 (peak 0.66, then constant output); Adam+L2 depth 8, α = 1, γ = 1e-3 (constant output, test 0.114, labelled "never"); teacher–student with coupled γ = 1e-4 (partial loss of fit) |
| Mechanism | Nothing pins the norm, so the data gradient keeps driving train loss down by growing the output scale and over-tuning features to the training points | Decay outpulls the data gradient, the norm shrinks past the zone, and the network falls to the trivial (constant-output) fixed point |
| Train accuracy | stays at 100%: **invisible to B1 and B4** | **falls**, so a train-accuracy guard sees it |
| ρ (S1 radial) | **ρ < 0** (norm growing) | ρ ≈ 1 while the fit degrades |
| Literature | Lyu et al., arXiv:2311.18817 †: *misgrokking* (the late-phase implicit bias generalizes worse than the early one, so test accuracy drops after the transition). Prakash & Martin †: *anti-grokking*, detected without test data by HTSR α < 2 and correlation traps (B5). For cross-entropy (modular), Prieto et al., arXiv:2501.04697 †: after the data is classified, cross-entropy keeps falling by scaling the logits, so the norm grows ("naive loss minimization") until floating-point softmax collapse | The left arm of Omnigrok's U ("norm too small"); no separate name in the papers we use |

### Decay pulls inward at every init; depth makes collapse an attractor

Weight decay adds −γw to every step, whatever the init. In ρ terms, the norm's log-rate
is −lr·γ·ρ with ρ = 1 + ⟨g,w⟩/(γ‖w‖²): the "1" is the decay, and it is always present.
**The init only decides where you start relative to the zone.**
- From the right (large α), the pull is the cure: that is grokking.
- From inside the zone (small α), the pull is harmless only if the data gradient can
  balance it, i.e. ρ = 0 is reachable at a norm inside the zone.

**Depth argument** (our derivation, exact only for bias-free ReLU networks; our MLPs have
biases and teacher–student uses tanh, so treat it as the leading-order picture).
Scale every weight by s in a depth-D homogeneous network:
- the output scales as s^D, so the data push ⟨g,w⟩ = D·⟨∂ℓ/∂f, f⟩ scales roughly as s^D;
- the decay term γ‖w‖² scales as s².

So as s → 0, ρ ≈ 1 + O(s^{D−2})/γ → 1 for D > 2. **At small norm the decay always
wins in a deep network**, and w ≈ 0 (constant output once biases are included) is an
attractor for any γ > 0. A run collapses if decay drags it into that basin before the
data push can hold it. Larger γ and larger D make the basin bigger. That is why the hard
collapses are at depth 8. At depth 3 collapse needs coupled decay, which under Adam+L2
is amplified per coordinate by the preconditioner.

**What the init controls is the shape of a Type B failure:**

| Start | With strong γ |
|---|---|
| Right of the zone (large α) | shrinks, passes **through** the zone (test peaks), overshoots into collapse: *late collapse* |
| Inside or left of the zone (small α) | slides into collapse without generalizing much: *never* |

### Consequences for the metric set

- **All five candidates stop late on late-collapse runs.** They wait for the dynamics to
  settle, and on these runs the damage is done before then. Section 2's shared weakness.
- **Two cheap additions to test in Phase 5:**
  - stop when ρ < 0 persists (norm growing), which should catch Type A early;
  - a train-accuracy guard (stop, or flag, when train accuracy falls from its fitted
    level), which should catch Type B.
- **B5 is the literature's Type A detector**, so the ρ-sign rule is compared with B5
  specifically on Type A runs.
- Phase 5's `GroundTruth` should split late collapse into A and B, using whether train
  accuracy also drops. The current provisional label does not distinguish them.

---

## 6. Open decisions (need the user)

1. **Confirm the set:** N1, S1 (ρ, τ), S1b, F2, R1, plus B0–B6.
2. **ρ definition:** use the step-based form for all optimizers (proposed), since the
   gradient form is wrong for Adam-type optimizers.
3. **F2 normalization:** self-normalize by weight motion (proposed), or by
   lr_eff·γ·Δ.
4. **S1b on full-batch tasks:** MNIST only, or a nominal batch size.
5. **γ = 0 fallback** for S1 (normalize by ‖g‖ or ‖Δw‖).
6. **Layer aggregation** for S1, S1b, R1 and B5 (all parameters, max over layers, or a
   single layer).

Dropped from the original list (PLAN.md §5, "Original candidate list"): S1c (its useful
part is S1's tangential component), R2 (except inside B5), F1 (a cheap proxy for F2,
kept only as a fallback), F3 (no grokking-specific theory), participation ratio and
spectral entropy.

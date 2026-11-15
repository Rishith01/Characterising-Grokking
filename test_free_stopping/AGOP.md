# AGOP, AGOP alignment, and an online version of it

**What this is.** A first-principles explanation of the Average Gradient Outer Product
(AGOP) and of Mallinar et al.'s AGOP alignment, then a step-by-step reading and review
of the online-alignment proposal in `BTP/BTP_Proof_AGOP_online_variant.pdf` ("the
proposal"), and a proposed alternative with its normalization worked out. Written
2026-11-04; §4 rewritten 2026-11-14 as a self-contained derivation. **Status: a
proposal. The user has not decided F2 (§6), and nothing here is built.** Toy simulation behind the numbers in §5:
[`results/phase4/agop_toy_sim.py`](results/phase4/agop_toy_sim.py).

It feeds metric F2 in [METRICS.md](METRICS.md) §3, whose entry now summarizes §4 as the
proposed replacement; the choices in §6 are the user's. The planned implementation is in
DESIGN.md §13.8.

**Sources.**
- Mallinar, Beaglehole, Zhu, Radhakrishnan, Pandit, Belkin, *Emergence in non-neural
  models: grokking modular arithmetic via average gradient outer product*,
  arXiv:2407.20199 (local PDF `Emergence in non-neural models.pdf`). Definition 2.1
  (AGOP), §3 (AGOP alignment, eq. 8), §4 (neural networks, the NFM).
- *Online Estimation of Average Gradient Outer Product Alignment During Grokking*
  (local PDF `BTP_Proof_AGOP_online_variant.pdf`, 14 pages). Equation numbers below are
  that document's.
- Radhakrishnan et al., *Mechanism for feature learning in neural networks*, Science
  2024, arXiv:2212.13881 † (Neural Feature Ansatz; † = from memory, not checked).

**Verdict in four lines.**
1. The proposal's algebra checks out (list in §3.1).
2. Its estimator is exact only on its own model: the AGOP moves on a straight line
   between two fixed matrices, at an exponentially slowing rate from the moment of
   memorization. On a grokking-shaped path it is unusable without the paper's own
   chord-collinearity gate (D_chord). With the gate, it worked on every toy case.
3. The alignment it estimates is fully determined by **one unknown number**: how far the
   AGOP will still travel. That is the real content of the method, and it suggests a
   simpler estimator with no straight-line assumption (§4).
4. That estimator also settles the normalization question: dividing the AGOP's speed by
   the decay rate of that same speed gives a remaining *angle*. The angle is
   dimensionless, so lr, γ, the optimizer, the init scale and the input dimension all
   drop out.

---

## 1. The AGOP from first principles

### 1.1 Definition

Take a network f: ℝ^d → ℝ^c and a training input x. The **input Jacobian**
J(x) ∈ ℝ^{c×d} has entries J_{ℓs} = ∂f_ℓ(x)/∂x_s. It answers one question: if I nudge
the input by a small vector u, how much does the output move? To first order,
f(x + u) − f(x) ≈ J(x)u.

The **AGOP** over the n training points is

> **G = (1/n) Σ_i J(x_i)ᵀ J(x_i)** ∈ ℝ^{d×d}

(Mallinar Def. 2.1 writes the Jacobian transposed, d × c; it is the same matrix).

**What it measures.** For any unit direction u in input space,

  uᵀ G u = (1/n) Σ_i ‖J(x_i) u‖²,

the mean squared output change when every training input is nudged along u. So:
- G is symmetric and positive semi-definite (PSD), as an average of JᵀJ's.
- **Eigenvectors with large eigenvalues are the input directions the function is most
  sensitive to**: its "features". Directions in the null space are ignored.
- tr G = (1/n) Σ_i ‖J(x_i)‖²_F is the total sensitivity.

**Three small examples.**
- Linear model f(x) = Ax: J = A everywhere, so G = AᵀA. Its eigenvectors are A's
  right singular vectors.
- Single-index model f(x) = g(βᵀx): J = g′(βᵀx) βᵀ, so G = mean(g′²) · ββᵀ. That is
  rank 1, pointing at β, the only direction that matters.
- k-index model f(x) = g(Bx), with B of size k × d: G has rank ≤ k and lives in
  B's row space. This is why the AGOP finds low-dimensional structure in multi-index
  problems.

### 1.2 Why the AGOP is "the features" of an MLP

Write the first layer explicitly: f(x) = h(W₁x + b₁), where h is the rest of the
network. By the chain rule, J(x) = ∂h(z)/∂z · W₁ with z = W₁x + b₁, so

  G = W₁ᵀ Ā W₁,  Ā = (1/n) Σ_i ∂h(z_i)ᵀ ∂h(z_i)  (width × width).

So the AGOP is the first-layer weights' Gram matrix W₁ᵀW₁ (the **Neural Feature Matrix**,
NFM), seen through the metric Ā of the downstream layers. If Ā were proportional to the
identity, G would be proportional to the NFM. The **Neural Feature Ansatz** (NFA,
Radhakrishnan et al. †) says that empirically the NFM tracks a power of the AGOP,
W₁ᵀW₁ ∝ G^s with s ≈ ½. Mallinar et al. (§4, Fig. 6) find Pearson correlations of
0.92–0.96 between the NFM and G^{1/2} for modular arithmetic. That is why they call
**M = G^{1/2} the feature matrix**.

### 1.3 What it looks like in our three tasks

| Task | d (G is d × d) | What G's structure means |
|---|---|---|
| Modular (p = 97) | 194 | Inputs are one-hot e_a ⊕ e_b, and column j of W₁ is the embedding of token j. So G_{jk} = embed_jᵀ Ā embed_k is the Gram matrix of token embeddings. **Circulant blocks** (Mallinar Obs. 1: entries depend only on j − k mod p) mean that embedding similarity depends only on the difference of two tokens, i.e. a Fourier embedding, the known generalizing solution. |
| MNIST | 784 | A pixel-space feature map: which pixel patterns move the output. |
| Teacher–student | 5 | Only 5 × 5. The teacher is a generic tanh MLP of all 5 inputs, so the target AGOP is anisotropic but very likely **full rank**. |

**An MNIST trap worth knowing.** A pixel that is 0 on every training image gets no data
gradient in its W₁ column (∂L/∂W₁[:, j] ∝ x_j = 0). With weight decay that column
decays exponentially, so G loses rank **at exactly the decay rate, for reasons unrelated
to learning**. Any AGOP-rank or AGOP-motion signal on MNIST must mask input coordinates
with zero training variance. B2 already masks such coordinates for the same reason.

### 1.4 Scale: why the raw AGOP is not comparable across runs

For a bias-free ReLU network of depth D, scaling every weight by c scales f, and so J,
by c^D, and **G by c^{2D}**. Two consequences:
- Init scale α changes G's magnitude by α^{2D}. On the Omnigrok plateau the norm shrinks
  by large factors (×0.1 in our MNIST pilot), so ‖G‖ changes by orders of magnitude
  without any change in the features. Only the **direction** of G (cosine,
  trace-normalized G) is meaningful.
- **Pure weight-decay motion does not move the normalized AGOP.** Decay shrinks every
  layer by the same factor (1 − lr·γ), which is a common rescaling, so G/tr G is unchanged
  (exactly without biases, approximately with them). Only the tangential, data-driven
  part of the update rotates the features. This is the AGOP version of S1's τ (METRICS
  §3): features move only when weights move *across* w, not along it.

### 1.5 Mallinar's AGOP alignment, and why it is not online

Let M_t be the feature matrix at step t and M* the feature matrix of the fully trained
model. Then

> **AGOP alignment ρ(t) = cos(M_t, M*) = ⟨M_t, M*⟩_F / (‖M_t‖_F ‖M*‖_F)** (Mallinar eq. 8)

It is scale-invariant, so the c^{2D} problem disappears. Mallinar show it rises
**gradually, nearly linearly**, while test accuracy is still flat, then reaches 1. It is a
"hidden progress measure". But it needs M*, the end of training. In their words it is
"an a posteriori measurement of progress". A stopping rule cannot use it.

Note: Mallinar use M = G^{1/2}. The proposal uses G itself (its Remark 1.1 notes the
difference). G weights large eigenvalues quadratically more than G^{1/2}, so the two
alignments are different numbers.

---

## 2. The proposal, step by step

**Goal:** estimate s(t) = cos(G_t, G_T) at time t, where G_T is the end-of-training AGOP,
without seeing the future.

**Step 1: put G on the spectraplex.** G̃ = G / tr G has trace 1, so its eigenvalues
λ_j ≥ 0 sum to 1, like a probability distribution (eq. 2). Two quantities follow:
- **Purity** P = ‖G̃‖²_F = Σ λ_j² ∈ [1/d, 1].
- The **participation ratio** r_eff = 1/P (eq. 6), an effective rank: d for a flat
  spectrum, 1 for rank 1. It is a cousin of the entropy-based erank in our R1.

**Step 2: anchor at memorization.** t_mem is the first step with train loss ≤ ε_train.
G₀ is the average of G over a short window [t_mem, t_mem + w], and t₀ = t_mem + w
(eqs. 3–4). All of this is causal.

**Step 3: the chord model** (Assumption 2.1). After t₀,
G̃_t = (1 − α_t) G̃₀ + α_t G̃_T, with α_t rising from ≈0 to 1. So **the normalized AGOP
moves on the straight segment from G̃₀ to G̃_T**, which is equivalent to G_t lying in
the 2-D cone spanned by G₀ and G_T. Eigenvectors may not rotate, except by reweighting
those two fixed matrices.

**Step 4: the defect δ** (Assumption 2.2). δ = ⟨G̃₀, G̃_T − G̃₀⟩ says whether the
direction of travel is orthogonal to the start (δ = 0) or not.

**Step 5: the purity law** (Lemma 3.1). Expanding ‖G̃₀ + α(G̃_T − G̃₀)‖² gives
P(t) = P₀ + 2α_t δ + α_t² ΔP_δ, where ΔP_δ = ‖G̃_T − G̃₀‖². If δ < 0, purity first
*dips* below P₀ (the "dead zone") before rising.

**Step 6: two online estimators** (Def. 3.1).
- ŝ_proj(t) = sin ∠(G_t, G₀): how far the AGOP has turned away from the anchor.
- ŝ_spiked(t) = √(1 − P₀/P(t)) = √(1 − r_eff(t)/r_eff,0): the fractional drop in
  effective rank. It equals ŝ_proj when δ = 0.

**Step 7: the angle-addition law** (Thm. 3.4). In the plane of G₀ and G_T, let
Φ = ∠(G₀, G_T) be the full angle and ψ = ∠(G₀, G_t) the angle travelled so far. Then
∠(G_t, G_T) = Φ − ψ, so s(t) = cos(Φ − ψ) = sin(ψ + θ_init) with θ_init = π/2 − Φ.
ψ is observable. **θ_init, i.e. how far away the target is, is not.** Corollary 3.6
says the same thing: ŝ_spiked alone tends to √(1 − r/d_eff) < 1, so a fixed threshold
near 1 may never be reached.

**Step 8: online checks of the model** (Thm. 4.1).
- D_chord(t) = 1 − cos(G̃_t − G̃₀, G̃_{t−Δ} − G̃₀) must be 0 on a straight line.
- v(t) = ⟨G̃_t, G̃₀⟩ − P₀ = α_t δ tracks the defect.

**Step 9: extrapolate the asymptote** (Assumption 5.1, Lemma 5.1, Thm. 5.2). Assume
α_t = 1 − c·e^{−κ(t−t₀)}. Then u(t) = √(P − P₀) (when δ = 0) and v(t) (any δ) are
single exponentials approaching their limits. Three window means at equal lags give the
limit by Aitken's Δ² formula, z_∞ = (z̄₁z̄₃ − z̄₂²)/(z̄₁ + z̄₃ − 2z̄₂), and
q = e^{−κΔ}. From those come α̂_t, P̂_T, δ̂, and finally ŝ_cal(t) (eqs. 37, 39), a
predicted crossing time t* (eq. 41), and three checks that the exponential model holds
(eq. 42).

### 2.1 The same thing, said more simply

On a straight line, three things are **observable at time t**:
- the start G̃₀;
- the direction of travel, d = (G̃_t − G̃₀)/‖G̃_t − G̃₀‖;
- the distance covered so far, a(t) = ‖G̃_t − G̃₀‖.

**The only unknown is the total length L = ‖G̃_T − G̃₀‖**, because G̃_T = G̃₀ + L·d.
Then δ = L⟨G̃₀, d⟩, P_T = ‖G̃₀ + L·d‖², α_t = a(t)/L, and
s(t) = cos(G̃_t, G̃₀ + L·d) are all functions of L. Steps 5–9 are a long route to
**estimating one number, L**, which the proposal does by assuming a(t) relaxes
exponentially from t₀ and extrapolating. This view makes the two weak points obvious:
the direction must not change (chord), and the history from t₀ must follow one
exponential (Aitken).

---

## 3. Review of the proposal

### 3.1 What I checked and found correct

- Lemma 3.1: the purity law, the roots, and the dead-zone threshold −2δ/ΔP_δ.
- Prop. 2.1: cases 1–2, and case 3's Cauchy–Schwarz bound and Haar expectation.
- Lemma 3.2: eqs. 17–19.
- Thm. 3.4 parts 1–2: the angle-addition law.
- Lemma 5.1: Aitken on window means. The offset c cancels, as claimed.
- Thm. 5.2: eqs. 36, 38–40.
- I did not re-derive every step of Thm. 3.4(3)'s perturbation bound.

### 3.2 Where it breaks as an online estimator

**(a) The exponential-from-t₀ assumption does not match grokking.** Under grokking the
AGOP does not start relaxing at memorization. There is a long plateau, during which
Mallinar see gradual, roughly linear alignment progress, and then a sharp transition:
sigmoid-like, not 1 − e^{−κt}. On the plateau the three window means are nearly
collinear or noise-dominated, so the Aitken denominator z̄₁ + z̄₃ − 2z̄₂ is noise. Then
δ̂, α̂ and ŝ_cal are arbitrary. The paper's Remark 5.1 flags the conditioning issue for
slow plateaus; the toy (§5) shows it is fatal without a gate. **With 5% noise on a
grokking-shaped path, the bare estimator stopped on the plateau in 10/10 seeds, at a true
alignment of 0.15–0.21.**

**(b) The chord model excludes feature rotation.** Grokking is described as features
*forming*: new eigen-directions appear and rotate into place. The paper's Remark 2.1
says so itself. A path that visits an intermediate structure stops the bare estimator
early (toy: 9/10 premature, true s down to 0.51).

**(c) The D_chord gate is what rescues it.** With the paper's own collinearity test
(window means, accept only D_chord < 0.05), every premature stop disappeared in the toy.
So the gate is not an optional diagnostic: **the method is only safe with it.** But that
makes the method's reach as narrow as its model. If real AGOP trajectories keep
rotating, the gate never opens and the rule never stops. Only real traces can answer
that.

**(d) The rank-compression premise is not universal** (Assumption 2.3).
- Teacher–student: G is 5 × 5 and the target is likely full rank, so purity has little
  room to move.
- Mallinar (§2) stress that the modular features are "not low-rank" for RFM. With weight
  decay, neural networks do concentrate on a few Fourier frequencies, so compression is
  plausible for our modular runs, but it is task-dependent.
- MNIST shows spurious compression from dead pixels (§1.3).

**(e) The sign of δ is not generically negative.** Prop. 2.1(3) derives E[δ] < 0 under a
Haar-random eigenbasis. But Σλ_j² ≤ λ₁Σλ_j = λ₁, so if the AGOP compresses onto G₀'s
*own* dominant directions (G̃_T = top eigenvector), δ = λ₁ − P₀ ≥ 0 (toy: +0.068, versus
−0.038 for a random rank-3 target). Learned features grow out of the memorizing state, so
the Haar model is the wrong prior. This is not a mathematical error, since the general-δ
route covers both signs; it just means the dead zone is not the typical case.

**(f) Shared with every stationarity signal:**
- G_T is whatever training ends at. On a late-collapse run that is the collapsed model.
- "s ≥ 1 − ε" means "the features have almost finished changing", which is late by
  construction on late-collapse runs (METRICS §5).
- A second transition after an apparent settling cannot be predicted causally by any
  method.

### 3.3 What to keep

- Cosine / trace normalization, which removes the c^{2D} scale problem.
- Window-averaged AGOPs against noise.
- **Measuring progress as a fraction of the total change**, which is the key idea for
  time normalization.
- D_chord as a diagnostic: it is cheap, and worth recording.

---

## 4. Proposed online version: the remaining-angle bound

This section stands on its own. It needs only §1 (what an AGOP is), not the proposal.

### 4.1 What we want

We want a signal that says **"the features have essentially stopped changing, and will
not change much more"**. It must be computed at step t from the training data and the
current and past weights only. On a grokking plateau the training loss is flat but the
features are still forming, so the signal must look at the features themselves.

### 4.2 The derivation, step by step

**Step 1. The object.** At an eval step t, compute the AGOP G_t on the training set
from the current weights, and take **M_t = G_t^{1/2}**. G^{1/2}, not G, is the matrix
that matches the first-layer Gram matrix W₁ᵀW₁ (§1.2, the Neural Feature Ansatz). M_t is
"the features at step t".

**Step 2. Keep only the shape.** M's size carries no feature information: it scales as
c^D with the weights (§1.4), so the init scale and the norm shrinking under decay would
dominate. Keep **M̂_t = M_t/‖M_t‖_F**, a point on the unit sphere of d × d matrices. The
distance between two feature states is the **angle**
θ(A, B) = arccos⟨Â, B̂⟩_F ∈ [0, π/2]. The angle is at most π/2 because PSD matrices have
⟨A, B⟩ ≥ 0. Mallinar's alignment is cos θ(M_t, M_final). During training, M̂_t traces
a path on this sphere, from memorizing features towards generalizing ones.

**Step 3. What "done" means.** Being done means the current point is already close to
the end of the path: θ(M̂_t, M̂_T) ≤ ε, with T the end of training. M̂_T is in the
future. Knowing where you are and where you have been does not tell you where you will
end, so every online estimate of this angle is a prediction. The question is which
prediction needs the least.

**Step 4. Trade "where" for "how much more".** Angles between unit vectors obey the
triangle inequality. Hence the direct distance from now to the end is at most the length
of the path still to travel. With ω(τ) the angular speed in radians per step,

> **θ(M̂_t, M̂_T) ≤ R(t) := ∫_t^T ω(τ) dτ**  (the remaining path length).

This holds **for any path shape**: turns, wiggles and detours included. We no longer
need to know where the features are going, only how much more they will move.
(Analogy: you don't know where a car is heading, but its speed and how hard it brakes
bound how far it can still get.)

**Step 5. Measure the speed causally.** ω̂(t) = θ(M̂_t, M̂_{t−Δ})/Δ, using window
means of M̂ over a few nearby eval steps against minibatch jitter. Full-batch tasks
(modular, teacher–student) are already smooth.

**Step 6. Predict the future speed: the one modelling assumption.** Assume that **from
now on** the speed decays exponentially, ω(τ) = ω(t)·e^{−κ(τ−t)}.
- Why this is reasonable near the end of training: near a stable minimum of the
  regularized loss, gradient dynamics are approximately linear, w − w* ∝ e^{−λt}.
  Anything that depends smoothly on w, including M̂, then approaches its limit
  exponentially, and so does its speed.
- The slowest mode dominates the tail, so one rate κ describes it.

Integrating gives R(t) = ω(t)/κ. The rate is read off how fast the speed itself has
been falling: κ̂ = −d log ω̂/dt over a trailing window.

**Step 7. The signal and the rule.**

> **R̂(t) = ω̂(t)/κ̂(t)** radians; R̂ = ∞ when κ̂ ≤ 0 (the speed is not falling).
> Stop when R̂ < ε_R for a few consecutive checks.

If the tail model holds, alignment with the final features is then at least cos ε_R
(ε_R = 0.1 rad gives ≥ 0.995). This fits the existing `RuleConfig` threshold template
(DESIGN §13.4) on the column R.

**Correction (2026-11-14): with lags proportional to t, use the discrete form.** ω̂ is
measured over a lag Δ, so it is the *average* speed over that lag, which exceeds the
current speed by (e^{κΔ} − 1)/(κΔ). That factor is ≈ 1 only when κΔ is small, but with
Δ = t/k it grows with t, so ω̂/κ̂ overstates the remaining angle more and more late in a
run (×1.9 at κΔ = 1.2 on an exponential test path). The exact remaining angle on an
exponential tail is the Banach form R̂ = θ_Δ·q/(1 − q), with θ_Δ the angle over the last
lag and q = e^{−κΔ}; ω/κ is its small-κΔ limit. Estimate q from the last two equal lags,
q̂ = (θ_Δ/Δ)/(θ_{previous Δ}/Δ′) (the textbook contraction-factor estimate), or from a
trailing regression of log speed. The F2 experiment (`stopping/agop_study.py`) uses
this form and tests both estimators (unit test: exact to ~5% on an exponential path).

**Worked example.**
- ω̂ = 1e-4 rad/step, and the speed halved over the last 2,000 steps: κ̂ = ln 2/2000 =
  3.5e-4 per step, so R̂ = 0.29 rad (≈ 17°). Not done.
- Later, ω̂ = 1e-5 with the same κ̂: R̂ = 0.029 rad, so alignment ≥ 0.9996. Stop.

### 4.3 Why this is the right normalization

A raw change measure such as ‖ΔM‖/‖M‖ per step depends on how big the steps are, and
our learning rates span 3e-4 to 1.0. In R̂, ω has units of rad/step and κ has units of
1/step, so **the time unit cancels**:

| Change | ω̂ | κ̂ | R̂ |
|---|---|---|---|
| Double the learning rate (same path, run twice as fast) | ×2 | ×2 | unchanged |
| Larger init α, or the norm shrinking | unchanged (angles ignore scale) | unchanged | unchanged |
| γ = 0; AdamW vs SGD | no γ or lr appears in the formula | | still defined |
| d = 5 / 194 / 784 | | | an angle means the same in any dimension |

**R̂ is normalized by its own timescale.** Compare the two options of METRICS §6.3:
- dividing by lr_eff·γ·Δ breaks at γ = 0, and Adam has no single effective lr;
- dividing by ‖Δw‖/‖w‖ becomes 0/0 at convergence, and measures feature change *per
  unit of weight change*, a different quantity from progress.

One threshold ε_R can therefore be calibrated globally (PLAN §6 Phase 5's
"one global setting" test).

### 4.4 Behaviour on the cases that matter

| Phase | Speed ω̂ | κ̂ | R̂ | Decision |
|---|---|---|---|---|
| Grokking plateau: loss flat, features drifting steadily | roughly constant | ≈ 0 | ∞ | **no stop** (B1 and B4 stop here) |
| Transition under way | rising | < 0 | ∞ | no stop |
| Settling after generalization | falling exponentially | > 0 | shrinks to 0 | **stop** |
| Fast run | falls early | > 0 | small early | early stop (correct) |
| Late collapse | small, then rises again | | | stops before the collapse; correct only if that is before the test peak fades (shared weakness, METRICS §5) |

The proposal (§2) fails exactly where the bound is safe: on the plateau its estimate is
arbitrary, while the bound's estimate is ∞.

### 4.5 Practical details

- **Windows scale-free in time too.** A lag or fit window fixed in steps reintroduces
  the lr timescale. Use a lag proportional to the step, Δ = t/k (the log-spaced eval
  grid gives this naturally), and fit κ over a window in log-time.
- **Noise floor (MNIST only).** Under minibatch SGD the weights jitter, so the angle
  between window means has a floor of about √2·σ/√w, where σ is the within-window
  angular scatter (measurable). Subtract it in quadrature: ω² ← max(ω² − floor², 0)/Δ².
  - **Risk:** a real plateau drift *below* the floor would read as "stopped". The lag
    growing with t is the defence, since drift accumulates over Δ while the floor does
    not. Check on MNIST.
- **Which matrix.** Recommended: M = G^{1/2} (Mallinar's feature matrix, matches the
  NFM). Keep G as a config option. On MNIST, mask zero-variance pixels (§1.3).
- **Cost and storage.**
  - Per eval step: d_out backward passes over the training set (97 modular, 10 MNIST,
    5 teacher–student; METRICS §3 F2) and one d × d eigendecomposition.
  - Store **scalars only**: the full G at 600 eval steps would be ~1.5 GB per MNIST run.
    Store the angle to the previous window mean at a few lags, ω, κ̂, R̂, purity and
    erank, plus the proposal's v(t), D_chord and ŝ_cal, so both methods can be scored
    from one trace.
- **Fits the code as it is.** One new `eval`-cadence signal under `stopping/signals/`
  (requires `models` and `train_data`) that keeps the last few window means in memory;
  the rule layer is unchanged (DESIGN §13.8).

### 4.6 What it assumes, and its weaknesses

**It does not assume** the direction the features move in, a straight path, low rank,
or a reference point at memorization. **It assumes** only that the *tail* of the speed
decays exponentially, and that assumption fails safe: a speed that is not clearly
falling gives R̂ = ∞.

**Weaknesses:**
- **It stops late.** It is a lower bound on the alignment, so it is conservative. In the
  toy it stopped 1.4k steps (exponential path) to 5k steps (sigmoid path) after s
  reached 0.95, at s ≈ 0.98–1.0. A late stop costs regret only on late-collapse runs.
- **It is still a stationarity signal** (§3.2 f). The late-collapse guards of METRICS
  §5 (ρ < 0, the train-accuracy guard) remain needed.
- **Cross-entropy at γ = 0:** the weights grow forever and the direction converges
  slowly (like 1/log t, not exponentially), so κ̂ stays small and the rule stops late or
  never.
- **The tail model can only be checked after the fact:** if κ̂ drifts, the estimate is
  not yet trustworthy. The proposal's D_exp check (eq. 42) applies unchanged.
- **Untested on real AGOP traces.** The toy only shows that the bound behaves as
  designed.

---

## 5. Toy check

[`results/phase4/agop_toy_sim.py`](results/phase4/agop_toy_sim.py) (numpy, about 1 min).

**Setup.**
- Synthetic trace-normalized AGOPs in d = 30, moving from a full-rank G₀ to a rank-3
  G_T. Log every 50 steps, 2e4 steps, 10 seeds.
- α_t is either the paper's exponential (κ = 3e-4) or a grokking-shaped sigmoid (flat
  until ~7k, crossing at 9k).
- Optional off-chord detour; optional symmetric noise with ‖noise‖ = 5% or 15% of ‖G̃‖.
- Every method stops when its estimate of s reaches 0.95. A stop is **premature** if the
  true s is below 0.95 there.

| Path | Paper (bare) | Paper + D_chord gate | Remaining-angle bound |
|---|---|---|---|
| exponential, 5% noise | 2/10 (min s 0.947) | 2/10 (min s 0.947) | 0/10 (min s 0.983) |
| sigmoid, clean | 0/10 | 0/10 | 0/10 |
| sigmoid, off-chord detour | **9/10 (min s 0.51)** | 0/10 (min s 0.993) | 0/10 (min s 1.000) |
| sigmoid, 5% noise | **10/10 (min s 0.15)** | 0/10 (min s 0.996) | 0/10 (min s 0.999) |
| sigmoid, 15% noise | **10/10 (min s 0.15)** | 0/10 (min s 0.988) | 0/10 (min s 0.993) |

Entries are premature stops out of 10. "Min s" is the lowest true alignment at the
stop. No method failed to stop on any run. The exponential "premature" cases stop at
s = 0.947, which is a rounding-level miss on the paper's own model.

**Reading.**
- On its own model the proposal is exact and stops on time.
- Off its model, the bare estimator is dangerous exactly on the plateau. The gate makes
  it safe *in this toy*.
- The bound is never premature, and pays with a later stop.
- The toy does not show whether the gate stays open on real trajectories; §6 item 4 is
  how to find out.

---

## 5b. After the corpus experiment: settling tests instead of extrapolation (2026-11-15)

**What failed.** On 61 corpus runs (results/phase4/RESULTS.md) the bound stopped too early
in about half the runs. The features *do* keep moving on the plateau, but right after
memorization their speed falls fast, so any extrapolation of the speed's trend says
"almost done" while most of the feature motion is still ahead. The flaw is the
extrapolation, not the feature matrix.

**Idea.** Don't predict the future speed. Ask a question about the recent *shape* of the
path, over a trailing window [t/f, t] (log-time, so it has no learning-rate unit): is
the feature matrix still **going somewhere** (directed motion), or is it **staying in
place** apart from noise (confined motion)? Two standard tools answer exactly this:

1. **Mean-squared displacement (MSD) scaling**, from trajectory analysis in physics and
   biophysics (single-particle tracking; Michalet 2010 †, Metzler et al. 2014 †). For two
   points of the path a lag ℓ apart, MSD(ℓ) = mean squared distance. Then
   MSD(ℓ) ∝ ℓ^α with α ≈ 2 for directed motion (drift), α ≈ 1 for a random walk, α ≈ 0
   for motion confined around a point. **Settled ⇔ α small.** On the unit sphere the
   squared chord distance is exactly the squared displacement, so every MSD is already in
   the trace. The fit uses the longest lags of the window (ℓ ∈ [L/8, L]), where a drift
   shows and jitter does not.
2. **Geweke's convergence diagnostic** (Geweke 1992 †; Gelman & Rubin 1992 † is the
   multi-chain version), the standard test for whether an MCMC chain has reached
   stationarity from its own history. Split the window into an early and a late segment
   and compare their means, in units of their standard errors:
   Z = ‖mean_late − mean_early‖² / (s²_early/n_early + s²_late/n_late), with s² the
   segment's total variance. Z ≈ 1 for jitter around a fixed point; Z ≫ 1 for a drift.
   **Settled ⇔ Z small.** Also computed from the stored distances alone.

**Why these avoid the failure.** At the end of memorization the path is still drifting
slowly. Over a window [t/2, t], that drift makes the longest-lag displacements grow like
ℓ² (α ≈ 2) and separates the two segment means (Z large), however fast the speed fell
just before. Neither test assumes anything about the future. Adam's jitter on full-batch
tasks reads as confined (α ≈ 0, Z ≈ 1), which is the right answer there: the features have
stopped going anywhere. Unit test: on a synthetic fast-transient-then-drift path the
bound says "done" (R̂ < 0.1) while the MSD exponent stays above 1
(`tests/test_agop.py`).

**What they cannot do.** They detect "has settled", not "will not move again": a second
transition after a quiet stretch is invisible to any causal test (§3.2 f). And late
collapse still needs the separate guards of METRICS §5.

Both are evaluated offline on the stored traces (`settle_rows` in
`stopping/agop_study.py`): object (M, G, W) × window factor (2, 4) × threshold × norm
gate (same test on log‖w‖).

---

## 6. Decisions for the user

1. **Signal definition for F2.** Replace METRICS F2's "consecutive change" with the
   remaining-angle bound R̂ (recommended). Alternatives: keep the proposal's ŝ_cal as the
   signal, or record both. Recording both costs only a few extra columns in one trace.
2. **Matrix:** G^{1/2} (recommended, Mallinar's feature matrix) or G (the proposal's).
3. **Lags and windows proportional to t** (recommended), not fixed in steps.
4. **Validate before committing to a rule.** Build the signal and run it on the Phase 4
   pilot runs (MNIST), plus a few teacher–student and, once trained, modular runs. Look
   at three things:
   - whether real AGOP paths are straight (D_chord);
   - whether the speed tail is exponential (stability of κ̂);
   - where R̂ and ŝ_cal stop relative to t_ε.

# Online Progress Measures from AGOP Feature Learning

**Project plan — reproduce, then extend, Mallinar et al. (2024)**

---

## 1. Goal

The paper shows that grokking is driven entirely by *feature learning*, and that two
"hidden progress measures" improve gradually while test accuracy and test loss sit flat.
Both of their measures, however, are **a posteriori** — they cannot be computed at time `t`
using only information available at time `t`.

- **Circulant deviation** requires knowing in advance that the generalising features are
  block-circulant (and, for mul/div, requires the discrete-logarithm reordering, which in
  turn requires knowing `Z*_p` is cyclic of order `p-1`).
- **AGOP alignment** `rho(M_t, M*)` requires `M*`, the feature matrix of the *fully trained*
  model.

The paper concedes this directly (Section 6): all such measures "require either non-trivial
understanding of the algorithm implemented by a fully generalizable trained model ... or
access to such a model."

**Our objective:** construct a *causal / online* progress measure derived from the AGOP
trajectory alone — using only `M_0 ... M_t` — that (a) rises during the accuracy plateau,
(b) stays flat on runs that never grok, and ideally (c) forecasts *when* the transition
will occur.

Secondary framing: this is exactly the reward signal a bandit-style model-selection scheme
needs — available at decision time, rather than after training completes.

---

## 2. Source material

### Paper
Mallinar, Beaglehole, Zhu, Radhakrishnan, Pandit, Belkin.
*Emergence in non-neural models: grokking modular arithmetic via average gradient outer product.*
arXiv:2407.20199v3 (stat.ML, 9 Jul 2025).

### Code
**https://github.com/nmallinar/rfm-grokking** — GPL-3.0, 4 commits, built on
`danielmamay/grokking`.

| File | Contents |
|---|---|
| `train_kernel.py` | RFM loop; kernels: gaussian, laplace, quadratic, general_quadratic, jax_ntk_nngp |
| `train_net.py` | One-hidden-layer quadratic-activation networks |
| `train_random_circulant_{kernel,net}.py` | Figures 4 and 7 |
| `utils.py` | Discrete-log reordering (`get_a_from_p`, `get_lg_idx_from_p`, `reorder`, `unorder`), random circulant/Hankel generators, AGOP visualisation |
| `models/` | Kernel implementations and AGOP update rules |
| `requirements.txt` | Includes jax / neural-tangents (only needed for the NTK kernel — skippable) |

### Two gotchas in that repo

1. **Neither progress measure is implemented.** `train_kernel.py` logs only accuracy, MSE,
   correct-logit loss, and `tr(M)`. Circulant deviation and AGOP alignment were computed
   offline from saved `M.npy` files; no plotting script is included.
2. **The checkpoint cadence is wrong for short runs.** Feature matrices are saved only when
   `--save_agops` is passed *and* `rfm_iter % 25 == 0`, so a 30-iteration run stores
   iterations 0 and 25 only. The line the authors evidently used for the figures sits
   directly above, commented out:
   ```python
   # if (rfm_iter < 31) or \
   if (rfm_iter < 100 and rfm_iter % 25 == 0) or ...
   ```

**Recommendation: reimplement rather than adapt.** The kernel-RFM core is roughly 100 lines.
Lift only `utils.py`'s discrete-log reordering and circulant generators (respect GPL-3.0 —
either keep the derived files under GPL or rewrite them from the paper's Appendix C
description). Adapting the original means inheriting a wandb dependency, a known-broken jax
DLPack path, and hardcoded `ntk_depth`, while still having to write the measurement code.

---

## 3. Paper facts needed to reproduce

### Task encoding
`f*(a,b) = g(a,b) mod p`. Input is `e_a (+) e_b` in `R^{2p}` (concatenated one-hots),
label is `e_{f*(a,b)}` in `R^p`. Total inputs `N = p^2`, except division where `N = p(p-1)`.
Training set is a random subset of size `n = r * N`, `r` = training fraction.

### RFM algorithm
```
M_0 = I_d
for t in 0..T-1:
    alpha  = k(X, X; M_t)^{-1} y          # ridgeless kernel regression
    f_t(x) = k(x, X; M_t) alpha
    M_{t+1} = [G(f_t)]^s                  # AGOP, matrix power s = 1/2
```
AGOP: `G(f) = (1/n) sum_j J_f(x_j) J_f(x_j)^T`, in `R^{d x d}` with `d = 2p`.

Kernels:
- Quadratic: `k(x, x'; M) = (x^T M x')^2`
- Gaussian: `k(x, x'; M) = exp(-||x - x'||^2_M / L)`, bandwidth `L = 2.5`

All experiments use `s = 1/2`. Ridgeless regression means train loss is numerically zero at
every iteration — which is the whole point: feature learning proceeds with a *constant,
identically zero* training signal.

### Neural network config (Appendix B)
One hidden layer, no biases, quadratic activation `sigma(z) = z^2`, width 1024,
AdamW, batch size 32, lr `1e-3`, weight decay 1.0, MSE loss, standard PyTorch init,
`p = 61`, training fraction 50%, 50 epochs.
NFM = `W_1^T W_1`; compare against `sqrt(AGOP)` (paper reports Pearson correlation > 0.92).

### The two baseline measures

**Circulant deviation.** With `A` the bottom-left `p x p` sub-block of `M`, and `S` the shift
operator that shifts row `l` right by `l` positions:
```
D(A) = (1 / ||A||_F^2) * sum_j Var( S(A)[:, j] )
```
Zero iff `A` is exactly circulant. For mul/div, apply discrete-log reordering first.

**AGOP alignment.** Cosine similarity of vectorised matrices, `rho(M_t, M*)`, where `M*` is
the final feature matrix. Non-causal by construction.

### Reordering (Appendix C)
`Z*_p` is cyclic of order `p-1`. For a generator `g`, map entry `(r, c)` to `(phi_g(r), phi_g(c))`
where `phi_g(g^i) = i` (discrete log base `g`). Only the bottom-right `(p-1) x (p-1)`
sub-block is reordered; row/column 0 are identically zero. Lemma G.1 guarantees the choice of
generator is irrelevant — convention is the smallest generator.

---

## 4. Compute budget

`p = 61`, 50% training fraction gives `n ~ 1860`, `d = 122`. The kernel solve is a
1860 x 1860 dense solve; the AGOP is 122 x 122. **This is a CPU workload.** Only the
long-horizon NN runs want a GPU.

| Item | Estimate |
|---|---|
| Env setup (skip jax / neural-tangents) | 1–2 h |
| Progress-measure post-processing | 1–2 h |
| RFM run, 30 iters | ~1–2 min |
| Figs 1–3 + alignment: 4 ops x 2 kernels x 3 seeds | < 1 h |
| Fig 5 (NN, 50 epochs) + Fig 6 NFM/AGOP correlation | ~1 h |
| Appendix Fig 6 sweep (training fraction, p = 97) | 1–2 h |
| Figs 4, 7 (random circulant, 3000-epoch NNs) | 2–4 h GPU — **optional, skip** |
| Appendix Fig 5 (200k epochs, 3 reg settings) | hours to ~a day each — **skip** |

Figures 4, 7 and Appendix Fig 5 are not needed for this project's argument. Cutting them
takes the reproduction from ~3 days to ~1 day.

---

## 5. Phase plan

### Phase 0 — Scaffold (half day)
Write the package skeleton (Section 6), no science. Deliverable: one RFM run that emits
feature-matrix checkpoints and a `metrics.jsonl`.

### Phase 1 — Reproduce (1 day)
Quadratic-kernel RFM, `p = 61`, `r = 0.5`, 30 iterations, all four operations; then the NN
for 50 epochs.

**Gate:** circulant deviation and AGOP alignment curves visually match Figures 2B and 5B —
alignment rising near-linearly during the plateau, deviation falling to ~0 by the end,
both moving while accuracy and loss are flat for the first ~8–10 iterations. Do not proceed
until this passes; everything downstream depends on the trajectory being right.

Sanity checks: train loss numerically zero at every RFM iteration; final feature matrix
matches Observation 1 (`M* = [[A, C^T], [C, A]]`, `C` a non-degenerate circulant,
`A = c_1 I + c_2 11^T`); NFM vs `sqrt(AGOP)` Pearson > 0.92 for the NN.

### Phase 2 — Freeze the evaluation corpus (1 day)
**Build this before designing any new measure.** Runs with known outcomes:

- **Grokkers** — 4 ops x {quadratic, Gaussian} x 3 seeds; NN equivalents.
- **Non-grokkers** — training fraction 5–20% (below the ~25% threshold in Appendix Fig 6);
  NN with no regularisation (Appendix Fig 5, left panel); random-label control.
- **Fast learners** — random-circulant-transformed inputs (Eq. 9), which generalise almost
  immediately. These separate "measures a real signal" from "counts iterations".
- **Multi-task** — the `x+y` / `x^2+y^2` pair from Appendix E, which groks at two distinct
  points. A good online measure should fire twice.

Label each run with its grok step (first `t` where test accuracy crosses 0.9) or `None`.
Freeze the corpus. Any measure is scored against exactly this set.

### Phase 3 — Online probes (2–3 days)

Preprocessing that applies to all of them: extract the **off-diagonal block only** (the
diagonal dominates `M` and swamps the structural signal), and **Frobenius-normalise before
differencing** — otherwise you measure trace growth from the `s = 1/2` power rather than
direction. For NN runs, smooth over epochs; single-epoch AdamW increments are minibatch noise.

Candidates:

1. **Increment coherence.** `Delta_t = M_hat_{t+1} - M_hat_t`; track `cos(Delta_t, Delta_{t-1})`.
   Coherent drift toward a fixed target gives ~1; aimless wandering ~0. Since the paper's
   alignment curve is near-linear during the plateau, increments should be strongly
   collinear — and that collinearity is visible at time `t`.
2. **Path-direction persistence.** `cos(Delta_t, M_hat_t - M_hat_0)`. Cheaper, similar content.
3. **Sliding-window extrapolated saturation.** Fit `rho(M_t, M_{t+k})` over a trailing window
   and extrapolate to where it would reach 1. If this forecasts the grokking iteration several
   steps early, it is genuinely predictive rather than merely hidden.
4. **Spectral variants (stretch).** Track the entropy or participation ratio of the AGOP
   spectrum, or the rotation angle of its top eigenspace between steps.

**Explicitly rejected:** raw `rho(M_t, M_{t+1})`. It is confounded — consecutive similarity is
high both when nothing is happening and when training has converged, so it cannot separate a
dead run from a finished one.

Score every probe on three numbers:
- **Lead time** — steps between the probe firing and the accuracy transition.
- **False-positive rate** — must stay flat on the non-grokker arm.
- **Seed variance** — a measure that needs per-run tuning is not a measure.

Sweep `p` and the operation. A probe tuned on `x+y` at `p=61` will overfit.

### Phase 4 — Forecasting (open-ended)
Turn the best probe into a predictor of *when*, not just *whether*. Success looks like:
at iteration 10 of a 30-iteration run, output a calibrated estimate of the transition step,
with error bars, validated on held-out `(p, operation)` pairs.

---

## 6. System design

The abstraction that matters: **decouple the learner from the measurement.** RFM iterations
and NN epochs are both just a stream of feature matrices. Everything downstream is blind to
which produced it.

```
agopx/
  learners/     base.py, rfm.py, neural.py
  probes/       base.py, registry.py, offline.py, online.py
  features.py   block extraction, normalisation, reordering, smoothing
  runner.py     config -> run directory
  corpus.py     grok labels, run manifests
  evaluate.py   probe scoring protocol
  plots.py
configs/        one yaml per experiment
runs/           <run_id>/{config.yaml, metrics.jsonl, features/M_t.npy}
```

### Learner: a generator, not a training script

```python
@dataclass
class Snapshot:
    t: int
    M: np.ndarray          # 2p x 2p feature matrix
    metrics: dict          # train/test accuracy and loss

class Learner(Protocol):
    def steps(self) -> Iterator[Snapshot]: ...
```

`RFMLearner` yields after each AGOP update. `NNLearner` yields `W_1^T W_1` and the AGOP every
`k` epochs. Adding a third learner later means one new file and nothing else.

### Probe: where the research lives

```python
class Probe:
    causal: bool                              # may it see the future?
    def update(self, snap) -> dict | None: ...
    def finalize(self, traj) -> dict: ...
```

`AGOPAlignment` and `CirculantDeviation` set `causal = False` — they are the paper's baselines
and legitimately need the endpoint. New probes set `causal = True`, and **the harness must
refuse to score a causal probe using anything past step `t`**. Enforce this in code, not by
discipline: leaking the future is the easiest mistake here and the hardest to notice.

### Supporting modules

- **`features.py`** — off-diagonal block extraction, Frobenius normalisation, discrete-log
  reordering, EMA smoothing. Centralising this prevents the same subtle preprocessing bug
  from being reintroduced in five places.
- **`evaluate.py`** — `(probe, corpus) -> table of lead time / FPR / variance`. Once this
  exists, testing a new idea is a 20-line class plus a registry entry. That is the payoff of
  the whole structure and what makes a semester of iteration cheap.

### Practical notes

- Store `M` as float32. 122 x 122 x ~50 steps x ~40 runs is a few hundred MB — keep
  everything, never rerun a training job to test a new probe.
- Skip wandb. A `metrics.jsonl` per run plus a pandas loader is less friction on Kaggle and
  molab, and the corpus loader wants local files anyway.
- Seed everything and record seeds in `config.yaml`. The original repo has its seeds
  commented out.

---

## 7. Immediate next actions

1. Clone `nmallinar/rfm-grokking`; read `models/quadratic_kernel.py` and
   `models/gaussian_kernel.py` for the AGOP update, and `utils.py` for the reordering.
2. Build the Phase 0 scaffold.
3. Run quadratic-kernel RFM on `x+y`, `p=61`, `r=0.5`, 30 iterations; plot alignment and
   circulant deviation against Figure 2B.
4. Only after that gate passes, start Phase 2.
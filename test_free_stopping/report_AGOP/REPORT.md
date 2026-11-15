# F2 (online AGOP settling) as a stopping rule: final report

Status (2026-11-15): protocol fixed, notebook ready (`molab/phase4_agop_final.py`).
**Results pending the run.** The figures, tables and the conclusions below get filled
from `phase4/agop_final/` on HF once the run is done. This file then becomes the
standalone record of F2.

## 1. The question

F2 asks whether the network's features have stopped moving. The features are
M = G^{1/2}, where G is the input AGOP on the training set, normalized to unit
Frobenius norm. Mallinar et al. show that M ends up aligned with its final value as the
network generalizes. F2 turns that into a test-free stopping rule: stop once M has
settled. The earlier experiment (results/phase4/RESULTS.md, 2026-11-14/15) found two
things:

- **MNIST:** the features keep turning long after test accuracy has settled.
- **Modular:** a remaining-angle rule halved the cost of training to the horizon.

That modular result was picked as the best of about 400 settings on 17 runs, so it was
optimistic. This study settles it with held-out runs.

## 2. Protocol

**Runs.** Whole hyperparameter cells, all 4 seeds each. Cells are picked round-robin
over the seed-0 outcome categories (grokker, fast, never, late collapse, …) per variant
and optimizer:

| | Cells | Runs |
|---|---|---|
| Modular x+y | 6 per optimizer | 24 cells |
| Modular x*y, x/y | 2 per optimizer each | 16 cells |
| **Modular total** | | **160 runs** |
| MNIST d3, d8 | 2 per optimizer each | 16 cells, **64 runs** |

Each run is replayed once to record the path of G, M and the per-layer weights on the
unit sphere, at about 220 eval steps.

**Score.** cost_λ = (max test acc − test acc at the stop) + λ·(steps trained after t_ε)/T.

- t_ε is the first step within ε of the best test accuracy; T is the training budget.
- A rule that never stops is scored as stopping at T.
- The headline uses λ = 0.1; λ = 0.03 and 0.3 are reported too, recalibrated for each.

**Rules compared.** Each rule has a grid of settings.

| Family | Rule | Causal? |
|---|---|---|
| bound | remaining-angle bound R = θ_lag·q/(1−q), with q from the decay of the speed | yes |
| rel_delta | angle over the last lag f·t < ε (the bound without κ) | yes |
| const_delta | angle between M_t and M_{t−Δ}, with fixed Δ = c·T, < ε: **the simplest F2** | yes |
| msd, geweke | settling tests: MSD exponent; Geweke-style early-vs-late Z | yes |
| fixed_frac | stop at c·T whatever happens: **the bar F2 must clear** | yes |
| horizon | never stop | yes |
| oracle_cos | first t with cos(M_t, M_T) ≥ c: Mallinar's alignment with the final AGOP | no (needs M_T) |
| oracle_settle | first t after which the angle to M_T stays < δ | no (needs M_T) |
| oracle_teps | stop exactly at t_ε | no |

Objects: M, G, and W (weights; all layers must agree). Optional norm gate: log‖w‖ must
have settled too.

**Calibrate, then test.** For each family, choose the setting with the lowest mean cost
on the calibration runs, then score that one setting on the test runs.

- **Seed split:** calibrate on seeds 0–1, test on seeds 2–3 (same hyperparameters).
- **Transfer split:** modular x+y → x*y and x/y; MNIST depth 3 → depth 8.
- 95% CIs: bootstrap over hyperparameter cells.
- Sign test over cells, against the horizon and against the calibrated fixed fraction.
- The best setting chosen on the test runs themselves ("hindsight") is reported as an
  upper bound for each family.

**What counts as conclusive:**

- **"Works on modular":** on both splits, the calibrated F2 rule beats the horizon *and*
  the calibrated fixed fraction. The CI on the difference must exclude 0, with few
  premature stops.
- **"Does not work on MNIST":** the oracles that know the final features are no better
  than the horizon, or than a fixed fraction. Then no online F2 rule can be either, and
  the calibrated online rules are no better on held-out runs.

## 3. Results

_Pending the run._

## 4. How close is the online AGOP to the final AGOP?

_Pending: alignment cos(M_t, M_T) around t_ε, and the online estimates against the true
remaining angle._

## 5. Conclusion

_Pending._

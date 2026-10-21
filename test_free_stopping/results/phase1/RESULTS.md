# Phase 1 results (Phase 1 + Phase 1b clean-up run)

Raw reports: HF dataset `advay333/tfs-phase1`, under `phase1/` and `phase1b/`.
Figures: [`repro1b_teacher_student.png`](repro1b_teacher_student.png) and
[`repro1b_mnist.png`](repro1b_mnist.png).

## Determinism (gate G1-det): passed

| Check | Result |
|---|---|
| CUDA, 16 configs (4 apps × 4 optimizers), 20k steps, 2 keyframe intervals | 16/16 bit-identical pairs, 16/16 replay L0, zero error. MNIST SGD now exercised at α = 1. |
| CPU, unpinned (Phase 1 environment), 2k steps | 15/16. MNIST depth-8 Adam+L2: the two trainings differed. Reproduces the Phase 1 drift (2/16 there). |
| CPU, pinned (`tfs/cpu_env.py`), 2k steps | 16/16 bit-identical pairs, 16/16 replay L0. |

Cause: load-dependent CPU numerics. The drift is intermittent (1–2 of 16 configs per
pass) and appears only under load. It never appears on an idle machine and never on the
GPU. Pinning removes it in the test run. With a 1–2/16 failure rate, a single clean pass
of 16 is evidence rather than proof (about a 20% chance of zero failures by luck). The
corpus is GPU-only, so this affects only development and tests.

Molab reports the same count from `os.cpu_count()` and the affinity mask (20 CPUs), so
oversubscription from a wrong core count was not the trigger there. That leaves dynamic
threading or oneDNN. Both are now pinned off; which one was responsible was not isolated.

## Reproductions (gate G1)

| Setup | Result | Verdict |
|---|---|---|
| Modular x+y, p = 97 (Phase 1) | Test 0 → 1.0 long after training fits (about 13k steps at 30% data, 2.3k at 50%); weight norm rises then falls (Omnigrok Fig. 7a) | ✅ |
| Teacher–student (Omnigrok Fig. 2b), RMS-error accuracy | γ = 0: never generalizes (max 0.02). γ = 0.03: grokks, test 0 → 0.95 between 1e4 and 1e5 (still rising at the horizon). γ = 1: fast (0.95 by 3k). α = 0.5: test follows train | ✅ |
| MNIST depth 3, α = 8 (Omnigrok Fig. 3d) | Train fits by step 550. Steps until test accuracy is within 1% of its best: 88k at γ = 0.01, 52k at γ = 0.03, 23k at γ = 0.1 (roughly ∝ γ^-0.6) | ✅ |
| MNIST depth 3, α = 1 control | Test peaks 0.919 at step 150, then collapses to 0.574 by 3e5 while train stays at 100% | 🎁 late-collapse case |
| Deep Grokking two-stage (depth 8/12, n 5k/7k) | A sharp joint train+test transition after a depth-induced plateau (about 2–4k steps), then a slow test creep from about 0.90 to 0.96 up to about 75k. No sharp second surge. | ❌ two-stage not reproduced; dropped as a target |

## Other findings

- **SGD at α = 8 on MNIST depth 3 diverges at every learning rate down to 1e-6** (initial
  loss about 2.5e7). Depth 8 at α = 8 trains, but slowly: the best of the scan was
  SGD+momentum at 1e-5, which reached 61% train accuracy in 5k steps.
- **Storage:** summaries were 85–93% of each Phase 1 run (dense WᵀW). Removed; Phase 1b
  runs are 3.7 MB (teacher–student), 31 MB (MNIST d3 at 1e5 steps), and 188–280 MB
  (MNIST d8/d12).
- **Throughput:** aggregate steps per second were 488 at 8 GPU workers, 109 at 16 and 24
  at 32 (MPS at 16 workers: 125), with GPU utilization at a median of 19%. The cause was
  CPU oversubscription: GPU workers had no thread share, so each started 20 threads on
  a 20-CPU machine. On top of that, the batch order was rebuilt on the CPU every step.
  Both are fixed. Results were bit-identical at every concurrency level and under MPS.

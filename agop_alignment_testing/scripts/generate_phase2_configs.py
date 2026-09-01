"""Generates the Phase 2 corpus configs (project.md Section 5, Phase 2). Writes one
YAML per run, matching the project's "one yaml per experiment" convention -- run this
once, inspect the files, then run them via scripts/run_batch.py.

Two output directories, because the corpus is being rebuilt in stages:

  configs/phase2/           the runs to execute now -- the two grokker categories.
  configs/phase2_deferred/  the rest of the planned corpus, generated so the plan is
                            on disk and reviewable, but not executed yet.

Horizons differ from the first corpus build. RFM goes 30 -> 50 iterations: quadratic
groks around t=14 but gaussian only around t=18-20 and is still climbing at t=30,
which leaves almost no post-transition tail for a trailing-window probe to fit. NN
goes 50 -> 300 epochs, snapshotting every epoch: at 50 epochs the NN's circulant
deviation is still moving, so a reversal cannot be told apart from an incomplete
curve. See NN_SNAPSHOT_EVERY below for why the stride stays at 1.

NN runs are pinned to device: cpu. Measured on this machine (RTX 4060 Laptop), a
50-epoch run takes 15.7s on CPU and 18.7s on CUDA with a warm context -- width 1024
at batch size 32 is small enough that per-step kernel-launch overhead outweighs the
compute saving. The device knob stays for larger configurations.

Categories (see corpus.CATEGORY_EXPECTS_GROK):
    grokker         4 ops x {quadratic, gaussian} x 3 seeds (RFM) + 4 ops x 3 seeds (NN)
    non_grokker     RFM low training fraction, RFM random labels, NN random labels
    partial_learner RFM at sub-threshold training fractions -- see below
    fast_learner    oracle random-circulant M_0 (the paper's Eq. 9)
    held-out        p=97, reserved for Phase 4; never used to tune a probe

The partial_learner category is new. Measured on quadratic x+y at p=61 out to 100
iterations, training fractions 0.20-0.30 give runs where circulant deviation falls
several-fold (0.0155 -> 0.0028 at r=0.30) while test accuracy stays at or near zero:

    r      acc@30  acc@100   dev@30   dev@100
    0.05    0.000    0.000   0.01590  0.01591
    0.10    0.000    0.000   0.01551  0.01552
    0.20    0.000    0.000   0.01175  0.01091
    0.25    0.004    0.018   0.00711  0.00590
    0.30    0.053    0.122   0.00356  0.00279

r <= 0.10 is a genuine non-grokker: dead at 3x the horizon with deviation pinned.
r in [0.20, 0.30] is something else -- real feature learning without generalization
-- and it is the sharpest negative in the corpus, because any probe that fires on
"features are becoming circulant" will fire here and be wrong.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUN_NOW_DIR = ROOT / "configs" / "phase2"
DEFERRED_DIR = ROOT / "configs" / "phase2_deferred"

OPS = ["x+y", "x-y", "x*y", "x/y"]
SEEDS = [0, 1, 2]

RFM_ITERS = 50
NN_EPOCHS = 300
# Every epoch. Do not raise this to save disk. Subsampling is not smoothing: project.md
# warns that single-epoch AdamW increments are minibatch noise, but the remedy is a
# filter (features.ema_smooth, causal) applied to a dense series, not decimation --
# skipping epochs aliases the noise instead of reducing it and discards the samples the
# filter needs. It also quantises lead time, the headline Phase 3 measurement, to the
# stride, and the NN transition is narrow (~epoch 14). project.md Section 6 is explicit:
# "keep everything, never rerun a training job to test a new probe."
NN_SNAPSHOT_EVERY = 1


def op_slug(op: str) -> str:
    return op.replace("+", "plus").replace("-", "minus").replace("*", "mult").replace("/", "div")


def write(out_dir: Path, run_id: str, config: dict) -> str:
    config = {"run_id": run_id, **config}
    with open(out_dir / f"{run_id}.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return run_id


def rfm(operation: str, seed: int, kernel_type: str = "quadratic", p: int = 61, **overrides) -> dict:
    config = {
        "learner": "rfm",
        "operation": operation,
        "p": p,
        "training_fraction": 0.5,
        "iters": RFM_ITERS,
        "kernel_type": kernel_type,
        "agop_power": 0.5,
        "seed": seed,
    }
    if kernel_type == "gaussian":
        config["bandwidth"] = 2.5
    config.update(overrides)
    return config


def nn(operation: str, seed: int, p: int = 61, **overrides) -> dict:
    config = {
        "learner": "nn",
        "operation": operation,
        "p": p,
        "training_fraction": 0.5,
        "width": 1024,
        "epochs": NN_EPOCHS,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 1.0,
        "optimizer": "adamw",
        "snapshot_every": NN_SNAPSHOT_EVERY,
        "device": "cpu",
        "seed": seed,
    }
    config.update(overrides)
    return config


def generate_grokkers() -> list:
    """The two categories to run now: 24 RFM + 12 NN = 36 runs."""
    generated = []
    for op in OPS:
        for kernel_type in ["quadratic", "gaussian"]:
            for seed in SEEDS:
                run_id = f"grok_rfm_{kernel_type}_{op_slug(op)}_seed{seed}"
                generated.append(write(RUN_NOW_DIR, run_id, rfm(op, seed, kernel_type)))
    for op in OPS:
        for seed in SEEDS:
            run_id = f"grok_nn_{op_slug(op)}_seed{seed}"
            generated.append(write(RUN_NOW_DIR, run_id, nn(op, seed)))
    return generated


def generate_deferred() -> list:
    """Planned but not executed yet: negatives, fast learners, and the held-out p=97
    arm. Written to configs/phase2_deferred/ so the corpus plan is reviewable on disk.
    """
    generated = []

    # Non-grokkers: training fractions verified dead at 100 iterations.
    for frac in [0.05, 0.10]:
        for op in OPS:
            for seed in [0, 1]:
                run_id = f"nongrok_rfm_lowfrac{frac:.2f}_{op_slug(op)}_seed{seed}"
                generated.append(
                    write(DEFERRED_DIR, run_id, rfm(op, seed, training_fraction=frac))
                )

    # Non-grokkers: random-label control, both learners.
    for op in OPS:
        for seed in SEEDS:
            run_id = f"nongrok_rfm_random_labels_{op_slug(op)}_seed{seed}"
            generated.append(write(DEFERRED_DIR, run_id, rfm(op, seed, random_labels=True)))
    for seed in [0, 1]:
        run_id = f"nongrok_nn_random_labels_x_plus_y_seed{seed}"
        generated.append(write(DEFERRED_DIR, run_id, nn("x+y", seed, random_labels=True)))
    for seed in [0, 1]:
        run_id = f"nongrok_nn_lowfrac0.10_x_plus_y_seed{seed}"
        generated.append(
            write(DEFERRED_DIR, run_id, nn("x+y", seed, training_fraction=0.10))
        )

    # Partial learners: structure without generalization.
    for frac in [0.20, 0.25, 0.30]:
        for op in OPS:
            for seed in [0, 1]:
                run_id = f"partial_rfm_frac{frac:.2f}_{op_slug(op)}_seed{seed}"
                generated.append(
                    write(DEFERRED_DIR, run_id, rfm(op, seed, training_fraction=frac))
                )

    # Fast learners: oracle random-circulant M_0, i.e. the paper's Eq. 9.
    for op in OPS:
        for seed in [0, 1]:
            run_id = f"fastlearn_rfm_{op_slug(op)}_seed{seed}"
            generated.append(write(DEFERRED_DIR, run_id, rfm(op, seed, fast_learner=True)))

    # Held-out: p=97, one seed, never used to tune a probe (project.md Phase 4,
    # "validated on held-out (p, operation) pairs"). One seed rather than two keeps
    # the gaussian solves affordable -- at p=97 and r=0.5 that is a 4704 x 4704 dense
    # solve per iteration.
    for op in OPS:
        for kernel_type in ["quadratic", "gaussian"]:
            run_id = f"heldout_rfm_{kernel_type}_p97_{op_slug(op)}_seed0"
            generated.append(write(DEFERRED_DIR, run_id, rfm(op, 0, kernel_type, p=97)))

    return generated


def main():
    RUN_NOW_DIR.mkdir(parents=True, exist_ok=True)
    DEFERRED_DIR.mkdir(parents=True, exist_ok=True)

    grokkers = generate_grokkers()
    deferred = generate_deferred()

    print(f"configs/phase2/          {len(grokkers)} configs (run now)")
    for run_id in grokkers:
        print(f"  {run_id}")
    print(f"\nconfigs/phase2_deferred/ {len(deferred)} configs (planned, not run yet)")
    for run_id in deferred:
        print(f"  {run_id}")


if __name__ == "__main__":
    main()

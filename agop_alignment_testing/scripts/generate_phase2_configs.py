"""Generates the Phase 2 frozen-corpus configs (project.md Section 5, Phase 2).
Writes one YAML per run under configs/phase2/, matching the project's "one yaml per
experiment" convention -- run this once, inspect the files, then run them via
scripts/run_batch.py.

Categories:
    grokkers      -- 4 ops x {quadratic, gaussian} x 3 seeds (RFM) + 4 ops x 3 seeds (NN)
    non_grokkers  -- low training fraction (RFM), NN with no weight decay, random-label control
    fast_learners -- fast_learner=true (RFM), 4 ops, 1 seed
"""
from __future__ import annotations

from pathlib import Path

import yaml

OUT_DIR = Path(__file__).resolve().parent.parent / "configs" / "phase2"
OPS = ["x+y", "x-y", "x*y", "x/y"]
SEEDS = [0, 1, 2]


def op_slug(op: str) -> str:
    return op.replace("+", "plus").replace("-", "minus").replace("*", "mult").replace("/", "div")


def write(run_id: str, config: dict) -> None:
    config = {"run_id": run_id, **config}
    with open(OUT_DIR / f"{run_id}.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    # --- Grokkers: RFM, 4 ops x {quadratic, gaussian} x 3 seeds ---
    for op in OPS:
        for kernel_type in ["quadratic", "gaussian"]:
            for seed in SEEDS:
                run_id = f"grok_rfm_{kernel_type}_{op_slug(op)}_seed{seed}"
                write(
                    run_id,
                    {
                        "learner": "rfm",
                        "operation": op,
                        "p": 61,
                        "training_fraction": 0.5,
                        "iters": 30,
                        "kernel_type": kernel_type,
                        "bandwidth": 2.5,
                        "agop_power": 0.5,
                        "seed": seed,
                    },
                )
                generated.append(run_id)

    # --- Grokkers: NN equivalents, 4 ops x 3 seeds ---
    for op in OPS:
        for seed in SEEDS:
            run_id = f"grok_nn_{op_slug(op)}_seed{seed}"
            write(
                run_id,
                {
                    "learner": "nn",
                    "operation": op,
                    "p": 61,
                    "training_fraction": 0.5,
                    "width": 1024,
                    "epochs": 50,
                    "batch_size": 32,
                    "lr": 0.001,
                    "weight_decay": 1.0,
                    "snapshot_every": 1,
                    "seed": seed,
                },
            )
            generated.append(run_id)

    # --- Non-grokkers: low training fraction (RFM, quadratic), below the ~25%
    # threshold in Appendix Fig 6 ---
    for op in OPS:
        run_id = f"nongrok_rfm_lowfrac0.10_{op_slug(op)}_seed0"
        write(
            run_id,
            {
                "learner": "rfm",
                "operation": op,
                "p": 61,
                "training_fraction": 0.10,
                "iters": 30,
                "kernel_type": "quadratic",
                "agop_power": 0.5,
                "seed": 0,
            },
        )
        generated.append(run_id)

    run_id = "nongrok_rfm_lowfrac0.05_x_plus_y_seed0"
    write(
        run_id,
        {
            "learner": "rfm",
            "operation": "x+y",
            "p": 61,
            "training_fraction": 0.05,
            "iters": 30,
            "kernel_type": "quadratic",
            "agop_power": 0.5,
            "seed": 0,
        },
    )
    generated.append(run_id)

    # --- Non-grokkers: NN with no weight decay (Appendix Fig 5, left panel) ---
    for seed in [0, 1]:
        run_id = f"nongrok_nn_no_decay_x_plus_y_seed{seed}"
        write(
            run_id,
            {
                "learner": "nn",
                "operation": "x+y",
                "p": 61,
                "training_fraction": 0.5,
                "width": 1024,
                "epochs": 50,
                "batch_size": 32,
                "lr": 0.001,
                "weight_decay": 0.0,
                "snapshot_every": 1,
                "seed": seed,
            },
        )
        generated.append(run_id)

    # --- Non-grokkers: random-label control (RFM, quadratic, x+y) ---
    for seed in [0, 1]:
        run_id = f"nongrok_rfm_random_labels_x_plus_y_seed{seed}"
        write(
            run_id,
            {
                "learner": "rfm",
                "operation": "x+y",
                "p": 61,
                "training_fraction": 0.5,
                "iters": 30,
                "kernel_type": "quadratic",
                "agop_power": 0.5,
                "seed": seed,
                "random_labels": True,
            },
        )
        generated.append(run_id)

    # --- Fast learners: fast_learner=true (RFM, quadratic), 4 ops, 1 seed ---
    for op in OPS:
        run_id = f"fastlearn_rfm_{op_slug(op)}_seed0"
        write(
            run_id,
            {
                "learner": "rfm",
                "operation": op,
                "p": 61,
                "training_fraction": 0.5,
                "iters": 30,
                "kernel_type": "quadratic",
                "agop_power": 0.5,
                "seed": 0,
                "fast_learner": True,
            },
        )
        generated.append(run_id)

    print(f"wrote {len(generated)} configs to {OUT_DIR}")
    for run_id in generated:
        print(f"  {run_id}")


if __name__ == "__main__":
    main()

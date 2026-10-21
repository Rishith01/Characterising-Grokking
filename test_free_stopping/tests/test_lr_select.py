from tfs.sweep.grid import expand
from tfs.sweep.lr_select import scan_row, select_lrs, variant_label


def _row(variant, optim, alpha, lr, stable):
    return {"variant": variant, "optim": optim, "alpha": alpha, "lr": lr, "stable": stable}


def test_adam_keeps_published_lr_when_stable():
    rows = [_row("teacher_student", "adamw", 1.0, lr, True) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]
    (sel,) = select_lrs(rows)
    assert sel["chosen_lr"] == 3e-4 and sel["max_stable_lr"] == 3e-3


def test_adam_falls_back_below_unstable_published_lr():
    rows = [_row("mnist d8", "adam_l2", 4.0, lr, lr < 1e-3) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]
    (sel,) = select_lrs(rows)
    assert sel["chosen_lr"] == 3e-4


def test_sgd_steps_below_and_takes_min_over_alpha():
    grid = [1.0, 0.3, 0.1, 3e-2, 1e-2, 3e-3, 1e-3]
    max_stable = {1.0: 0.3, 2.0: 3e-2, 4.0: None}
    rows = [_row("modular x+y", "sgd", a, lr, m is not None and lr <= m)
            for a, m in max_stable.items() for lr in grid]
    summary, *detail = select_lrs(rows)
    assert [d["chosen_lr"] for d in detail] == [0.1, 1e-2, None]
    assert summary["chosen_lr"] == 1e-2
    assert "drop" in detail[2]["note"]


def test_scan_row_reads_training_telemetry_only(dummy_run):
    row = scan_row(dummy_run)
    assert row["variant"] == "dummy" and row["status"] == "complete"
    assert row["stable"] and row["loss_end"] < row["loss_0"]


def test_phase2_lr_scan_sweep():
    cfgs = expand("configs/sweeps/phase2_lr_scan.yaml")
    assert len(cfgs) == 516
    labels = {variant_label(c) for c in cfgs}
    assert labels == {"teacher_student", "modular x+y", "modular x*y", "modular x^2+xy+y^2", "mnist d3", "mnist d8"}
    sgd_alphas = {(variant_label(c), c.model.init_scale) for c in cfgs if c.optim.name == "sgd"}
    assert ("mnist d3", 8.0) not in sgd_alphas and ("mnist d3", 4.0) in sgd_alphas

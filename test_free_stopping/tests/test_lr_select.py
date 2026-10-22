from tfs.sweep.grid import expand
from tfs.sweep.lr_select import scan_row, select_lrs, variant_label


def _row(variant, optim, alpha, lr, stable, reason=None):
    return {"variant": variant, "optim": optim, "alpha": alpha, "lr": lr, "stable": stable,
            "reason": reason if reason is not None else ("" if stable else "diverged")}


def test_adam_keeps_published_lr_when_stable():
    rows = [_row("teacher_student", "adamw", 1.0, lr, True) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]
    (sel,) = select_lrs(rows)
    assert sel["chosen_lr"] == 3e-4 and sel["max_stable_lr"] == 3e-3


def test_adam_falls_back_below_unstable_published_lr():
    rows = [_row("mnist d8", "adam_l2", 4.0, lr, lr < 1e-3) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]
    (sel,) = select_lrs(rows)
    assert sel["chosen_lr"] == 3e-4


def test_sgd_margin_only_against_instability():
    grid = [1.0, 0.3, 0.1, 3e-2, 1e-2, 3e-3, 1e-3]
    rows = [_row("modular x+y", "sgd", 1.0, lr, True) for lr in grid]  # usable up to the grid top: keep it
    rows += [_row("modular x+y", "sgd", 2.0, lr, lr <= 3e-2) for lr in grid]  # 0.1 diverged: step down
    rows += [_row("modular x+y", "sgd", 4.0, lr, lr <= 3e-2, "not fitting") for lr in grid]  # 0.1 too slow: keep
    rows += [_row("modular x+y", "sgd", 8.0, lr, False) for lr in grid]
    summary, *detail = select_lrs(rows)
    assert [d["chosen_lr"] for d in detail] == [1.0, 1e-2, 3e-2, None]
    assert summary["chosen_lr"] == 1e-2
    assert "drop" in detail[3]["note"]


def test_classify_rejects_mean_predictor():
    """Phase 2a: MNIST depth 8 at alpha 1 reaches the mean-prediction loss and stops."""
    from tfs.sweep.lr_select import classify

    base = {"status": "complete", "loss_0": 0.13, "loss_min": 0.0899}
    assert classify({**base, "loss_end": 0.09, "train_acc_end": 0.12})["reason"] == "not fitting"
    assert classify({**base, "loss_end": 0.09, "train_acc_end": 0.99})["stable"]
    assert classify({"status": "diverged", "loss_0": 1.0})["reason"] == "diverged"


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

from tfs.evaluation.pilot import g2_checks, project


def _r(variant, optim, category, t_eps, **kw):
    return {"variant": variant, "optim": optim, "category": category, "t_eps": t_eps, "replay_level": "L0",
            "status": "complete", "wall_s": 600.0, "size_mb": 30.0, "steps": 100_000, **kw}


def test_g2_spread_categories_and_replay():
    rows = [_r("mnist d3", "adamw", "grokker", 88_000), _r("mnist d3", "adamw", "fast", 1_000),
            _r("mnist d3", "sgd", "never", 100_000), _r("teacher_student", "adamw", "fast", 3_000),
            _r("teacher_student", "sgd", "fast", 4_000, replay_level="failed")]
    by_app = {r["variant"]: r for r in g2_checks(rows)}
    assert by_app["mnist d3"]["t_eps_spread_decades"] == 1.94 and by_app["mnist d3"]["spread_ok"]
    assert by_app["mnist d3"]["has_never"] and by_app["mnist d3"]["replay_ok"]
    assert not by_app["teacher_student"]["spread_ok"] and not by_app["teacher_student"]["replay_ok"]
    per_opt = g2_checks(rows, by=("variant", "optim"))
    assert len(per_opt) == 4


def test_projection_scales_horizon_and_concurrency():
    rows = [_r("mnist d3", "adamw", "fast", 10), _r("teacher_student", "adamw", "fast", 10, steps=300_000,
                                                    wall_s=1800.0, size_mb=12.0)]
    out = {r["variant"]: r for r in project(rows, workers=16, runs={"mnist d3": 128, "teacher_student": 576})}
    assert out["mnist d3"]["train_gpu_h"] == round(128 * 600 / 16 / 3600, 1)
    assert out["teacher_student"]["wall_s_per_run"] == 600 and out["teacher_student"]["size_mb_per_run"] == 4.0
    assert out["TOTAL"]["size_ok"]
    long = {r["variant"]: r for r in project(rows, 16, runs={"teacher_student": 576}, ts_horizon=300_000)}
    assert long["teacher_student"]["wall_s_per_run"] == 1800


def test_phase2_sweeps_expand_with_float_hyperparameters():
    from tfs.sweep.grid import expand

    sizes = {}
    for name in ("phase2_gamma", "phase2_det", "phase2_check"):
        cfgs = expand(f"configs/sweeps/{name}.yaml")
        sizes[name] = len(cfgs)
        # YAML reads 3e-4 as a string; ints would change the run-id hash
        assert all(isinstance(c.optim.lr, float) and isinstance(c.optim.weight_decay, float) for c in cfgs)
    assert sizes == {"phase2_gamma": 146, "phase2_det": 24, "phase2_check": 43}
    old = {c.run_id for c in expand("configs/sweeps/phase2_gamma.yaml")}
    assert not old & {c.run_id for c in expand("configs/sweeps/phase2_check.yaml")}

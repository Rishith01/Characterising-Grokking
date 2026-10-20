from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tfs.config import OptimConfig, RunConfig
from tfs.training import determinism

determinism.apply()

PROJECT = Path(__file__).resolve().parents[1]
OPTIM = {
    "adamw": OptimConfig("adamw", lr=1e-3, weight_decay=0.1),
    "adam_l2": OptimConfig("adam_l2", lr=1e-3, weight_decay=1e-3),
    "sgd_momentum": OptimConfig("sgd_momentum", lr=0.05, weight_decay=1e-3, momentum=0.9),
    "sgd": OptimConfig("sgd", lr=0.1, weight_decay=1e-3),
}


@pytest.fixture(scope="session")
def dummy_cfg() -> RunConfig:
    return RunConfig.from_yaml(PROJECT / "configs" / "runs" / "dummy.yaml")


def make_cfg(base: RunConfig, optim: str = "adamw", **kw) -> RunConfig:
    return replace(base, optim=OPTIM[optim], **kw)


def train(cfg: RunConfig, root: Path) -> Path:
    from tfs.training.trainer import Trainer

    return Trainer(cfg, root, "cpu").run()


@pytest.fixture(scope="session")
def dummy_run(dummy_cfg, tmp_path_factory) -> Path:
    """One trained dummy run, shared read-only by tests that don't modify it."""
    return train(dummy_cfg, tmp_path_factory.mktemp("runs"))

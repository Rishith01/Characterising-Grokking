"""Run configuration: frozen dataclasses, YAML/dict (de)serialisation, deterministic run ids.

A RunConfig fully determines a run (DESIGN.md §4): together with the code version and
the environment, it is everything needed to regenerate the data and replay training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass(frozen=True)
class TaskConfig:
    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    depth: int  # number of Linear layers (depth 3 = 2 hidden layers)
    width: int
    activation: str
    init_scale: float = 1.0  # Omnigrok's alpha: every parameter is multiplied by this at init


@dataclass(frozen=True)
class OptimConfig:
    name: str  # adamw | adam_l2 | sgd_momentum | sgd
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.0
    betas: tuple = (0.9, 0.98)


@dataclass(frozen=True)
class RecordConfig:
    log_every: int = 50
    summary_every: int = 500
    keyframe_every: int = 10_000
    n_fingerprint_proj: int = 8


@dataclass(frozen=True)
class RunConfig:
    task: TaskConfig
    model: ModelConfig
    optim: OptimConfig
    loss: str  # "ce" | "mse"
    steps: int
    seed: int
    batch_size: Optional[int] = None  # None = full batch
    record: RecordConfig = field(default_factory=RecordConfig)

    def __post_init__(self):
        r = self.record
        if r.summary_every % r.log_every or r.keyframe_every % r.summary_every:
            raise ValueError(
                "schedules must nest: summary_every must be a multiple of log_every, and "
                "keyframe_every a multiple of summary_every"
            )

    # ---- (de)serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["optim"]["betas"] = list(d["optim"]["betas"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        optim = dict(d["optim"])
        if "betas" in optim:
            optim["betas"] = tuple(optim["betas"])
        return cls(
            task=TaskConfig(**d["task"]),
            model=ModelConfig(**d["model"]),
            optim=OptimConfig(**optim),
            record=RecordConfig(**d.get("record", {})),
            loss=d["loss"],
            steps=d["steps"],
            seed=d["seed"],
            batch_size=d.get("batch_size"),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RunConfig":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def with_seed(self, seed: int) -> "RunConfig":
        return replace(self, seed=seed)

    # ---- identity ----------------------------------------------------------------
    @property
    def app(self) -> str:
        return self.task.name

    @property
    def run_id(self) -> str:
        """<app>__<readable axes>__<8-char hash of the canonical config>. Deterministic,
        so regenerating a sweep regenerates the same ids (what makes resuming work)."""
        o, m = self.optim, self.model
        readable = f"{o.name}_wd{o.weight_decay:g}_a{m.init_scale:g}_s{self.seed}"
        return f"{self.app}__{readable}__{_hash(self.to_dict())}"

    @property
    def group_id(self) -> str:
        """run_id with the seed removed: runs differing only by seed share a group."""
        d = self.to_dict()
        d.pop("seed")
        o, m = self.optim, self.model
        return f"{self.app}__{o.name}_wd{o.weight_decay:g}_a{m.init_scale:g}__{_hash(d)}"


def _hash(d: dict) -> str:
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


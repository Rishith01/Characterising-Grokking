"""Signal and SignalConfig: the abstract classes every metric implements (DESIGN.md §13.2).

A signal turns a run into a time series. Its config holds only the settings that change
the series (layer aggregation, normalization, ...). Stopping hyperparameters (threshold,
patience, smoothing) live in `stopping.rule.RuleConfig`, so calibrating them never
recomputes a signal.

A config is a frozen dataclass; its hash (over the name, the version and every field)
names the cached trace. Bump `version` whenever a signal's computation changes.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar, Optional

from .source import Observation, RunContext

CADENCES = ("log", "summary", "eval")


@dataclass(frozen=True)
class SignalConfig:
    name: ClassVar[str] = ""
    version: ClassVar[int] = 1

    def to_dict(self) -> dict:
        return {"signal": self.name, "version": self.version, **asdict(self)}

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    @property
    def key(self) -> str:
        return f"{self.name}-{self.hash()}"

    def build(self) -> "Signal":
        return SIGNALS[self.name][1](self)


class Signal(ABC):
    """Streaming: reset(ctx) once, then update(obs) at every step of its cadence, in
    increasing step order. update returns the row to record ({"value": ..., ...}) or
    None to record nothing at this step. Any history lives in the signal's own state."""

    uses_heldout: ClassVar[bool] = False

    def __init__(self, cfg: SignalConfig):
        self.cfg = cfg
        self.ctx: Optional[RunContext] = None

    @property
    @abstractmethod
    def requires(self) -> frozenset[str]: ...

    @property
    @abstractmethod
    def cadence(self) -> str: ...

    def reset(self, ctx: RunContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def update(self, obs: Observation) -> Optional[dict[str, float]]: ...


SIGNALS: dict[str, tuple[type[SignalConfig], type[Signal]]] = {}


def register(cfg_cls: type[SignalConfig]):
    def deco(sig_cls: type[Signal]) -> type[Signal]:
        if not cfg_cls.name:
            raise ValueError(f"{cfg_cls.__name__} has no name")
        if cfg_cls.name in SIGNALS:
            raise ValueError(f"signal {cfg_cls.name!r} registered twice")
        SIGNALS[cfg_cls.name] = (cfg_cls, sig_cls)
        return sig_cls
    return deco


def config_from_dict(d: dict) -> SignalConfig:
    d = dict(d)
    name = d.pop("signal")
    version = d.pop("version", None)
    cfg_cls = SIGNALS[name][0]
    if version is not None and version != cfg_cls.version:
        raise ValueError(f"{name}: stored version {version}, code version {cfg_cls.version}")
    return cfg_cls(**d)

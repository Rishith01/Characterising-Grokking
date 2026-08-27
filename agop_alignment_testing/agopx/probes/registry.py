"""Probe registry (project.md Section 6): a new probe is a 20-line class plus one entry
here -- that is the payoff of decoupling learners from measurement.
"""
from __future__ import annotations

_REGISTRY: dict = {}


def register(name: str):
    def _decorator(cls):
        if name in _REGISTRY:
            raise ValueError(f"probe {name!r} already registered")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str):
    return _REGISTRY[name]


def available() -> list:
    return sorted(_REGISTRY)

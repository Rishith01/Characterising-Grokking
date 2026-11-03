"""The Phase 4 methods with **uncalibrated** rule settings (METRICS.md §3-4).

These are placeholders for looking at the signals in the pilot: Phase 5 calibrates every
threshold and patience on seeds {0, 1} and scores on seeds {2, 3}. B6 (oracle) is in
stopping.heldout.b6_method. S1, S1b and F2 are not built yet.
"""
from __future__ import annotations

from .method import Method
from .rule import RuleConfig
from .signals import (DisparityConfig, ERankConfig, EvidenceConfig, HtsrAlphaConfig, HtsrTrapsConfig,
                      LabelWaveConfig, StepConfig, TrainLossConfig, WeightNormConfig)


def default_methods(budget: int = 50_000, replay: bool = True) -> list[Method]:
    """replay=False leaves out the methods that need replayed models (B2, B3 replay form,
    B5 traps), which are the expensive ones."""
    ms = [
        Method("B0 fixed budget", ((StepConfig(), RuleConfig(threshold=budget - 0.5, direction="above",
                                                              smoother="none")),)),  # stops at step == budget
        Method("B1 train-loss patience", ((TrainLossConfig(), RuleConfig(
            mode="plateau", direction="below", threshold=0.01, patience=20)),)),
        Method("N1 weight-norm settles", ((WeightNormConfig(), RuleConfig(
            column="abs_rate", direction="below", threshold=0.01, patience=20, smoother="ema", smooth_param=0.2)),)),
        Method("R1 effective rank settles", ((ERankConfig(), RuleConfig(
            column="abs_rate", direction="below", threshold=0.01, patience=4, smoother="none")),)),
        Method("B3 gradient disparity (summary)", ((DisparityConfig(), RuleConfig(
            mode="plateau", direction="below", threshold=0.0, patience=5, smoother="none")),)),
        Method("B4 label wave", ((LabelWaveConfig(), RuleConfig(
            mode="plateau", direction="below", threshold=0.0, patience=20, smoother="median", smooth_param=5)),)),
        Method("B5 HTSR alpha", ((HtsrAlphaConfig(), RuleConfig(
            direction="below", threshold=2.0, patience=2, smoother="none")),)),
    ]
    if replay:
        ms += [
            Method("B2 Mahsereci EB", ((EvidenceConfig(), RuleConfig(
                direction="above", threshold=0.0, patience=3, smoother="ema", smooth_param=0.3)),)),
            Method("B3 gradient disparity (replay)", ((DisparityConfig(form="replay"), RuleConfig(
                mode="plateau", direction="below", threshold=0.0, patience=10, smoother="ema", smooth_param=0.3)),)),
            Method("B5 HTSR alpha or traps", (
                (HtsrAlphaConfig(), RuleConfig(direction="below", threshold=2.0, patience=2, smoother="none")),
                (HtsrTrapsConfig(), RuleConfig(direction="above", threshold=0.5, patience=2, smoother="none")),
            )),
        ]
    return ms


def signal_configs(methods: list[Method]) -> list:
    seen, out = set(), []
    for m in methods:
        for s in m.signals():
            if s.key not in seen:
                seen.add(s.key)
                out.append(s)
    return out

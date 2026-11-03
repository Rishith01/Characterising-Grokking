"""Importing this package registers every test-free signal. B6 (held-out loss) lives in
stopping.heldout and registers only when that module is imported."""
from .basic import StepConfig, TrainLossConfig, WeightNormConfig
from .gradients import DisparityConfig, EvidenceConfig
from .label_wave import LabelWaveConfig
from .spectra import ERankConfig, HtsrAlphaConfig, HtsrTrapsConfig

__all__ = ["StepConfig", "TrainLossConfig", "WeightNormConfig", "DisparityConfig", "EvidenceConfig",
           "LabelWaveConfig", "ERankConfig", "HtsrAlphaConfig", "HtsrTrapsConfig"]

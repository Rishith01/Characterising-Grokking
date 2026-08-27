"""
Metric callbacks for tracking grokking diagnostics.

Provides a ``MetricCallback`` base class and several built-in metrics.
Users can define new metrics by sub-classing and (optionally) registering
them via ``@register_metric``.

Repos unified:
  - nmallinar/rfm-grokking   (weight-norm tracking)
  - Tikquuss/grokking_fda    (activation / FDA-style metrics)
  - neelnanda-io/Grokking    (representation-level diagnostics)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Type
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Metric Registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: Dict[str, Type] = {}


def register_metric(name: str):
    """Decorator to register a custom metric callback.

    Example::

        @register_metric("my_metric")
        class MyMetric(MetricCallback):
            def on_epoch_end(self, model, epoch, **kw):
                val = ...
                self._log("my_value", val)
    """
    def decorator(cls: Type):
        _METRIC_REGISTRY[name] = cls
        return cls
    return decorator


def create_metric(name: str, **kwargs) -> "MetricCallback":
    """Instantiate a registered metric by name."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {list(_METRIC_REGISTRY.keys())}"
        )
    return _METRIC_REGISTRY[name](**kwargs)


def get_available_metrics() -> List[str]:
    return list(_METRIC_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------------

class MetricCallback(ABC):
    """Abstract base for all metric callbacks.

    Sub-classes must implement :meth:`on_epoch_end`.  Use :meth:`_log` to
    record scalar values that will be returned by :meth:`get_history`.

    Lifecycle hooks (override as needed):

    * ``on_epoch_end``  – called at the end of every epoch.
    * ``on_train_start`` – called once before training begins.
    * ``on_train_end``   – called once after training finishes.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def _log(self, key: str, value: float):
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(value)

    @abstractmethod
    def on_epoch_end(
        self,
        model: nn.Module,
        epoch: int,
        train_loss: float = 0.0,
        train_acc: float = 0.0,
        val_loss: float = 0.0,
        val_acc: float = 0.0,
        **kwargs,
    ):
        """Called at the end of every training epoch."""

    # Optional hooks – override in sub-classes if needed.
    def on_train_start(self, model: nn.Module, **kwargs):
        pass

    def on_train_end(self, model: nn.Module, **kwargs):
        pass

    def get_history(self) -> Dict[str, List[float]]:
        return self.history

    def reset(self):
        self.history.clear()


# ---------------------------------------------------------------------------
# Built-in Metrics
# ---------------------------------------------------------------------------

@register_metric("weight_norm")
class WeightNormCallback(MetricCallback):
    """Total L2 norm of all model parameters (scalar per epoch)."""

    def on_epoch_end(self, model: nn.Module, epoch: int, **kwargs):
        total = sum(p.data.norm(2).item() ** 2 for p in model.parameters())
        self._log("weight_norm", total ** 0.5)


@register_metric("gradient_norm")
class GradientNormCallback(MetricCallback):
    """Total L2 norm of gradients (scalar per epoch)."""

    def on_epoch_end(self, model: nn.Module, epoch: int, **kwargs):
        total = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        self._log("gradient_norm", total ** 0.5)


@register_metric("per_layer_norm")
class PerLayerNormCallback(MetricCallback):
    """L2 norm of each named parameter, logged separately."""

    def on_epoch_end(self, model: nn.Module, epoch: int, **kwargs):
        for name, p in model.named_parameters():
            self._log(f"norm/{name}", p.data.norm(2).item())


@register_metric("activation_variance")
class ActivationVarianceCallback(MetricCallback):
    """Variance of activations at Linear / LayerNorm layers.

    Inspired by FDA-style analysis from Tikquuss/grokking_fda.
    Call :meth:`register_hooks` **once** before training.
    """

    def __init__(self):
        super().__init__()
        self._hooks: list = []
        self._activations: Dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._activations[name] = out.detach()
        return hook

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on Linear and LayerNorm layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def on_epoch_end(self, model: nn.Module, epoch: int, **kwargs):
        for name, act in self._activations.items():
            self._log(f"act_var/{name}", act.var().item())

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def on_train_end(self, model: nn.Module, **kwargs):
        self.remove_hooks()

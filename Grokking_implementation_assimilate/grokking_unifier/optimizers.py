"""
Optimizer modifications for grokking experiments.

Implements the Grokfast gradient-filtering family from:
  Lee et al., "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
  (ironjr/grokfast)

Two variants are provided:
  * **GrokfastEMA** – exponential moving average filter (recommended).
  * **GrokfastMA**  – simple sliding-window moving average filter.

Usage::

    grokfast = GrokfastEMA(model, alpha=0.98, lamb=2.0)

    for epoch in range(num_epochs):
        loss.backward()
        grokfast.apply()      # modifies .grad in-place
        optimizer.step()
"""

import torch
import torch.nn as nn
from typing import Dict
from collections import deque


class GrokfastEMA:
    """Grokfast with Exponential Moving Average gradient filtering.

    Maintains an EMA of each parameter's gradient and adds it (scaled by
    ``lamb``) back to the current gradient.  This amplifies the slow /
    low-frequency gradient signal that drives generalisation while the
    high-frequency memorisation signal is attenuated.

    Parameters
    ----------
    model : nn.Module
        The model whose gradients will be filtered.
    alpha : float
        EMA decay factor (0 < alpha < 1).  Higher → smoother.
    lamb : float
        Amplification factor for the slow-gradient component.
    """

    def __init__(
        self, model: nn.Module, alpha: float = 0.98, lamb: float = 2.0
    ):
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        self.ema_grads: Dict[str, torch.Tensor] = {}

    def apply(self):
        """Call after ``loss.backward()`` and before ``optimizer.step()``."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name not in self.ema_grads:
                self.ema_grads[name] = torch.zeros_like(param.grad)

            self.ema_grads[name].mul_(self.alpha).add_(
                param.grad, alpha=1.0 - self.alpha
            )
            param.grad.add_(self.ema_grads[name], alpha=self.lamb)

    def reset(self):
        self.ema_grads.clear()


class GrokfastMA:
    """Grokfast with a sliding-window Moving Average filter.

    Parameters
    ----------
    model : nn.Module
        The model whose gradients will be filtered.
    window_size : int
        Number of past gradients to average over.
    lamb : float
        Amplification factor for the slow-gradient component.
    """

    def __init__(
        self, model: nn.Module, window_size: int = 100, lamb: float = 5.0
    ):
        self.model = model
        self.window_size = window_size
        self.lamb = lamb
        self.grad_history: Dict[str, deque] = {}

    def apply(self):
        """Call after ``loss.backward()`` and before ``optimizer.step()``."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name not in self.grad_history:
                self.grad_history[name] = deque(maxlen=self.window_size)

            self.grad_history[name].append(param.grad.detach().clone())

            if len(self.grad_history[name]) > 0:
                avg_grad = torch.stack(list(self.grad_history[name])).mean(dim=0)
                param.grad.add_(avg_grad, alpha=self.lamb)

    def reset(self):
        self.grad_history.clear()

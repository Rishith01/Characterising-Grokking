"""The single architecture: a plain fully connected MLP, no normalization layers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "quadratic": lambda z: z * z,
    "identity": lambda z: z,
}


class MLP(nn.Module):
    """`depth` Linear layers (so depth - 1 hidden layers of size `width`).

    Init reproduces PyTorch's default nn.Linear init (U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    for weight and bias), drawn from an explicit generator instead of the global RNG, and
    then multiplies every parameter by `init_scale` (Omnigrok's alpha).
    """

    def __init__(self, d_in: int, d_out: int, depth: int, width: int, activation: str,
                 init_scale: float, generator: torch.Generator):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; have {list(ACTIVATIONS)}")
        dims = [d_in] + [width] * (depth - 1) + [d_out]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.activation = activation
        self._act = ACTIVATIONS[activation]
        with torch.no_grad():
            for layer in self.layers:
                bound = 1.0 / math.sqrt(layer.in_features)
                for p in (layer.weight, layer.bias):
                    p.copy_((torch.rand(p.shape, generator=generator) * 2 - 1) * bound * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        return self.layers[-1](x)

    def features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Post-activation output of every hidden layer."""
        out = []
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
            out.append(x)
        return out

    def param_list(self) -> list[tuple[str, torch.Tensor]]:
        """All parameters in a fixed order: (l0.weight, l0.bias, l1.weight, ...).
        Every per-parameter record (fingerprints, keyframes) uses this order."""
        out = []
        for i, layer in enumerate(self.layers):
            out.append((f"l{i}.weight", layer.weight))
            out.append((f"l{i}.bias", layer.bias))
        return out

    def layer_params(self) -> list[list[torch.Tensor]]:
        """Parameters grouped per Linear layer, index i = layer i. Every per-layer
        telemetry column `l{i}_*` is computed over this group."""
        return [[layer.weight, layer.bias] for layer in self.layers]

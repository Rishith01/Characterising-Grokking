"""
Model architectures for grokking experiments.

Includes a 1-layer Transformer (Neel Nanda style) and an MLP,
with a registry for easy extension.

Repos unified:
  - neelnanda-io/Grokking  (Transformer architecture, no biases)
  - 7vik/grokking           (Transformer for modular arithmetic)
  - nmallinar/rfm-grokking  (MLP baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Type, List


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class.

    Example::

        @register_model("my_model")
        class MyModel(nn.Module):
            def __init__(self, p=97, **kwargs): ...
            def forward(self, x): ...
    """
    def decorator(cls: Type[nn.Module]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def get_available_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Transformer Components
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head causal self-attention (no bias, following Neel Nanda)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.W_Q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # Causal mask – upper-triangular entries become -inf
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.W_O(out)


class TransformerBlock(nn.Module):
    """Transformer block without LayerNorm (canonical Neel Nanda / 7vik grokking setup).

    LayerNorm is omitted by default because scale-invariance combined with
    strong weight decay (1.0) causes weight shrinkage and catastrophic loss spikes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        act_fn: str = "relu",
        use_ln: bool = False,
    ):
        super().__init__()
        self.use_ln = use_ln
        if use_ln:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        act = nn.ReLU() if act_fn == "relu" else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=False),
            act,
            nn.Linear(d_mlp, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_ln:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Full Models
# ---------------------------------------------------------------------------

@register_model("transformer")
class GrokkingTransformer(nn.Module):
    """1-layer Transformer for modular arithmetic (Neel Nanda style).

    Input  : ``(batch, 2)`` containing ``[a, b]``.
    Output : ``(batch, p)`` logits over the p residue classes.

    Internally the forward pass constructs the token sequence
    ``[a, b, <op>]`` and reads the logits at the last position.
    """

    def __init__(
        self,
        p: int = 97,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        n_layers: int = 1,
        act_fn: str = "relu",
        use_ln: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.use_ln = use_ln
        # p number-tokens + 1 special <op> token
        self.token_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)  # positions 0, 1, 2

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, d_mlp, act_fn=act_fn, use_ln=use_ln
                )
                for _ in range(n_layers)
            ]
        )

        if use_ln:
            self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

        if init_weights:
            self._init_weights()

    def _init_weights(self):
        """Standard Neel Nanda initialization: Gaussian with std = 1 / sqrt(d_model)."""
        std = 1.0 / math.sqrt(self.d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Build sequence [a, b, <op>]
        op_token = torch.full((B, 1), self.p, dtype=torch.long, device=x.device)
        tokens = torch.cat([x, op_token], dim=1)  # (B, 3)

        pos = torch.arange(3, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(tokens) + self.pos_embed(pos)

        for block in self.blocks:
            h = block(h)

        if self.use_ln:
            h = self.ln_final(h)
        logits = self.unembed(h[:, -1, :])  # read at last position
        return logits


@register_model("mlp")
class GrokkingMLP(nn.Module):
    """Simple MLP for modular arithmetic.

    Input  : ``(batch, 2)`` containing ``[a, b]``.
    Output : ``(batch, p)`` logits.

    Internally one-hot encodes ``a`` and ``b``, concatenates them, and
    passes through fully-connected layers with GELU activations.
    """

    def __init__(
        self,
        p: int = 97,
        hidden_dim: int = 256,
        n_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.p = p

        layers: List[nn.Module] = []
        in_dim = 2 * p  # one-hot(a) ∥ one-hot(b)
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, p))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_oh = F.one_hot(x[:, 0], self.p).float()
        b_oh = F.one_hot(x[:, 1], self.p).float()
        return self.net(torch.cat([a_oh, b_oh], dim=1))

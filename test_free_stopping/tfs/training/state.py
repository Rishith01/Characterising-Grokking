"""TrainState: the complete training state, i.e. a keyframe's payload (DESIGN.md §5).

There is no RNG state to save: minibatch order is a pure function of the step
(seeding.batch_indices), and nothing else in the training step is random. Restoring
weights + optimizer state + step is therefore an exact restart.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ..models import MLP


@dataclass
class TrainState:
    step: int
    params: dict[str, torch.Tensor]  # name -> fp32 tensor, MLP.param_list() order
    optim: dict[str, torch.Tensor]  # "{param index}/{state key}" -> tensor

    @classmethod
    def capture(cls, step: int, model: MLP, opt: torch.optim.Optimizer) -> "TrainState":
        params = {name: p.detach().to("cpu", copy=True) for name, p in model.param_list()}
        optim = {}
        for idx, st in opt.state_dict()["state"].items():
            for key, val in st.items():
                if val is None:
                    continue
                t = val if torch.is_tensor(val) else torch.tensor(val)
                optim[f"{idx}/{key}"] = t.detach().to("cpu", copy=True)
        return cls(step, params, optim)

    def restore(self, model: MLP, opt: torch.optim.Optimizer) -> None:
        device = next(model.parameters()).device
        with torch.no_grad():
            for name, p in model.param_list():
                p.copy_(self.params[name].to(device))
        state: dict[int, dict] = {}
        for k, v in self.optim.items():
            idx, key = k.split("/", 1)
            state.setdefault(int(idx), {})[key] = v.clone()
        sd = opt.state_dict()
        opt.load_state_dict({"state": state, "param_groups": sd["param_groups"]})

    # ---- flat form for safetensors -----------------------------------------------
    def to_flat(self) -> dict[str, torch.Tensor]:
        prefix = f"k{self.step}"
        out = {f"{prefix}/p/{n}": t.contiguous() for n, t in self.params.items()}
        out.update({f"{prefix}/o/{k}": t.contiguous() for k, t in self.optim.items()})
        return out

    @classmethod
    def from_flat(cls, step: int, flat: dict[str, torch.Tensor]) -> "TrainState":
        prefix = f"k{step}/"
        params, optim = {}, {}
        for k, t in flat.items():
            if not k.startswith(prefix):
                continue
            kind, rest = k[len(prefix):].split("/", 1)
            (params if kind == "p" else optim)[rest] = t
        return cls(step, params, optim)

    def hashes(self) -> dict[str, str]:
        return {k: tensor_sha256(t) for k, t in self.to_flat().items()}


def tensor_sha256(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

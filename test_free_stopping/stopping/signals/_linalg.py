"""Shared math for the signals (METRICS.md). Each function is unit-tested on cases with a
known answer (tests/test_stopping.py)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
import torch
from torch.func import functional_call, grad, vmap


# ---- spectra ----------------------------------------------------------------------------

def effective_rank(eigs: np.ndarray) -> float:
    """Roy & Vetterli 2007: exp of the Shannon entropy of the normalized spectrum.
    1 for a rank-1 spectrum, d for d equal eigenvalues. NaN for an all-zero spectrum."""
    lam = np.clip(np.asarray(eigs, dtype=np.float64), 0.0, None)
    total = lam.sum()
    if not total > 0:
        return float("nan")
    p = lam[lam > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def powerlaw_fit(eigs: np.ndarray, min_tail: int = 10) -> tuple[float, float, float, int]:
    """Power-law tail fit p(x) ~ x^-alpha (Clauset et al. 2009, as in WeightWatcher):
    for every candidate x_min, the MLE alpha = 1 + n / sum(log(x / x_min)) over the
    n values >= x_min; keep the x_min whose fit has the smallest Kolmogorov-Smirnov
    distance. Returns (alpha, x_min, ks, n_tail); NaNs if fewer than min_tail values."""
    x = np.sort(np.asarray(eigs, dtype=np.float64))
    x = x[x > 0]
    nan = (float("nan"), float("nan"), float("nan"), 0)
    if len(x) < min_tail:
        return nan
    best = nan
    logx = np.log(x)
    for i in range(len(x) - min_tail + 1):
        tail = x[i:]
        n = len(tail)
        s = (logx[i:] - logx[i]).sum()
        if s <= 0:
            continue
        alpha = 1.0 + n / s
        fit_cdf = 1.0 - (tail / tail[0]) ** (1.0 - alpha)
        emp_hi = np.arange(1, n + 1) / n
        emp_lo = np.arange(0, n) / n
        ks = float(max(np.max(np.abs(emp_hi - fit_cdf)), np.max(np.abs(emp_lo - fit_cdf))))
        if not np.isfinite(best[2]) or ks < best[2]:
            best = (float(alpha), float(tail[0]), ks, n)
    return best


def correlation_traps(w: torch.Tensor, rng: np.random.Generator, margin: float = 0.1) -> int:
    """Number of eigenvalues of the element-wise randomized W that exceed the
    Marchenko-Pastur bulk edge by more than `margin` (relative). Randomizing destroys
    learned correlations, so outliers left afterwards come from a few very large
    elements: a 'correlation trap' (Martin & Mahoney; Prakash & Martin)."""
    a = w.detach().to(torch.float64).cpu().numpy()
    if a.shape[0] < a.shape[1]:
        a = a.T
    n, m = a.shape  # n >= m
    flat = a.reshape(-1)
    r = flat[rng.permutation(flat.size)].reshape(n, m)
    sigma2 = float(r.var())
    if sigma2 <= 0:
        return 0
    eig = np.linalg.svd(r, compute_uv=False) ** 2 / n
    edge = sigma2 * (1.0 + np.sqrt(m / n)) ** 2
    return int(np.sum(eig > edge * (1.0 + margin)))


# ---- gradients --------------------------------------------------------------------------

def layer_groups(model: torch.nn.Module) -> "OrderedDict[str, list[str]]":
    """Parameter names grouped per module (name up to the last dot): for the MLP,
    'layers.0' -> ['layers.0.weight', 'layers.0.bias'], ..., in layer order."""
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for name, _ in model.named_parameters():
        groups.setdefault(name.rsplit(".", 1)[0], []).append(name)
    return groups


def per_example_moments(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor,
                        loss_fn: Callable, chunk: int = 128) -> tuple[dict, dict, int]:
    """Sum and sum of squares, per parameter coordinate, of the per-example gradients
    g_i = grad of loss_fn(model(x_i), target_i), over all examples. float64. Only the
    two running sums are kept, never the full [B, P] matrix."""
    params = {k: v.detach() for k, v in model.named_parameters()}

    def loss_one(p, xi, ti):
        return loss_fn(functional_call(model, p, (xi.unsqueeze(0),)), ti.unsqueeze(0))

    per_ex = vmap(grad(loss_one), in_dims=(None, 0, 0))
    s1 = {k: torch.zeros(v.numel(), dtype=torch.float64, device=v.device) for k, v in params.items()}
    s2 = {k: torch.zeros_like(z) for k, z in s1.items()}
    for i in range(0, x.shape[0], chunk):
        g = per_ex(params, x[i:i + chunk], target[i:i + chunk])
        for k, v in g.items():
            f = v.reshape(v.shape[0], -1).to(torch.float64)
            s1[k] += f.sum(0)
            s2[k] += (f * f).sum(0)
    return s1, s2, int(x.shape[0])


def batch_grad(model: torch.nn.Module, x: torch.Tensor, target: torch.Tensor,
               loss_fn: Callable) -> dict[str, torch.Tensor]:
    """Gradient of the mean loss over (x, target), flattened per parameter. The model's
    .grad fields are not touched."""
    names, ps = zip(*model.named_parameters())
    with torch.enable_grad():
        loss = loss_fn(model(x), target)
        gs = torch.autograd.grad(loss, ps)
    return {n: g.detach().reshape(-1).to(torch.float64) for n, g in zip(names, gs)}


def evidence(s1: torch.Tensor, s2: torch.Tensor, n: int, var_floor: float = 0.0) -> tuple[float, int]:
    """Mahsereci et al. 2017 EB criterion over the coordinates of one group:
    EB = 1 - (B/P) sum_k g_k^2 / s_k^2, with g the mean and s^2 the unbiased variance of
    the per-example gradients. Coordinates with variance <= var_floor (dead units, unused
    inputs) are dropped and P counts the rest. Returns (EB, P)."""
    mean = s1 / n
    var = (s2 - n * mean * mean) / (n - 1)
    keep = var > var_floor
    p = int(keep.sum())
    if p == 0:
        return float("nan"), 0
    return float(1.0 - (n / p) * (mean[keep] ** 2 / var[keep]).sum()), p


def finite_or_nan(x: Optional[float]) -> float:
    return float(x) if x is not None and np.isfinite(x) else float("nan")

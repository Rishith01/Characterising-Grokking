"""Toy check of online AGOP-alignment estimators on synthetic trajectories (AGOP.md §5).

No network is trained: trace-normalized "AGOP" matrices G_t (d = 30) are generated along
known paths from a memorizing state G0 (full rank) to a generalizing state GT (rank 3),
so the true alignment s(t) = cos(G_t, GT) is known. Three estimators stop when their
estimate of s reaches 0.95:
  paper      Theorem 5.2(2) of BTP_Proof_AGOP_online_variant.pdf: Aitken on
             v(t) = <G_t, G0> - P0, then s_cal,delta (eq. 39)
  paper+gate the same, accepted only where the chord defect D_chord (Thm 4.1, on window
             means) is below 0.05
  path       AGOP.md §4: remaining-angle bound, s >= cos(speed / decay rate)
A stop is premature if the true s is still below 0.95 there.

Run: .venv/Scripts/python.exe results/phase4/agop_toy_sim.py   (numpy only, ~1 min)
"""
from __future__ import annotations

import numpy as np

D, RANK, LOG_EVERY = 30, 3, 50
T = np.arange(0, 20000, LOG_EVERY)
WEXT, LAG, FITWIN, THR = 10, 20, 10, 0.95


def cos(a, b):
    return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))


def ang(a, b):
    return float(np.arccos(np.clip(cos(a, b), -1.0, 1.0)))


class World:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.G0 = self.psd(D)
        self.GT = self.psd(RANK)
        self.R = self.psd(RANK)  # an intermediate structure, for the off-chord detour
        self.P0 = float((self.G0 ** 2).sum())

    def psd(self, rank):
        a = self.rng.normal(size=(D, rank))
        g = a @ a.T
        return g / np.trace(g)

    def path(self, alpha, detour=0.0, noise=0.0):
        out = []
        for t in T:
            a = alpha(t)
            g = (1 - a) * self.G0 + a * self.GT + detour * 4 * a * (1 - a) * (self.R - self.GT)
            if noise:  # symmetric noise with ||e||_F = noise * ||g||_F, projected back to PSD
                e = self.rng.normal(size=(D, D))
                e = (e + e.T) / 2
                g = g + e * noise * np.linalg.norm(g) / np.linalg.norm(e)
                lam, u = np.linalg.eigh(g)
                g = (u * np.clip(lam, 0, None)) @ u.T
            out.append(g / np.trace(g))
        return np.asarray(out)


def alpha_exp(t, kappa=3e-4):  # the paper's Assumption 5.1
    return 1 - np.exp(-kappa * t)


def alpha_sig(t, tc=9000, width=900):  # grokking-like: long plateau, then a sharp rise
    a0 = 1 / (1 + np.exp(tc / width))
    return (1 / (1 + np.exp(-(t - tc) / width)) - a0) / (1 - a0)


def window_means(z, i):
    return [z[i - (3 - k) * LAG - WEXT + 1: i - (3 - k) * LAG + 1].mean(0) for k in (1, 2, 3)]


def paper_s_cal(w, Gs):
    v = np.array([(g * w.G0).sum() - w.P0 for g in Gs])
    P = np.array([(g ** 2).sum() for g in Gs])
    out = np.full(len(Gs), np.nan)
    for i in range(WEXT + 2 * LAG, len(Gs)):
        z1, z2, z3 = window_means(v, i)
        den = z1 + z3 - 2 * z2
        if abs(den) < 1e-15:
            continue
        dhat = (z1 * z3 - z2 ** 2) / den
        ahat = v[i] / dhat if dhat else np.nan
        if not 0.05 < ahat <= 1.5:  # the paper's minimum-progress condition, alpha_hat >= alpha_min
            continue
        P0 = w.P0
        PT = P0 + 2 * dhat + (P[i] - P0 - 2 * v[i]) / ahat ** 2
        if PT > 0:
            out[i] = (P0 + dhat + ahat * (PT - P0 - dhat)) / np.sqrt(P[i] * PT)
    return out


def d_chord(w, Gs):
    out = np.full(len(Gs), np.nan)
    for i in range(WEXT + LAG, len(Gs)):
        a = Gs[i - WEXT + 1: i + 1].mean(0) - w.G0
        b = Gs[i - LAG - WEXT + 1: i - LAG + 1].mean(0) - w.G0
        out[i] = 1 - cos(a, b)
    return out


def path_bound(Gs):
    speed = np.full(len(Gs), np.nan)
    for i in range(WEXT + LAG, len(Gs)):
        speed[i] = ang(Gs[i - WEXT + 1: i + 1].mean(0), Gs[i - LAG - WEXT + 1: i - LAG + 1].mean(0)) / (LAG * LOG_EVERY)
    out = np.full(len(Gs), np.nan)
    for i in range(len(Gs)):
        s = speed[max(0, i - FITWIN * LAG): i + 1: LAG]
        s = s[np.isfinite(s) & (s > 0)]
        if len(s) < 3:
            continue
        kappa = -np.polyfit(np.arange(len(s)) * LAG * LOG_EVERY, np.log(s), 1)[0]
        remaining = speed[i] / kappa if kappa > 0 else np.inf
        out[i] = np.cos(min(remaining, np.pi / 2))
    return out


def s_at_stop(s_true, est):
    hit = np.flatnonzero(np.nan_to_num(est) >= THR)
    return s_true[hit[0]] if len(hit) else np.nan


CASES = [("exponential, 5% noise", alpha_exp, 0.0, 0.05),
         ("sigmoid, clean", alpha_sig, 0.0, 0.0),
         ("sigmoid, off-chord detour", alpha_sig, 0.6, 0.0),
         ("sigmoid, 5% noise", alpha_sig, 0.0, 0.05),
         ("sigmoid, 15% noise", alpha_sig, 0.0, 0.15)]

if __name__ == "__main__":
    w = World(0)
    lam = np.linalg.eigvalsh(w.G0)
    top = np.outer(*[np.linalg.eigh(w.G0)[1][:, -1]] * 2)
    print(f"sign of delta: G_T = top eigenvector of G0 -> {(w.G0 * top).sum() - w.P0:+.4f} "
          f"(lambda1 {lam[-1]:.4f} >= P0 {w.P0:.4f}); G_T random rank {RANK} -> {(w.G0 * w.GT).sum() - w.P0:+.4f}")
    print(f"\n{'case':28s} {'method':11s} premature  never  min s at stop")
    for name, alpha, detour, noise in CASES:
        res = {"paper": [], "paper+gate": [], "path": []}
        for seed in range(10):
            w = World(seed)
            Gs = w.path(alpha, detour, noise)
            s_true = np.array([cos(g, w.GT) for g in Gs])
            sc = paper_s_cal(w, Gs)
            res["paper"].append(s_at_stop(s_true, sc))
            res["paper+gate"].append(s_at_stop(s_true, np.where(np.nan_to_num(d_chord(w, Gs), nan=1) < 0.05, sc, np.nan)))
            res["path"].append(s_at_stop(s_true, path_bound(Gs)))
        for m, v in res.items():
            v = np.asarray(v)
            print(f"{name:28s} {m:11s} {np.sum(v < THR):4d}/10  {np.isnan(v).sum():4d}/10  "
                  f"{np.nanmin(v) if np.isfinite(v).any() else float('nan'):.3f}")

"""Exact samples of x ~ N(0, Sigma) conditioned on ||x||^2 = E (fixed L2 energy).

In the eigenbasis Sigma = V diag(lam) V^T, the conditional is sqrt(E) * Bingham on the sphere.
We sample it with the ACG-envelope rejection sampler of Kent, Ganeiber & Mardia (2018).
Validated against ABC.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def bingham_energy_sampler(lam: np.ndarray, E: float, rng: np.random.Generator):
    """Return draw(n) -> (z (k, q) on the sphere of radius sqrt(E), acceptance rate)."""
    lam = np.asarray(lam, dtype=np.float64)
    q = lam.size
    a = 0.5 * E / lam
    a = a - a.min()
    b = brentq(lambda b: np.sum(1.0 / (b + 2 * a)) - 1.0, 1e-12, q)
    om = 1 + 2 * a / b
    logM = -(q - b) / 2 + (q / 2) * np.log(q / b)

    def draw(n):
        x = rng.standard_normal((n, q)) / np.sqrt(om)
        z = x / np.linalg.norm(x, axis=1, keepdims=True)
        logr = -(z**2 @ a) + (q / 2) * np.log(z**2 @ om) - logM
        acc = np.log(rng.random(n)) < logr
        return np.sqrt(E) * z[acc], acc.mean()

    return draw


def sample_fixed_energy(evals: np.ndarray, evecs: np.ndarray, E: float, n: int, seed: int = 0):
    """n exact samples of N(0, V diag(evals) V^T) | ||x||^2 = E, in physical coordinates."""
    rng = np.random.default_rng(seed)
    keep = evals > 1e-14 * evals.max()
    draw = bingham_energy_sampler(evals[keep], E, rng)
    out, total, batch = [], 0, 8192
    while total < n:
        z, acc = draw(batch)
        out.append(z)
        total += len(z)
        batch = int(min(2**20, max(8192, 1.2 * (n - total) / max(acc, 1e-4))))
    z = np.concatenate(out)[:n]
    return z @ evecs[:, keep].T

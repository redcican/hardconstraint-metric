"""Exact output law of any sampler that is affine in its Gaussian noise (Prop. 2 machinery).

For a Gaussian oracle model and affine constraints, every sampler in tcflow.samplers maps
(x0, xi_1, ..., xi_K) -> x1 affinely: x1 = a + B z, z ~ N(0, I_D). So
    law(x1) = N(a, B B^T)
exactly, with no Monte Carlo error. We get a and B by running the sampler on the batch [0; I_D].
Pass 1 counts the noise dimension D; pass 2 runs on the structured batch.
"""
from __future__ import annotations

import torch

from ..core import DTYPE


class _Counter:
    def __init__(self):
        self.total = 0

    def __call__(self, b, N):
        self.total += N
        return torch.zeros(b, N, dtype=DTYPE, device=self.device)


class _Feeder:
    """Hands out column blocks of [0; I_D] without ever forming the (D+1) x D matrix."""

    def __init__(self, D, offset, device):
        self.D, self.off, self.device = D, offset, device

    def block(self, N):
        out = torch.zeros(self.D + 1, N, dtype=DTYPE, device=self.device)
        r = torch.arange(N, device=self.device)
        out[1 + self.off + r, r] = 1.0
        return out

    def __call__(self, b, N):
        out = self.block(N)
        self.off += N
        return out


def exact_law(sampler, model, con, N, steps, device, **kw):
    """Return (mean (N,), cov (N, N)) of the sampler's output distribution."""
    c = _Counter()
    c.device = device
    x0 = torch.zeros(1, N, dtype=DTYPE, device=device)
    sampler(model, con, x0, steps, c, **kw)
    D = N + c.total
    x0 = _Feeder(D, 0, device).block(N)
    feeder = _Feeder(D, N, device)
    X = sampler(model, con, x0, steps, feeder, **kw)
    a = X[0]
    B = X[1:] - a
    cov = B.T @ B
    return a, 0.5 * (cov + cov.T)

"""Conservative (gradient-field) FM velocity: v(x, t) = grad_x psi(x, t), psi = 1/2 <x, f_theta(x, t)>.

The true FM velocity is a gradient field (v = ((t-1)... ) = grad of a scalar for the OT path with a
Gaussian source), so its Jacobian, and the denoiser Jacobian J_D = I + (1-t) J_v, are symmetric. An
unconstrained FNO does not respect that (measured: 21-32% asymmetry on KS). Here symmetry holds by
construction. For linear f = M x, grad psi = (M + M^T) x / 2: any symmetric linear field is reachable.
torch.func.grad is used so jvp/vjp/vmap (TC-Flow's Gamma products) compose through the gradient.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.func import grad

from .fno import FNOVelocity


class PotentialVelocity(nn.Module):
    def __init__(self, dims, modes, **kw):
        super().__init__()
        self.f = FNOVelocity(dims, modes, **kw)
        self.dims = tuple(dims)

    def psi(self, x, t):
        return 0.5 * (x * self.f(x, t).reshape(x.shape)).flatten(1).sum(1)  # f returns (b, *dims); x may be flat

    def forward(self, x, t):
        with torch.enable_grad():
            return grad(lambda u: self.psi(u, t).sum())(x)

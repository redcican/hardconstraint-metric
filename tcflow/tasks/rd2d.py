"""2-D FitzHugh-Nagumo reaction-diffusion (the PDEBench 2D diffusion-reaction system) on a periodic grid.

    u_t = Du Lap u + u - u^3 - k - v,     v_t = Dv Lap v + u - v,
    Du = 1e-3, Dv = 5e-3, k = 5e-3, domain [-1, 1]^2 (periodic), 64 x 64 grid, t in [0, 5].
    u0 ~ the Matern random field of the random-field tasks (alpha = 2, tau = 3, unit pointwise variance), v0 = 0.
A sample is the activator u at 16 equispaced times (t = 0, 1/3, ..., 5), stored as (s, s, 16) = (x1, x2, t) and
flattened x1-major, then x2, then t (as tasks/ns2d.py). IMEX pseudo-spectral scheme: diffusion implicit, reaction
explicit, dt = 5e-3.
"""
import math

import numpy as np
import torch

S, FRAMES, T_END = 64, 16, 5.0
DU, DV, K = 1e-3, 5e-3, 5e-3
L = 2.0


def matern_lam(s=S, alpha=2.0, tau=3.0):
    k = torch.fft.fftfreq(s, d=1.0 / s, dtype=torch.float64)
    lam = (k[:, None] ** 2 + k[None, :] ** 2 + tau**2) ** (-alpha)
    return lam * (s * s / lam.sum())


def prior_cov(s=S):
    """Dense (s^2, s^2) covariance of the Matern field (row-major pixels)."""
    c = torch.fft.ifft2(matern_lam(s).to(torch.complex128)).real  # c[dx, dy]
    i = torch.arange(s)
    dx = (i[:, None] - i[None, :]) % s
    C = c[dx[:, None, :, None], dx[None, :, None, :]]  # (s, s, s, s): [x1, y1, x2, y2]
    return C.reshape(s * s, s * s)


def grf(n, generator, s=S):
    z = torch.randn(n, s, s, generator=generator, dtype=torch.float64)
    return torch.fft.ifft2(torch.fft.fft2(z) * matern_lam(s).sqrt()).real.float()


def matheron(u, idx, y, Sig):
    """u: (n, s*s) prior draws; idx: (m,) pixel indices; y: (m,). Exact draws of u | u[idx] = y."""
    Sa = Sig[:, idx]
    G = torch.linalg.solve(Sa[idx], Sa.T).T  # (s*s, m)
    return (u.double() + (y[None].double() - u[:, idx].double()) @ G.T).float()


@torch.no_grad()
def solve(u0, dt=5e-3):
    """u0: (b, s, s) on any device. Returns (b, s, s, FRAMES)."""
    b, s, _ = u0.shape
    dev = u0.device
    k = torch.fft.fftfreq(s, d=L / s, device=dev) * 2 * math.pi
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    du, dv = 1.0 / (1.0 + dt * DU * k2), 1.0 / (1.0 + dt * DV * k2)
    steps = round(T_END / dt)
    every = steps // (FRAMES - 1)
    u, v = u0.double(), torch.zeros_like(u0, dtype=torch.float64)
    du, dv = du.double(), dv.double()
    out = torch.empty(b, s, s, FRAMES, device=dev)
    out[..., 0] = u0
    c = 1
    for j in range(steps):
        fu = u - u**3 - K - v
        fv = u - v
        u = torch.fft.ifft2(torch.fft.fft2(u + dt * fu) * du).real
        v = torch.fft.ifft2(torch.fft.fft2(v + dt * fv) * dv).real
        if (j + 1) % every == 0 and c < FRAMES:
            out[..., c] = u.float()
            c += 1
    if not torch.isfinite(out).all():
        raise ValueError("NaN in reaction-diffusion solution")
    return out


def sensor_index(pix):
    """Flattened (x1, x2, t) indices of the t = 0 entries at the given pixels."""
    return np.asarray(pix) * FRAMES

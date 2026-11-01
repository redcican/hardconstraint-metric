"""2-D incompressible Navier-Stokes (vorticity form) on the periodic unit square: the space-time task of
ECI (Cheng et al., ICLR 2025, App. B.4) and PCFM (Utkarsh et al., NeurIPS 2025, App. B.2).

    d_t w + u . grad w = nu Lap w + f,   div u = 0,   w = curl u,
    nu = 1e-3, 64 x 64 grid, t in [0, 49], 50 snapshots (t = 0, 1, ..., 49),
    w0 ~ N(0, 7^{3/2} (-Lap + 49 I)^{-5/2})            (GRF, alpha = 2.5, tau = 7)
    f(x) = 0.1 sqrt(2) sin(2 pi (x1 + x2) + phi)        (phi in [0, pi/2])

Pseudo-spectral solver with 2/3 dealiasing; Crank-Nicolson for viscosity, explicit advection and forcing,
dt = 1e-3, as in Li et al. (FNO) and the ECI generator (reimplemented here from the published description).
A field is stored as (s, s, 50) = (x1, x2, t); flattened x1-major, then x2, then t.
"""
import math

import torch


def grf_sample(n, s=64, alpha=2.5, tau=7.0, generator=None, device="cpu"):
    """n samples of the periodic Gaussian random field N(0, tau^{2 alpha - 2} (-Lap + tau^2)^{-alpha})."""
    sigma = tau ** (0.5 * (2 * alpha - 2))
    k = torch.cat([torch.arange(0, s // 2), torch.arange(-s // 2, 0)]).to(device)
    kx, ky = k[:, None].expand(s, s), k[None, :].expand(s, s)
    sqrt_eig = s**2 * math.sqrt(2.0) * sigma * (4 * math.pi**2 * (kx**2 + ky**2) + tau**2) ** (-alpha / 2.0)
    sqrt_eig[0, 0] = 0.0
    z = torch.randn(n, s, s, 2, generator=generator, device=device)
    coeff = sqrt_eig * torch.complex(z[..., 0], z[..., 1])
    return torch.fft.ifftn(coeff, dim=(1, 2)).real


def forcing(phi, s=64, device="cpu"):
    """f(x) = 0.1 sqrt(2) sin(2 pi (x1 + x2) + phi) for a vector of phases phi -> (len(phi), s, s)."""
    x = torch.linspace(0, 1, s + 1, device=device)[:-1]
    X, Y = torch.meshgrid(x, x, indexing="ij")
    phi = torch.as_tensor(phi, dtype=torch.float32, device=device)
    return 0.1 * math.sqrt(2.0) * torch.sin(2 * math.pi * (X + Y)[None] + phi[:, None, None])


@torch.no_grad()
def solve(w0, f, visc=1e-3, T=49.0, dt=1e-3, frames=50):
    """w0, f: (b, s, s) on the same device. Returns (b, s, s, frames) at t = 0, T/(frames-1), ..., T."""
    b, s, _ = w0.shape
    steps = math.ceil(T / dt)
    every = steps // (frames - 1)
    k = torch.cat([torch.arange(0, s // 2), torch.arange(-s // 2, 0)]).to(w0.device)
    ky = k[None, :].expand(s, s)[:, : s // 2 + 1]
    kx = k[:, None].expand(s, s)[:, : s // 2 + 1]
    lap = 4 * math.pi**2 * (kx**2 + ky**2)
    lap[0, 0] = 1.0
    dealias = ((ky.abs() <= (2 / 3) * (s // 2)) & (kx.abs() <= (2 / 3) * (s // 2))).float()[None]
    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    out = torch.empty(b, s, s, frames, device=w0.device)
    out[..., 0] = w0
    c = 1
    ikx, iky = 2j * math.pi * kx, 2j * math.pi * ky
    num_l, den = 1.0 - 0.5 * dt * visc * lap, 1.0 + 0.5 * dt * visc * lap
    for j in range(steps):
        psi_h = w_h / lap
        q = torch.fft.irfft2(iky * psi_h, s=(s, s))       # u1 =  d psi / d x2
        v = torch.fft.irfft2(-ikx * psi_h, s=(s, s))      # u2 = -d psi / d x1
        w_x = torch.fft.irfft2(ikx * w_h, s=(s, s))
        w_y = torch.fft.irfft2(iky * w_h, s=(s, s))
        F_h = dealias * torch.fft.rfft2(q * w_x + v * w_y)
        w_h = (-dt * F_h + dt * f_h + num_l * w_h) / den
        if (j + 1) % every == 0 and c < frames:
            out[..., c] = torch.fft.irfft2(w_h, s=(s, s))
            c += 1
    if not torch.isfinite(out).all():
        raise ValueError("NaN in Navier-Stokes solution")
    return out

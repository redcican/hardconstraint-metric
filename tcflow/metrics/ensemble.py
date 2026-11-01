"""Sample-based distributional metrics between a generated ensemble G and a reference ensemble R
(both (n, N) on the same flattened grid). These see calibration errors that MMSE/SMSE miss."""
from __future__ import annotations

import torch


def marginal_w1(G, R):
    """Mean over pixels of the 1D Wasserstein-1 distance between marginals (equal n via quantiles)."""
    n = min(G.shape[0], R.shape[0])
    q = torch.linspace(0, 1, n, device=G.device, dtype=G.dtype)
    gs = torch.quantile(G, q, dim=0) if G.shape[0] != n else G.sort(0).values
    rs = torch.quantile(R, q, dim=0) if R.shape[0] != n else R.sort(0).values
    return (gs - rs).abs().mean().item()


def coverage(G, R, level=0.9):
    """Fraction of reference values inside the generated central `level` interval, averaged over pixels.
    Calibrated: = level. Over-dispersed: > level. Under-dispersed or biased: < level."""
    lo = torch.quantile(G, (1 - level) / 2, dim=0)
    hi = torch.quantile(G, (1 + level) / 2, dim=0)
    return ((R >= lo) & (R <= hi)).float().mean().item()


def energy_distance(G, R, n_max=1024):
    """Energy distance 2E|G-R| - E|G-G'| - E|R-R'| on full fields (L2 norm), on subsamples."""
    G, R = G[:n_max], R[:n_max]
    gr = torch.cdist(G, R).mean()
    gg = torch.cdist(G, G).sum() / (G.shape[0] * (G.shape[0] - 1))
    rr = torch.cdist(R, R).sum() / (R.shape[0] * (R.shape[0] - 1))
    return (2 * gr - gg - rr).item()


def per_frame(G, R, frames, npts):
    """Per-time-slice variance ratio and RMS mean error (in units of the reference sd)."""
    Gf, Rf = G.reshape(-1, frames, npts), R.reshape(-1, frames, npts)
    vr = Gf.var(0).sum(-1) / Rf.var(0).sum(-1)
    sd = Rf.var(0).mean(-1).sqrt()
    me = ((Gf.mean(0) - Rf.mean(0)) ** 2).mean(-1).sqrt() / sd
    return vr, me


def crps(G, R, chunk=64):
    """Mean over pixels and reference draws of CRPS(G_pixel, r) = E|G - r| - E|G - G'|/2.
    Its floor (G = the true law) is E|R - R'|/2, reported by crps_floor."""
    n = G.shape[0]
    Gs = G.sort(0).values
    w = (2 * torch.arange(1, n + 1, device=G.device, dtype=G.dtype) - n - 1) / n**2
    gg = (w[:, None] * Gs).sum(0) * 2  # E|G - G'| per pixel
    gr = torch.cat([(G[:, None, i:i + chunk] - R[None, :, i:i + chunk]).abs().mean((0, 1))
                    for i in range(0, G.shape[1], chunk)])
    return (gr - 0.5 * gg).mean().item()

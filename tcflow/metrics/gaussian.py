"""Exact metrics between Gaussian laws (Tier 0)."""
from __future__ import annotations

import torch


def _psd_sqrt(S):
    w, V = torch.linalg.eigh(0.5 * (S + S.T))
    return (V * w.clamp_min(0).sqrt()) @ V.T


def w2_gaussian(m1, S1, m2, S2):
    """2-Wasserstein distance between N(m1, S1) and N(m2, S2)."""
    r = _psd_sqrt(S2)
    cross = _psd_sqrt(r @ S1 @ r)
    val = ((m1 - m2) ** 2).sum() + torch.trace(S1 + S2 - 2 * cross)
    return val.clamp_min(0).sqrt().item()


def gaussian_report(m, S, m_true, S_true, con=None):
    """Mean bias (max over coordinates, in units of the true conditional std), W2,
    trace variance ratio, relative Frobenius covariance error and feasibility."""
    sd = torch.diagonal(S_true).clamp_min(0).sqrt()
    msk = sd > 1e-6
    out = {
        "bias_sigma": ((m - m_true).abs()[msk] / sd[msk]).max().item(),
        "W2": w2_gaussian(m, S, m_true, S_true),
        "var_ratio": (torch.trace(S) / torch.trace(S_true)).item(),
        "cov_err": (torch.linalg.norm(S - S_true) / torch.linalg.norm(S_true)).item(),
    }
    if con is not None:
        out["mean_residual"] = (con.y - con.A @ m).abs().max().item()
        out["cov_residual"] = (con.A @ S @ con.A.T).abs().max().item()
    return out

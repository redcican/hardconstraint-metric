"""Task registry: grid shape, FNO config and a training-data source for each trained task."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass
class TaskSpec:
    name: str
    dims: tuple
    modes: tuple
    pad: tuple
    width: int = 64
    n_layers: int = 4
    iters: int = 20000
    batch: int = 256
    lr: float = 3e-4
    notes: str = ""


TASKS = {
    "grf1d": TaskSpec("grf1d", (64,), (32,), (0,), iters=20000, notes="T0-a trained: Matérn alpha=2, tau=3, N=64 (same prior as the oracle)"),
    "grf2d": TaskSpec("grf2d", (64, 64), (24, 24), (0, 0), iters=20000, batch=128, notes="T0-a 2D: Matérn alpha=2 on 64^2"),
    "heat": TaskSpec("heat", (100, 100), (32, 32), (0, 8), batch=128, notes="ECI/PCFM heat: sin(x+phi) exp(-v t), v~U(1,5), phi~U(0,pi)"),
    "burgers": TaskSpec("burgers", (101, 101), (32, 32), (8, 8), batch=128, notes="PCFM Burgers, fresh seeds"),
    "ks": TaskSpec("ks", (64, 128), (24, 32), (8, 0), batch=128, iters=30000, notes="T1-KS space-time (frames, x)"),
}


# ----------------------------------------------------------------------------- data sources

def _grf_sampler(dims, alpha=2.0, tau=3.0):
    if len(dims) == 1:
        (N,) = dims
        k2 = torch.fft.fftfreq(N, d=1.0 / N) ** 2
    else:
        n = dims[0]
        k = torch.fft.fftfreq(n, d=1.0 / n)
        k2 = k[:, None] ** 2 + k[None, :] ** 2
    lam = (k2 + tau**2) ** (-alpha)
    lam = lam * (lam.numel() / lam.sum())
    sq = lam.sqrt()

    def sample(b, device):
        z = torch.randn(b, *dims, device=device)
        zf = torch.fft.fftn(z, dim=tuple(range(1, len(dims) + 1)))
        return torch.fft.ifftn(zf * sq.to(device), dim=tuple(range(1, len(dims) + 1))).real

    return sample


def _heat_sampler(nx=100, nt=100):
    xs = torch.linspace(0.0, 2 * math.pi, nx + 1)[:-1]
    ts = torch.linspace(0.0, 1.0, nt)

    def sample(b, device):
        v = torch.empty(b, 1, 1, device=device).uniform_(1.0, 5.0)
        phi = torch.empty(b, 1, 1, device=device).uniform_(0.0, math.pi)
        return torch.sin(xs.to(device)[None, :, None] + phi) * torch.exp(-ts.to(device)[None, None, :] * v)

    return sample


def _tensor_sampler(path):
    X = torch.from_numpy(np.load(path)).float()

    def sample(b, device):
        idx = torch.randint(0, X.shape[0], (b,))
        return X[idx].to(device, non_blocking=True)

    sample.size = X.shape[0]
    return sample


def data_sampler(name):
    if name == "grf1d":
        return _grf_sampler((64,))
    if name == "grf2d":
        return _grf_sampler((64, 64))
    if name == "heat":
        return _heat_sampler()
    if name == "burgers":
        return _tensor_sampler(DATA / "burgers_train.npy")
    if name == "ks":
        return _tensor_sampler(DATA / "ks_train.npy")
    raise KeyError(name)

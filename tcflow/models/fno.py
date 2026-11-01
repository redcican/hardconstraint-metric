"""Time-conditioned FNO velocity field for flow matching on 1D or 2D grids.

Follows the PCFM/ECI FFM encoder (FNO: 4 layers, hidden 64, lifting/projection 256, 32-dim time
embedding), with FiLM time conditioning per layer. Non-periodic dimensions are zero-padded
before the FFTs (Heat time axis, Burgers x/t axes, KS time axis).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim=32, max_freq=1000.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, 4 * dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_freq) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * freqs[None] * 1000.0
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=-1))


class SpectralConv(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes = list(modes)
        self.nd = len(modes)
        scale = 1.0 / (cin * cout)
        nw = 2 ** (self.nd - 1)  # 1D: one weight; 2D: positive and negative first-axis modes
        self.w = nn.ParameterList(
            [nn.Parameter(scale * torch.randn(cin, cout, *self.modes, dtype=torch.cfloat)) for _ in range(nw)])

    def forward(self, x):
        dims = tuple(range(-self.nd, 0))
        xf = torch.fft.rfftn(x.float(), dim=dims)
        out = torch.zeros(x.shape[0], self.w[0].shape[1], *xf.shape[2:], dtype=torch.cfloat, device=x.device)
        if self.nd == 1:
            m = min(self.modes[0], xf.shape[-1])
            out[..., :m] = torch.einsum("bix,iox->box", xf[..., :m], self.w[0][..., :m])
        else:
            m0, m1 = min(self.modes[0], xf.shape[-2] // 2), min(self.modes[1], xf.shape[-1])
            out[..., :m0, :m1] = torch.einsum("bixy,ioxy->boxy", xf[..., :m0, :m1], self.w[0][..., :m0, :m1])
            out[..., -m0:, :m1] = torch.einsum("bixy,ioxy->boxy", xf[..., -m0:, :m1], self.w[1][..., :m0, :m1])
        return torch.fft.irfftn(out, s=x.shape[-self.nd:], dim=dims).to(x.dtype)


class FNOVelocity(nn.Module):
    """v_theta(x_t, t): input (b, *dims) -> output (b, *dims)."""

    def __init__(self, dims, modes, width=64, n_layers=4, proj=256, emb=32, pad=None):
        super().__init__()
        self.dims = tuple(dims)
        self.nd = len(dims)
        self.pad = tuple(pad) if pad is not None else (0,) * self.nd
        self.temb = TimeEmbedding(emb)
        self.lift = nn.Conv1d(1 + self.nd, width, 1) if self.nd == 1 else nn.Conv2d(1 + self.nd, width, 1)
        Conv = nn.Conv1d if self.nd == 1 else nn.Conv2d
        self.spec = nn.ModuleList([SpectralConv(width, width, modes) for _ in range(n_layers)])
        self.skip = nn.ModuleList([Conv(width, width, 1) for _ in range(n_layers)])
        self.film = nn.ModuleList([nn.Linear(4 * emb, 2 * width) for _ in range(n_layers)])
        self.proj = nn.Sequential(Conv(width, proj, 1), nn.GELU(), Conv(proj, 1, 1))

    def _grid(self, b, device):
        axes = [torch.linspace(0, 1, n, device=device) for n in self.dims]
        g = torch.stack(torch.meshgrid(*axes, indexing="ij"), 0)
        return g.unsqueeze(0).expand(b, -1, *self.dims)

    def forward(self, x, t):
        b = x.shape[0]
        x = x.reshape(b, 1, *self.dims)
        if not torch.is_tensor(t) or t.ndim == 0:
            t = torch.full((b,), float(t), device=x.device)
        h = self.lift(torch.cat([x, self._grid(b, x.device).to(x.dtype)], 1))
        padspec = []
        for p in reversed(self.pad):
            padspec += [0, p]
        if any(self.pad):
            h = F.pad(h, padspec)
        e = self.temb(t.to(h.dtype))
        for i, (sc, sk, fm) in enumerate(zip(self.spec, self.skip, self.film)):
            g, bta = fm(e).chunk(2, -1)
            shape = (b, -1) + (1,) * self.nd
            z = sc(h) + sk(h)
            z = z * (1 + g.view(shape)) + bta.view(shape)
            h = F.gelu(z) if i < len(self.spec) - 1 else z
        if any(self.pad):
            sl = (slice(None), slice(None)) + tuple(slice(0, n) for n in self.dims)
            h = h[sl]
        return self.proj(h).reshape(b, *self.dims)

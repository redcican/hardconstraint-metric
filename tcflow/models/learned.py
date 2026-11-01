"""Wrap a trained FM velocity net in the sampler interface of tcflow.core.

Normalisation: the net works on z = (u - mean)/std. Samplers and constraints work in physical
units u, so we map with u = mean + std*z. The FM source is z0 ~ N(0, I).
In u-units: D_u = mean + std * D_z, Gamma_u = std^2 Gamma_z, J_u = J_z (dimensionless).

Tweedie covariance from the network (generalised Tweedie for x0 ~ N(0, I)):
    Gamma_t = ((1-t)^2 / t) * J_t,   J_t = dD/dx_t,
so Gamma_t A^T costs m JVPs (batched with torch.func.vmap). J_t^T A^T costs m VJPs.
The learned J is not exactly symmetric, so symmetrize='half' uses (J + J^T)/2 (one JVP + one VJP).
At t < t_min the 1/t factor is ill-conditioned; there we use the data covariance Sigma_hat (exact at t = 0).
"""
from __future__ import annotations

import torch
from torch.func import jvp, vjp, vmap

from ..core import DTYPE
from .fno import FNOVelocity


class LearnedFM:
    def __init__(self, net, mean, std, dims, sigma_hat=None, t_min=0.1, symmetrize="half",
                 dtype=torch.float32, chunk=64, psd_floor=0.05):
        self.net = net.eval()
        self.mean, self.std = float(mean), float(std)
        self.dims = tuple(dims)
        self.N = int(torch.tensor(dims).prod())
        self.sigma_hat = sigma_hat  # (N, N) in u-units, or None
        self.t_min = t_min
        self.symmetrize = symmetrize
        self.dtype = dtype
        self.chunk = chunk
        # field evaluations per batched net call (JVP/VJP); keeps peak memory ~constant across grids
        self.budget = max(1, int(2**19 // self.N))
        self.psd_floor = psd_floor
        self.n_fallback, self.n_checked = 0, 0
        self._U = self._lam = None
        if sigma_hat is not None:
            self.set_data_covariance(sigma_hat)
        for p in self.net.parameters():
            p.requires_grad_(False)

    # ----------------------------------------------------------------- z-space primitives
    def _Dz(self, z, t):
        v = self.net(z.to(self.dtype), t)
        return (z.to(self.dtype) + (1 - t) * v.reshape(z.shape[0], -1)).reshape(z.shape[0], -1)

    def _to_z(self, x):
        return (x - self.mean) / self.std

    # ----------------------------------------------------------------- interface (u-units, float64 out)
    @torch.no_grad()
    def denoise(self, x, t):
        z = self._to_z(x)
        out = []
        c = 2 * self.budget  # no autograd: cheaper per evaluation
        for i in range(0, z.shape[0], c):
            out.append(self._Dz(z[i : i + c], t))
        return (self.mean + self.std * torch.cat(out)).to(DTYPE)

    def velocity(self, x, t):
        return (self.denoise(x, t) - x) / (1 - t)

    def denoise_diff(self, x, t):
        """Differentiable denoiser (autograd through the net), u-units."""
        return (self.mean + self.std * self._Dz(self._to_z(x), t)).to(DTYPE)

    def _jac_cols(self, z, t, V, transpose=False):
        """J V (or J^T V), batched over samples: z (b, N), V (N, m) -> (b, N, m).
        The net acts on each sample independently, so one batched JVP with the same tangent for every
        sample gives J_i v for all i at once; vmap runs over the m tangents in chunks."""
        z = z.to(self.dtype)
        Vt = V.T.to(self.dtype)  # (m, N)
        f = lambda u: self._Dz(u, t)
        outs = []
        tc = min(self.chunk, Vt.shape[0])  # tangents per vmap call
        bs = max(1, self.budget // tc)  # samples per call
        for i in range(0, z.shape[0], bs):
            zi = z[i : i + bs]
            if not transpose:
                g = lambda v: jvp(f, (zi,), (v.expand_as(zi),))[1]
            else:
                _, pull = vjp(f, zi)
                g = lambda v: pull(v.expand_as(zi))[0]
            cols = [vmap(g)(Vt[j : j + tc]) for j in range(0, Vt.shape[0], tc)]
            outs.append(torch.cat(cols).permute(1, 2, 0))  # (b_i, N, m)
        return torch.cat(outs).to(DTYPE)

    def source_noise(self, b, N, gen=None):
        """Draws from the FM source in physical units: mean + std * N(0, I)."""
        dev = next(self.net.parameters()).device
        return self.mean + self.std * torch.randn(b, N, dtype=DTYPE, device=dev, generator=gen)

    def jac_AT(self, x, t, A, transpose=False):
        return self._jac_cols(self._to_z(x), t, A.T, transpose)

    def jac_T_AT(self, x, t, A):
        return self.jac_AT(x, t, A, transpose=True)

    def gamma_mv(self, x, t, V, chunk=None):
        """Per-sample Gamma_t v: x, V (b, N) -> (b, N). One JVP (+ one VJP if symmetrised), chunked over b."""
        if t < self.t_min:
            return self.gaussian_gamma_mv(V, t)
        f = lambda u: self._Dz(u, t)
        chunk = chunk or self.budget
        out = []
        for i in range(0, x.shape[0], chunk):
            z = self._to_z(x[i:i + chunk]).to(self.dtype)
            v = V[i:i + chunk].to(self.dtype)
            Jv = jvp(f, (z,), (v,))[1]
            if self.symmetrize == "half":
                _, pull = vjp(f, z)
                Jv = 0.5 * (Jv + pull(v)[0])
            out.append(((self.std**2) * ((1 - t) ** 2 / t) * Jv).to(DTYPE))
        return torch.cat(out)

    def jacT_mv(self, x, t, V, chunk=None):
        """Per-sample J_t^T v (one VJP), u-units (J is dimensionless), chunked over b."""
        chunk = chunk or self.budget
        out = []
        for i in range(0, x.shape[0], chunk):
            z = self._to_z(x[i:i + chunk]).to(self.dtype)
            _, pull = vjp(lambda u: self._Dz(u, t), z)
            out.append(pull(V[i:i + chunk].to(self.dtype))[0].to(DTYPE))
        return torch.cat(out)

    def gaussian_gamma_mv(self, V, t):
        g = (1 - t) ** 2 * self._lam / (t**2 * self._lam + (1 - t) ** 2)
        return ((V.to(DTYPE) @ self._U) * g) @ self._U.T

    # ----------------------------------------------------------------- Gaussian surrogate
    def set_data_covariance(self, cov_or_data):
        """Either a dense (N, N) covariance, or data (n, N) -> low-rank eigenpairs of Sigma_hat."""
        C = cov_or_data.to(DTYPE)
        if C.shape[0] == C.shape[1] == self.N:
            lam, U = torch.linalg.eigh(0.5 * (C + C.T))
        else:
            Xc = C - C.mean(0)
            _, sv, Vh = torch.linalg.svd(Xc, full_matrices=False)
            lam, U = sv**2 / (Xc.shape[0] - 1), Vh.T
        keep = lam > 1e-10 * lam.max()
        self._lam, self._U = lam[keep], U[:, keep]

    def gaussian_gamma_AT(self, x, t, A):
        """Gamma_t of N(mean_hat, Sigma_hat) under the FM path: (1-t)^2 Sigma (t^2 Sigma + (1-t)^2 I)^{-1}.
        Exact for Gaussian data, and correct to leading order in t for any data with covariance Sigma."""
        g = (1 - t) ** 2 * self._lam / (t**2 * self._lam + (1 - t) ** 2)
        GA = self._U @ (g[:, None] * (self._U.T @ A.T.to(DTYPE)))
        return GA.expand(x.shape[0], -1, -1)

    def gamma_AT(self, x, t, A):
        if t < self.t_min:
            return self.gaussian_gamma_AT(x, t, A)
        JA = self.jac_AT(x, t, A)
        if self.symmetrize == "half":
            JA = 0.5 * (JA + self.jac_AT(x, t, A, transpose=True))
        GA = (self.std**2) * ((1 - t) ** 2 / t) * JA
        # per-sample safeguard: A Gamma A^T must be clearly positive definite
        S = torch.einsum("mn,bnk->bmk", A.to(DTYPE), GA)
        ev = torch.linalg.eigvalsh(0.5 * (S + S.transpose(1, 2)))
        bad = ev[:, 0] < self.psd_floor * ev[:, -1].clamp_min(1e-30)
        self.n_checked += x.shape[0]
        if bad.any() and self._U is not None:
            self.n_fallback += int(bad.sum())
            GA = torch.where(bad[:, None, None], self.gaussian_gamma_AT(x[:1], t, A).expand_as(GA), GA)
        return GA


def data_cov_AT(X):
    """Sigma_hat A^T from data X (n, N) without forming the N x N covariance."""
    Xc = (X - X.mean(0)).to(DTYPE)

    def f(A):
        return Xc.T @ (Xc @ A.T.to(DTYPE)) / (Xc.shape[0] - 1)

    return f


def load_learned(ckpt_path, device, sigma_hat=None, **kw):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    s = ck["spec"]
    cls = FNOVelocity
    if ck.get("arch") == "potential":
        from .potential import PotentialVelocity as cls
    net = cls(s["dims"], s["modes"], width=s["width"], n_layers=s["n_layers"], pad=s["pad"]).to(device)
    net.load_state_dict(ck["ema"])
    model = LearnedFM(net, ck["mean"], ck["std"], s["dims"], sigma_hat=sigma_hat, **kw)
    if ck.get("arch") == "potential":  # JVP/VJP through grad psi is second order: ~3x memory per field
        model.budget = max(1, model.budget // 4)
    return model, ck

"""Shared interfaces: priors/models (denoiser + Tweedie covariance) and affine constraints.

Conventions (whole package)
- Flow-matching path x_t = t*x1 + (1-t)*x0, with x0 ~ N(0, I) and t running 0 -> 1.
- Fields are flattened: x has shape (b, N).
- A model exposes
    denoise(x, t)        -> D = E[x1 | x_t]                (b, N)
    gamma_AT(x, t, A)    -> Gamma_t A^T, Gamma_t = Cov[x1 | x_t]   (b, N, m)
    jac_T_AT(x, t, A)    -> J_t^T A^T, J_t = dD/dx_t        (b, N, m)  (PPR, PiGDM)
  For a learned model these come from JVP/VJPs; GaussianOracle gives them in closed form.
"""
from __future__ import annotations

import math
import torch

DTYPE = torch.float64


def default_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- priors

def matern_spectrum_1d(N: int, alpha: float, tau: float = 3.0) -> torch.Tensor:
    """Eigenvalues of a stationary periodic covariance, lam_k ∝ (k^2+tau^2)^(-alpha), mean 1."""
    k = torch.fft.fftfreq(N, d=1.0 / N, dtype=DTYPE)
    lam = (k**2 + tau**2) ** (-alpha)
    return lam * (N / lam.sum())


def matern_spectrum_2d(n: int, alpha: float, tau: float = 3.0) -> torch.Tensor:
    """2D periodic n×n grid; returns flattened (n*n,) eigenvalues ordered like fft2 of a row-major field."""
    k = torch.fft.fftfreq(n, d=1.0 / n, dtype=DTYPE)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    lam = (k2 + tau**2) ** (-alpha)
    lam = lam * (n * n / lam.sum())
    return lam.reshape(-1)


def circulant_from_spectrum_1d(lam: torch.Tensor) -> torch.Tensor:
    """Dense Sigma = F diag(lam) F^* (real symmetric) for a 1D periodic grid."""
    N = lam.numel()
    c = torch.fft.ifft(lam.to(torch.complex128)).real  # first column
    idx = (torch.arange(N)[:, None] - torch.arange(N)[None, :]) % N
    S = c[idx]
    return 0.5 * (S + S.T)


class GaussianOracle:
    """Exact FM quantities for x1 ~ N(mu, Sigma), x0 ~ N(0, I).

    D(x,t)   = mu + t Sigma C_t^{-1} (x - t mu),   C_t = (1-t)^2 I + t^2 Sigma
    J_t      = t Sigma C_t^{-1}            (symmetric)
    Gamma_t  = Sigma - t^2 Sigma C_t^{-1} Sigma = (1-t)^2 Sigma C_t^{-1}  (= ((1-t)^2/t) J_t for t>0)
    Works with a dense Sigma. Per-t matrices are cached.
    """

    def __init__(self, Sigma: torch.Tensor, mu: torch.Tensor | None = None):
        self.Sigma = Sigma.to(DTYPE)
        self.N = Sigma.shape[0]
        self.mu = torch.zeros(self.N, dtype=DTYPE, device=Sigma.device) if mu is None else mu.to(DTYPE)
        self.evals, self.evecs = torch.linalg.eigh(self.Sigma)
        self.evals = self.evals.clamp_min(0)
        self._cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

    def _mats(self, t: float):
        key = round(float(t), 12)
        if key not in self._cache:
            lam, V = self.evals, self.evecs
            c = (1 - t) ** 2 + t**2 * lam
            J = (V * (t * lam / c)) @ V.T
            G = (V * ((1 - t) ** 2 * lam / c)) @ V.T
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = (J, G)
        return self._cache[key]

    def denoise(self, x, t):
        J, _ = self._mats(t)
        return self.mu + (x - t * self.mu) @ J.T

    def velocity(self, x, t):
        return (self.denoise(x, t) - x) / (1 - t)

    def denoise_diff(self, x, t):
        return self.denoise(x, t)

    def gamma_matrix(self, t):
        return self._mats(t)[1]

    def jac_matrix(self, t):
        return self._mats(t)[0]

    def gamma_mv(self, x, t, V):
        return V @ self.gamma_matrix(t).T

    def gaussian_gamma_mv(self, V, t):
        return V @ self.gamma_matrix(t).T

    def jacT_mv(self, x, t, V):
        return V @ self.jac_matrix(t)

    def gamma_AT(self, x, t, A):
        G = self.gamma_matrix(t)
        return (G @ A.T).expand(x.shape[0], -1, -1)

    def jac_T_AT(self, x, t, A):
        J = self.jac_matrix(t)
        return (J.T @ A.T).expand(x.shape[0], -1, -1)

    def source_noise(self, b, N, gen=None):
        return torch.randn(b, N, dtype=DTYPE, device=self.Sigma.device, generator=gen)

    def sample_prior(self, n, gen=None):
        z = torch.randn(n, self.N, dtype=DTYPE, device=self.Sigma.device, generator=gen)
        return self.mu + (z * self.evals.sqrt()) @ self.evecs.T


# ----------------------------------------------------------------------------- constraints

class AffineConstraint:
    """A x = y with A (m, N) and y (m,) or (b, m)."""

    def __init__(self, A: torch.Tensor, y: torch.Tensor, eps: float = 0.0):
        self.A = A.to(DTYPE)
        self.y = y.to(DTYPE)
        self.eps = eps
        self._AAt_inv = torch.linalg.inv(self.A @ self.A.T)
        self._pinv = self.A.T @ self._AAt_inv  # (N, m)

    @property
    def m(self):
        return self.A.shape[0]

    def residual(self, x):
        return self.y - x @ self.A.T

    def project(self, x):
        """Euclidean (minimum-norm) projection onto {Ax = y}."""
        return x + self.residual(x) @ self._pinv.T

    def kalman(self, D, GAT, y=None):
        """D + Gamma A^T (A Gamma A^T + eps I)^{-1} (y - A D), GAT = Gamma A^T of shape (b, N, m)."""
        r = (self.y if y is None else y) - D @ self.A.T  # (b, m)
        S = torch.einsum("mn,bnk->bmk", self.A, GAT)
        if self.eps > 0:
            S = S + self.eps * torch.eye(self.m, dtype=DTYPE, device=D.device)
        w = torch.linalg.solve(S, r.unsqueeze(-1)).squeeze(-1)  # (b, m)
        return D + torch.einsum("bnk,bk->bn", GAT, w)

    def conditional(self, mu, Sigma):
        """Exact Gaussian conditional N(mu*, Sigma*) of N(mu, Sigma) given A x = y (y must be (m,))."""
        G = Sigma @ self.A.T @ torch.linalg.inv(self.A @ Sigma @ self.A.T)
        m_c = mu + G @ (self.y - self.A @ mu)
        S_c = Sigma - G @ self.A @ Sigma
        return m_c, 0.5 * (S_c + S_c.T)


def point_sensors(N: int, idx, values, device=None) -> AffineConstraint:
    A = torch.zeros(len(idx), N, dtype=DTYPE, device=device)
    A[torch.arange(len(idx)), torch.as_tensor(idx)] = 1.0
    return AffineConstraint(A, torch.as_tensor(values, dtype=DTYPE, device=device))


def mass_constraint(N: int, value: float, device=None) -> AffineConstraint:
    A = torch.full((1, N), 1.0 / N, dtype=DTYPE, device=device)
    return AffineConstraint(A, torch.tensor([value], dtype=DTYPE, device=device))


def time_grid(steps: int):
    return [i / steps for i in range(steps + 1)]


def _isclose(a, b):
    return math.isclose(a, b, abs_tol=1e-12)

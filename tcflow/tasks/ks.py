"""T1-KS: Kuramoto-Sivashinsky space-time task (replaces the NS trajectory tasks; plan v2 §5.2).

PDE (conservative form, Exponax KuramotoSivashinskyConservative):
    u_t + u u_x + u_xx + u_xxxx = 0,  periodic on [0, L),  L = 32*pi,  N = 128 points.
IC: u0 ~ N(0, Sigma0), a stationary Matérn-type GRF (alpha = 2, tau = 3) with trace N. Its k = 0
mode gives a random mean, which is conserved and induces a Galilean drift.
Trajectory: F = 64 frames saved every dt_save = 0.5 (T = 31.5), dt = 0.1 inside (ETDRK2).
Sample = u[0:F, 0:N], a 64×128 space-time field. Models see it flattened (row-major, time-major).

Exact conditional references (the push-forward lemma: conditioning acts only on the IC):
    (i)  IC sensors / (ii) full IC slice: Matheron-conditioned Gaussian IC, then solve.
    (iii) fixed IC energy: Bingham IC (tcflow.refs.bingham), then solve.
    (iv) mass: Gaussian conditioning on the IC mean, then solve (Sigma-invariant control).
JAX runs on CPU so it does not fight PyTorch for the GPU.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

L_DOMAIN = 32 * np.pi
N_POINTS = 128
N_FRAMES = 64
DT = 0.1
SUBSTEPS = 5  # dt_save = 0.5
ALPHA, TAU = 2.0, 3.0


def ic_covariance(N=N_POINTS, alpha=ALPHA, tau=TAU, scale=1.0):
    """Dense circulant Sigma0 (N, N) and its eigendecomposition (real)."""
    k = np.fft.fftfreq(N, d=1.0 / N)
    lam = (k**2 + tau**2) ** (-alpha)
    lam = lam * (N / lam.sum()) * scale
    c = np.real(np.fft.ifft(lam))
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    S = c[idx]
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return S, np.clip(w, 0, None), V


def make_solver(N=N_POINTS, L=L_DOMAIN, dt=DT, substeps=SUBSTEPS, frames=N_FRAMES):
    """Return solve(u0: (B, N) numpy) -> (B, frames, N) numpy, frame 0 = IC."""
    import jax
    import jax.numpy as jnp
    import exponax as ex

    stepper = ex.stepper.KuramotoSivashinskyConservative(1, L, N, dt)
    coarse = ex.repeat(stepper, substeps)
    roll = ex.rollout(coarse, frames - 1, include_init=True)

    @jax.jit
    def _solve(u0):  # (B, 1, N)
        return jax.vmap(roll)(u0)  # (B, frames, 1, N)

    def solve(u0, chunk=1024):
        out = []
        for i in range(0, len(u0), chunk):
            u = jnp.asarray(u0[i : i + chunk, None, :], dtype=jnp.float32)
            out.append(np.asarray(_solve(u))[:, :, 0, :])
        return np.concatenate(out).astype(np.float32)

    return solve


def sample_ic(n, seed=0, S=None):
    rng = np.random.default_rng(seed)
    if S is None:
        S, w, V = ic_covariance()
    else:
        w, V = np.linalg.eigh(S)
    z = rng.standard_normal((n, len(w)))
    return (z * np.sqrt(np.clip(w, 0, None))) @ V.T


def gaussian_conditional_ic(A, y, n, seed=0):
    """Exact samples of u0 ~ N(0, Sigma0) | A u0 = y via Matheron's rule."""
    S, _, _ = ic_covariance()
    u = sample_ic(n, seed, S)
    G = S @ A.T @ np.linalg.inv(A @ S @ A.T)
    return u + (y[None, :] - u @ A.T) @ G.T


def energy_conditional_ic(E, n, seed=0):
    from tcflow.refs.bingham import sample_fixed_energy

    _, w, V = ic_covariance()
    return sample_fixed_energy(w, V, E, n, seed)


def generate_dataset(n, seed=0):
    """n unconditional training trajectories (n, F, N), float32."""
    solve = make_solver()
    return solve(sample_ic(n, seed))

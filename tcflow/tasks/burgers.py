"""Burgers 1D (PCFM App. B, generator from cpfpengfei/PCFM datasets/generate_burgers1d_data.py, MIT).

u_t + (u^2/2)_x = 0 on [0,1], first-order Godunov, Nx = Nt = 100 (stored 101×101 as (x, t)).
IC: smoothed step at p_loc ~ U(0.2, 0.8); left Dirichlet BC u_bc ~ U(0, 1).
Leakage fix vs PCFM: training and conditional-reference sets use disjoint seeds and parameters.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np


def solve_burgers(p_loc, u_bc, L=1.0, T=1.0, Nx=100, Nt=100, eps=0.02):
    dx, dt = L / Nx, T / Nt
    x = np.linspace(0, L, Nx + 1)
    u = np.zeros((Nt + 1, Nx + 1))
    u[0] = 1.0 / (1 + np.exp((x - p_loc) / eps))

    def godunov_flux(uL, uR):
        flux = np.zeros_like(uL)
        rare = uL <= uR
        shock = uL > uR
        flux[rare] = np.minimum(0.5 * uL[rare] ** 2, 0.5 * uR[rare] ** 2)
        flux[shock] = np.where((uL[shock] + uR[shock]) / 2 > 0, 0.5 * uL[shock] ** 2, 0.5 * uR[shock] ** 2)
        return flux

    for n in range(Nt):
        if n > 0:
            u[n, 0] = u_bc
        flux = godunov_flux(u[n, :-1], u[n, 1:])
        u[n + 1, 1:-1] = u[n, 1:-1] - (dt / dx) * (flux[1:] - flux[:-1])
        u[n + 1, 0] = u_bc
        u[n + 1, -1] = u[n + 1, -2]
    return u.T.astype(np.float32)  # (x, t)


def _one(args):
    return solve_burgers(*args)


def generate(p_locs, u_bcs, nproc=12):
    """All (p_loc, u_bc) pairs given as aligned arrays -> (n, 101, 101) float32."""
    with mp.get_context("fork").Pool(nproc) as pool:
        return np.stack(pool.map(_one, list(zip(p_locs, u_bcs)), chunksize=64))


def training_set(n=6400, seed=7):
    """Random (p_loc, u_bc) pairs; PCFM used an 80×80 product grid with seed 42 (whose ICs leak into its eval)."""
    rng = np.random.default_rng(seed)
    p, b = rng.uniform(0.2, 0.8, n), rng.uniform(0.0, 1.0, n)
    return generate(p, b), p, b

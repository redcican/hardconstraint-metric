"""Darcy flow with log-normal permeability on the unit square.

    -div( exp(g) grad p ) = 1  in (0, 1)^2,   p = 0 on the boundary,
    g ~ the Matern random field of the random-field tasks (alpha = 2, tau = 3, unit pointwise variance), 64 x 64.
Cell-centred 5-point finite volumes with harmonic-mean face permeabilities; Dirichlet faces at distance h/2.
A sample is the pair (g, p / P_SCALE) stored as (2, 64, 64) and flattened channel-major (g first), d = 8192.
P_SCALE is the pointwise standard deviation of p over the training set (fixed once).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

S = 64


def solve_one(g, s=S):
    a = np.exp(np.asarray(g, dtype=np.float64))
    h = 1.0 / s
    idx = np.arange(s * s).reshape(s, s)
    rows, cols, vals = [], [], []
    diag = np.zeros(s * s)
    # interior faces along axis 0 and axis 1 (harmonic means)
    for ax in (0, 1):
        a1 = a[:-1, :] if ax == 0 else a[:, :-1]
        a2 = a[1:, :] if ax == 0 else a[:, 1:]
        i1 = idx[:-1, :] if ax == 0 else idx[:, :-1]
        i2 = idx[1:, :] if ax == 0 else idx[:, 1:]
        t = (2 * a1 * a2 / (a1 + a2)).ravel() / h**2
        i1, i2 = i1.ravel(), i2.ravel()
        rows += [i1, i2]
        cols += [i2, i1]
        vals += [-t, -t]
        np.add.at(diag, i1, t)
        np.add.at(diag, i2, t)
    # Dirichlet boundary faces: transmissibility 2 a / h^2
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        np.add.at(diag, idx[sl].ravel(), 2 * a[sl].ravel() / h**2)
    rows.append(np.arange(s * s))
    cols.append(np.arange(s * s))
    vals.append(diag)
    Amat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(s * s, s * s))
    return spl.spsolve(Amat.tocsc(), np.ones(s * s)).reshape(s, s)


def solve_many(G, workers=16):
    """G: (n, s, s) array -> (n, s, s) pressures (float32)."""
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        P = list(ex.map(solve_one, list(G), chunksize=32))
    return np.stack(P).astype(np.float32)


def half_index(s=S):
    """Flattened indices of the left half (first 32 columns of axis 0) of the g channel."""
    return np.arange(s * s).reshape(s, s)[: s // 2, :].reshape(-1)

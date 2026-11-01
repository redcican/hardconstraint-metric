from __future__ import annotations

import math
from functools import partial

import torch

from ..core import DTYPE

REGISTRY: dict[str, callable] = {}


def register(name, **defaults):
    def deco(fn):
        REGISTRY[name] = partial(fn, **defaults) if defaults else fn
        REGISTRY[name].__doc__ = fn.__doc__
        return fn
    return deco


def get_sampler(spec):
    """'name' or 'name@k=v,k=v' (hyperparameters for tuning grids), e.g. 'pigdm@scale=0.5'.
    'eci_M1_R5' is the legacy spelling of 'eci@M=1,R=5'."""
    if spec.startswith("eci_") and "@" not in spec:
        _, m, r = spec.split("_")
        spec = f"eci@M={m[1:]},R={r[1:]}"
    name, _, kv = spec.partition("@")
    kw = {}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        try:
            kw[k] = None if v == "None" else (int(v) if v.lstrip("-").isdigit() else float(v))
        except ValueError:
            kw[k] = v  # string option, e.g. integrator=heun
    fn = REGISTRY[name]
    return partial(fn, **kw) if kw else fn


def _times(steps):
    return [i / steps for i in range(steps + 1)]


def _endpoint_from_x0hat(x, D, t):
    """Implied source noise from (x_t, D): x0_hat = (x_t - t D) / (1 - t)."""
    return (x - t * D) / (1 - t)


# ============================================================================ unconstrained / post-hoc

@register("unconstrained")
def sample_unconstrained(model, con, x0, steps, noise):
    """Deterministic Euler on the flow ODE; no constraint."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        x = x + (tn - t) * model.velocity(x, t)
    return x


@register("posthoc")
def sample_posthoc(model, con, x0, steps, noise):
    """Unconstrained Euler, then one Euclidean projection at t = 1."""
    return con.project(sample_unconstrained(model, con, x0, steps, noise))


# ============================================================================ ECI / PCFM / ProFlow

@register("eci")
def sample_eci(model, con, x0, steps, noise, M=1, R=None, final_projection=True):
    """ECI (Cheng et al., ICLR 2025, Algs. 2-3).
    Each step: u1 = x + (1-t) v ; u1 = C(u1) (Euclidean projection) ; x = (1-t') u0 + t' u1,
    with M mixing iterations (the first M-1 at the same t) and u0 redrawn every R steps
    (R=None: never). final_projection=True adds PCFM's last-step projection (ECI itself ends on
    a projected u1 interpolated at t'=1, which is already feasible for affine constraints)."""
    u0 = x0.clone()
    x = x0.clone()
    ts = _times(steps)
    for i, (t, tn) in enumerate(zip(ts[:-1], ts[1:])):
        if R is not None and i > 0 and i % R == 0:
            u0 = noise(*x.shape)
        for mix in range(M):
            t_cur = t
            t_next = t if mix < M - 1 else tn
            u1 = con.project(model.denoise(x, t_cur))
            x = (1 - t_next) * u0 + t_next * u1
    return con.project(x) if final_projection else x


@register("pcfm")
def sample_pcfm(model, con, x0, steps, noise, final_projection=True):
    """PCFM (Utkarsh et al., NeurIPS 2025, Alg. 1) for affine constraints with lambda = 0
    (no relaxed-penalty refinement): Gauss-Newton projection of the extrapolated endpoint is
    exact in one step for affine c, then re-interpolation with the ORIGINAL noise u0, then the
    final projection. This equals ECI(M=1, R=None) + final projection (checked in tests).
    The lambda > 0 refinement variant is in sample_pcfm_relaxed."""
    return sample_eci(model, con, x0, steps, noise, M=1, R=None, final_projection=final_projection)


@register("pcfm_relaxed")
def sample_pcfm_relaxed(model, con, x0, steps, noise, lam=1.0, lr=0.01, n_inner=20):
    """PCFM with its default guided_interpolation refinement (repo defaults lam=1, lr=0.01, 20 steps):
    after projecting u1, refine x_t' = argmin ||x - ((1-t')u0 + t'u1)||^2 + lam ||A D(x) - y||^2
    by gradient descent through the model (here via closed-form J for the oracle)."""
    u0 = x0.clone()
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        u1 = con.project(model.denoise(x, t))
        target = (1 - tn) * u0 + tn * u1
        z = target.clone()
        if tn < 1:
            for _ in range(n_inner):
                r = model.denoise(z, tn) @ con.A.T - con.y  # (b, m)
                JTAT = model.jac_T_AT(z, tn, con.A)  # (b, N, m)
                grad = 2 * (z - target) + 2 * lam * torch.einsum("bnm,bm->bn", JTAT, r)
                z = z - lr * grad
        x = z
    return con.project(x)


@register("proflow")
def sample_proflow(model, con, x0, steps, noise):
    """ProFlow (Yu et al., arXiv 2601.20227, Alg. 1), hard-constraint limit: proximal step
    argmin_{u in C} ||u - u1_hat||^2 is the Euclidean projection for affine C; then
    re-interpolation with FRESH noise every step. Equals ECI(M=1, R=1) + final projection.
    (The paper's practical version uses 3 GD steps on a penalty; that is softer, not harder.)"""
    return sample_eci(model, con, x0, steps, noise, M=1, R=1, final_projection=True)


# ============================================================================ DDNM / PiGDM / PPR

@register("ddnm")
def sample_ddnm(model, con, x0, steps, noise):
    """DDNM (Wang et al., ICLR 2023), eta = 0, transcribed to the FM path: null-space-corrected
    endpoint x1_hat = A^+ y + (I - A^+ A) D, then a DDIM step that keeps the network's predicted
    noise: x_t' = t' x1_hat + (1-t') x0_hat(x_t, D)."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        D = model.denoise(x, t)
        e = _endpoint_from_x0hat(x, D, t)
        x = tn * con.project(D) + (1 - tn) * e
    return con.project(x)


@register("pigdm")
def sample_pigdm(model, con, x0, steps, noise, sigma_y=0.0, scale=1.0):
    """PiGDM (Song et al., ICLR 2023) with sigma_y -> 0 on the FM path. Surrogate
    p(x1|x_t) ~ N(D, r_t^2 I), r_t^2 = (1-t)^2 / (t^2 + (1-t)^2). Guidance g = J^T A^T (r_t^2 A A^T + sigma_y^2 I)^{-1} (y - A D);
    the endpoint shift from the score relation E[x1|x_t] = (x_t + (1-t)^2 grad log p_t)/t is
    Delta D = ((1-t)^2 / t) g. Then Euler with the guided endpoint, and a final projection.
    At t = 0 (no information) we use the t = 1/steps value to avoid the 1/t singularity."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        te = max(t, ts[1] * 0.5)
        D = model.denoise(x, t)
        r2 = (1 - te) ** 2 / (te**2 + (1 - te) ** 2)
        S = r2 * (con.A @ con.A.T) + sigma_y**2 * torch.eye(con.m, dtype=DTYPE, device=x.device)
        w = torch.linalg.solve(S, (con.y - D @ con.A.T).T).T  # (b, m)
        D = D + scale * ((1 - te) ** 2 / te) * model.jacT_mv(x, te, w @ con.A)  # one VJP, cotangent A^T w; scale: tuning knob
        x = x + (tn - t) * (D - x) / (1 - t)
    return con.project(x)


@register("ppr_limit")
def sample_ppr_limit(model, con, x0, steps, noise):
    """PPR, idealised converged limit (used for the closed-form oracle analysis only; the practical
    algorithm is sample_ppr below). PPR (Rochman-Sharabi & Louppe, arXiv 2601.21033; repo ppr/projection.py), converged limit
    with lambda = 0: the optimiser moves the noisy sample x_t (from its current value) until the
    denoised endpoint satisfies the constraint, A D(x) = y. For an affine D the minimum-norm such
    move is x <- x + J^T A^T (A J J^T A^T)^{-1} (y - A D(x)) (what gradient descent / L-BFGS from x_t
    converges to). Then a deterministic DDIM (Euler) step. No Langevin corrections. Final projection."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        if t > 0:
            JTAT = model.jac_T_AT(x, t, con.A)  # (b, N, m)
            S = torch.einsum("bnm,bnk->bmk", JTAT, JTAT)
            r = con.y - model.denoise(x, t) @ con.A.T
            w = torch.linalg.solve(S, r.unsqueeze(-1)).squeeze(-1)
            x = x + torch.einsum("bnm,bm->bn", JTAT, w)
        x = x + (tn - t) * model.velocity(x, t)
    return con.project(x)


@register("ppr")
def sample_ppr(model, con, x0, steps, noise, proj_steps=4, num_renoising=2, num_corrections=4, lam=0.0,
               langevin_step=0.1, batch_chunk=None):
    """PPR as released (repo ppr/: samplers.create_ppr_sampler + projection.make_ppr_projection, KS
    config: proj_steps=4, num_renoising=2, num_corrections=4, lambda=0, DDIM eta=0), on the FM path:
      1. predict: deterministic DDIM/Euler step t -> s;
      2. project, num_renoising rounds: proj_steps L-BFGS steps on x_s minimising
         ||A D(x_s) - y||^2 + lam ||x_s - x_s^0||^2, then renoise x_s <- s D(x_s) + (1-s) eps (fresh);
      3. num_corrections Langevin steps on p_s with the denoiser score (s D - x)/(1-s)^2.
    Deviation: their Langevin step is g*sigma^2 with g=0.1 in VE units; we use g*(1-s)^2 on the FM path.
    Then the shared final projection. Not affine in the noise (line search), so Monte Carlo only."""
    batch_chunk = batch_chunk or min(128, 2 * getattr(model, "budget", 64))  # grid-scaled (memory); L-BFGS couples samples only via the line search
    if x0.shape[0] > batch_chunk:
        return torch.cat([sample_ppr(model, con, x0[k:k + batch_chunk], steps, noise, proj_steps, num_renoising,
                                     num_corrections, lam, langevin_step, batch_chunk)
                          for k in range(0, x0.shape[0], batch_chunk)])
    x = x0.clone()
    ts = _times(steps)
    for t, s in zip(ts[:-1], ts[1:]):
        x = x + (s - t) * model.velocity(x, t)
        if s < 1:
            for _ in range(num_renoising):
                x_ref = x.detach().clone()
                z = x.detach().clone().requires_grad_(True)
                opt = torch.optim.LBFGS([z], lr=1.0, max_iter=proj_steps, line_search_fn="strong_wolfe")

                def closure():
                    opt.zero_grad()
                    r = model.denoise_diff(z, s) @ con.A.T - con.y
                    loss = (r**2).sum() + lam * ((z - x_ref) ** 2).sum()
                    loss.backward()
                    return loss

                opt.step(closure)
                with torch.no_grad():
                    x = s * model.denoise(z.detach(), s) + (1 - s) * noise(*x.shape)
            g = langevin_step * (1 - s) ** 2
            for _ in range(num_corrections):
                score = (s * model.denoise(x, s) - x) / (1 - s) ** 2
                x = x + g * score + (2 * g) ** 0.5 * noise(*x.shape)
    return con.project(x)


# ============================================================================ PBFM stochastic

@register("pbfm_stochastic")
def sample_pbfm_stochastic(model, con, x0, steps, noise, t_star=0.2):
    """PBFM sampler (Baldan et al., Alg. 2): for t < t*, jump to the endpoint and re-noise with fresh
    noise at t + dt; deterministic Euler afterwards. PBFM has no sampling-time constraint step,
    so we add the shared final projection."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        D = model.denoise(x, t)
        if t < t_star - 1e-12:
            x = (1 - tn) * noise(*x.shape) + tn * D
        else:
            x = x + (tn - t) * (D - x) / (1 - t)
    return con.project(x)


# ============================================================================ covariance-aware

@register("tcflow")
def sample_tcflow(model, con, x0, steps, noise, metric="tweedie", t_switch=None, gamma0=None):
    """TC-Flow (ours). Euler step with the conditional velocity (D_y - x)/(1-t), with
        D_y = D + Gamma_t A^T (A Gamma_t A^T)^{-1} (y - A D),   Gamma_t = Cov[x1 | x_t].
    metric = 'tweedie' : Gamma_t from the model (oracle: exact; learned: JVP).
             'euclidean': Gamma_t = I  (ablation: same velocity, wrong metric).
             'diag'     : TMPD-style diagonal Gamma_t ~ diag(Gamma_t 1) (row sums).
    t_switch: for t > t_switch use the Euclidean metric (cost saver; Thm 3 late-time limit).
    gamma0:   Gamma at t = 0 (Cov[x1] = prior covariance); default: model value at t = 0.
    Final step: exact Euclidean projection (machine-precision feasibility)."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        D = model.denoise(x, t)
        use_euc = metric == "euclidean" or (t_switch is not None and t > t_switch)
        if use_euc:
            Dy = con.project(D)
        else:
            GAT = model.gamma_AT(x, t, con.A)
            if metric == "diag":
                ones = torch.ones(1, x.shape[1], dtype=DTYPE, device=x.device)
                gdiag = model.gamma_AT(x, t, ones).squeeze(-1)  # (b, N): Gamma 1
                GAT = gdiag.unsqueeze(-1) * con.A.T.unsqueeze(0)
            Dy = con.kalman(D, GAT)
        x = x + (tn - t) * (Dy - x) / (1 - t)
    return con.project(x)


def _kalman_cg(model, con, x, t, D, iters=20, tol=1e-4):
    """Matrix-free conditional endpoint D + Gamma A^T w with (A Gamma A^T) w = r solved by
    preconditioned CG. Each iteration: one Gamma matvec (JVP [+VJP]). Preconditioner: the inverse of
    A Gamma_G A^T from the Gaussian surrogate (dense m x m, computed once per t)."""
    A = con.A
    r = con.y - D @ A.T  # (b, m)
    SG = A @ model.gaussian_gamma_mv(A, t).T  # (m, m): A Gamma_G A^T
    SG = 0.5 * (SG + SG.T)
    SG = SG + 1e-6 * SG.diagonal().mean() * torch.eye(A.shape[0], dtype=DTYPE, device=x.device)  # low-rank data
    Lc = torch.linalg.cholesky(SG)
    prec = lambda v: torch.cholesky_solve(v.T, Lc).T
    op = lambda w: model.gamma_mv(x, t, w @ A) @ A.T
    w0 = prec(r)  # Gaussian-surrogate solution: start point and per-sample fallback
    w = w0.clone()
    res = r - op(w)
    z = prec(res)
    p = z.clone()
    rz = (res * z).sum(-1, keepdim=True)
    r0 = r.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    bad = torch.zeros_like(rz, dtype=torch.bool)  # non-positive curvature: learned Gamma not PD here
    for _ in range(iters):
        Ap = op(p)
        pAp = (p * Ap).sum(-1, keepdim=True)
        bad |= pAp <= 1e-8 * (p * p).sum(-1, keepdim=True) * SG.diagonal().mean()
        alpha = torch.where(bad, torch.zeros_like(pAp), rz / pAp.clamp_min(1e-30))
        w = w + alpha * p
        res = res - alpha * Ap
        if torch.where(bad, 0.0, res.norm(dim=-1, keepdim=True) / r0).max() < tol:
            break
        z = prec(res)
        rz_new = (res * z).sum(-1, keepdim=True)
        p = z + (rz_new / rz) * p
        rz = rz_new
    out = D + model.gamma_mv(x, t, w @ A)
    if hasattr(model, "n_fallback") and t >= getattr(model, "t_min", 0.0):
        model.n_checked += bad.numel()
        model.n_fallback += int(bad.sum())
    if bad.any():
        b = bad[:, 0]
        out[b] = D[b] + model.gaussian_gamma_mv(w0[b] @ A, t)
    return out


@register("tcflow_cg")
def sample_tcflow_cg(model, con, x0, steps, noise, cg_iters=20, t_switch=None):
    """TC-Flow, matrix-free (plan 4.1, large m): the conditional endpoint via preconditioned CG."""
    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        D = model.denoise(x, t)
        Dy = con.project(D) if (t_switch is not None and t > t_switch) else _kalman_cg(model, con, x, t, D, cg_iters)
        x = x + (tn - t) * (Dy - x) / (1 - t)
    return con.project(x)


@register("tmpd")
def sample_tmpd(model, con, x0, steps, noise):
    """TMPD (Boys et al., TMLR 2024), sigma_y -> 0, full-covariance variant: the Tweedie-moment
    Gaussian likelihood gives exactly the Kalman-conditioned endpoint used by TC-Flow. In the
    Gaussian oracle setting the two coincide (expected tie); they differ for learned models in how
    Gamma is computed (TMPD: dense Jacobian) and for nonlinear constraints (no GN, no exactness)."""
    return sample_tcflow(model, con, x0, steps, noise, metric="tweedie")


@register("tmpd_diag")
def sample_tmpd_diag(model, con, x0, steps, noise):
    """TMPD-D: diagonal Tweedie covariance (row-sum approximation), as in the scalable TMPD variant."""
    return sample_tcflow(model, con, x0, steps, noise, metric="diag")


@register("tcflow_euclid")
def sample_tcflow_euclid(model, con, x0, steps, noise):
    """Ablation: TC-Flow's conditional-velocity update with the identity metric."""
    return sample_tcflow(model, con, x0, steps, noise, metric="euclidean")


@register("tcflow_gauss")
def sample_tcflow_gauss(model, con, x0, steps, noise):
    """Diagnostic variant (added 2026-10-28, post hoc): TC-Flow with the Gaussian-surrogate metric
    Gamma_G(t) from the data covariance at every t (no network Jacobians); the learned denoiser D is
    still used. With the surrogate the CG operator equals its preconditioner, so CG is exact at once."""
    if not hasattr(model, "t_min"):
        return sample_tcflow(model, con, x0, steps, noise)  # oracle: Gamma_G is the exact Gamma
    old = model.t_min
    model.t_min = 2.0
    try:
        return sample_tcflow_cg(model, con, x0, steps, noise, cg_iters=2)
    finally:
        model.t_min = old



def _laplace_gamma_rows(model, A, alpha):
    """Rows of A (I + alpha L)^{-1}: L = grid graph Laplacian (periodic on axes with FNO pad 0, Neumann
    otherwise), applied exactly in the separable eigenbasis. Time-independent; cached per (A, alpha)."""
    cache = model.__dict__.setdefault("_lap_cache", {})
    key = (id(A), float(alpha))
    if key in cache and cache[key][0] is A:  # holding A keeps its id from being reused
        return cache[key][1]
    dims = model.dims
    pads = getattr(model.net, "pad", (0,) * len(dims))
    if hasattr(model.net, "f"):  # PotentialVelocity wraps an FNO
        pads = model.net.f.pad
    dt = torch.float64
    Qs, mus = [], []
    for n, p in zip(dims, pads):
        L = 2 * torch.eye(n, dtype=dt) - torch.diag(torch.ones(n - 1, dtype=dt), 1) - torch.diag(torch.ones(n - 1, dtype=dt), -1)
        if p == 0:
            L[0, -1] = L[-1, 0] = -1
        else:
            L[0, 0] = L[-1, -1] = 1
        mu, Q = torch.linalg.eigh(L)
        Qs.append(Q.to(A.device)); mus.append(mu.to(A.device))
    mu = sum(m.reshape([-1 if i == k else 1 for i in range(len(dims))]) for k, m in enumerate(mus))
    V = A.to(dt).reshape(A.shape[0], *dims)
    for k, Q in enumerate(Qs):  # to eigenbasis
        V = torch.movedim(torch.tensordot(V, Q, dims=([k + 1], [0])), -1, k + 1)
    V = V / (1 + alpha * mu)
    for k, Q in enumerate(Qs):  # back
        V = torch.movedim(torch.tensordot(V, Q.T, dims=([k + 1], [0])), -1, k + 1)
    out = V.reshape(A.shape[0], -1).to(A.dtype)
    cache[key] = (A, out)
    return out


@register("tcflow_hybrid")
def sample_tcflow_hybrid(model, con, x0, steps, noise, t_learn=0.8, rcond=1e-3, cg_iters=10, integrator="euler", metric="gauss", alpha=10.0):
    """TC-Flow, practical metric (definition frozen 2026-10-28, before the dev sweep and the
    confirmation run). Conditional endpoint D_y = D + Gamma A^T w with (A Gamma A^T) w = y - A D:
      t <  t_learn : Gamma = Gamma_G(t), the FM-path Tweedie covariance of N(mean_hat, Sigma_hat)
                     (data covariance; exact for Gaussian data, leading order in t otherwise);
      t >= t_learn : Gamma = learned symmetrised Jacobian metric, solved by CG preconditioned with
                     the truncated pseudo-inverse of A Gamma_G A^T; samples meeting non-positive
                     curvature use the Gamma_G solution.
    All solves drop eigen-directions of A Gamma_G A^T below rcond * lambda_max (rank-deficient data
    covariance: those residual components are left to the final projection, not amplified).
    t_learn = 0.8 from the KS diagnostic on training trajectories (learned A Gamma A^T indefinite
    for t <= 0.6, usable for t >= 0.8). Final Euclidean projection: exact feasibility.
    integrator: 'euler' (frozen default) or 'heun' (2nd order; Euler on the last step; added 29 Oct)."""
    A = con.A
    lap_alpha = alpha  # the CG loop below assigns its own local `alpha`

    def cond_velocity(x, t):
        D = model.denoise(x, t)
        r = con.y - D @ A.T  # (b, m)
        if metric == "gauss":
            GA = model.gaussian_gamma_mv(A, t)  # (m, N): row i = Gamma a_i
        elif metric == "laplace":
            GA = _laplace_gamma_rows(model, A, lap_alpha)
        else:
            GA = A  # euclid: Gamma = I
        S = A @ GA.T
        lam, Q = torch.linalg.eigh(0.5 * (S + S.T))
        keep = lam > rcond * lam.max()
        Qk, lk = Q[:, keep], lam[keep]
        pinv = lambda v: ((v @ Qk) / lk) @ Qk.T
        w0 = pinv(r)
        Dy = D + w0 @ GA
        if t >= t_learn and hasattr(model, "n_fallback"):
            op = lambda w: model.gamma_mv(x, t, w @ A) @ A.T
            w = w0.clone()
            res = r - op(w)
            z = pinv(res)
            p = z.clone()
            rz = (res * z).sum(-1, keepdim=True)
            bad = torch.zeros_like(rz, dtype=torch.bool)
            scale = lk.mean()
            for _ in range(cg_iters):
                Ap = op(p)
                pAp = (p * Ap).sum(-1, keepdim=True)
                bad |= pAp <= 1e-8 * (p * p).sum(-1, keepdim=True) * scale
                alpha = torch.where(bad, torch.zeros_like(pAp), rz / pAp.clamp_min(1e-30))
                w = w + alpha * p
                res = res - alpha * Ap
                z = pinv(res)
                rz_new = (res * z).sum(-1, keepdim=True)
                p = z + (rz_new / rz.clamp_min(1e-30)) * p
                rz = rz_new
            model.n_checked += bad.numel()
            model.n_fallback += int(bad.sum())
            DyL = D + model.gamma_mv(x, t, w @ A)
            Dy = torch.where(bad, Dy, DyL)
        return (Dy - x) / (1 - t)

    x = x0.clone()
    ts = _times(steps)
    for t, tn in zip(ts[:-1], ts[1:]):
        v = cond_velocity(x, t)
        if integrator == "heun" and tn < 1:
            xe = x + (tn - t) * v
            x = x + 0.5 * (tn - t) * (v + cond_velocity(xe, tn))
        else:
            x = x + (tn - t) * v
    return con.project(x)



@register("tcflow_v2")
def sample_tcflow_v2(model, con, x0, steps, noise, rcond=1e-2):
    """TC-Flow v2 (amendment 2, 30 Oct 2026; plan 5.9): Gaussian-Tweedie metric Gamma_G(t) from the
    training-data covariance at every t (no network Jacobians), truncated pseudo-inverse (rcond),
    Heun integrator. NFE-matched: given a budget of `steps` network evaluations, it takes steps // 2
    Heun steps (2 evaluations each, Euler on the last), i.e. <= steps evaluations. Final projection."""
    return sample_tcflow_hybrid(model, con, x0, max(1, steps // 2), noise, t_learn=2.0, rcond=rcond, integrator="heun")


@register("tcflow_abl")
def sample_tcflow_abl(model, con, x0, steps, noise, metric="gauss", integrator="heun", rcond=1e-2):
    """Ablation (31 Oct 2026, plan 5.11): 2x2 of metric {gauss = Gamma_G(t), euclid = I} x integrator
    {euler, heun}, NFE-matched (heun takes steps // 2 steps). gauss + heun is tcflow_v2."""
    k = max(1, steps // 2) if integrator == "heun" else steps
    return sample_tcflow_hybrid(model, con, x0, k, noise, t_learn=2.0, rcond=rcond, integrator=integrator, metric=metric)


@register("projflow_lap")
def sample_projflow_lap(model, con, x0, steps, noise, alpha=10.0):
    """Added baseline (31 Oct 2026, plan 5.12): ProjFlow-style (Watanabe et al., 2602.22742) hand-made,
    time-independent metric Gamma = (I + alpha L)^{-1} (grid graph Laplacian). Everything else identical
    to tcflow_v2 (same update, Heun, NFE-matched, final projection), so only the metric differs."""
    return sample_tcflow_hybrid(model, con, x0, max(1, steps // 2), noise, t_learn=2.0, rcond=1e-6,
                                integrator="heun", metric="laplace", alpha=alpha)


@register("tcflow_v2L")
def sample_tcflow_v2L(model, con, x0, steps, noise, rcond=1e-2, t_learn=0.8):
    """Extra study (31 Oct 2026, plan 5.14): v2 (Heun, steps // 2) plus the learned symmetrised-Jacobian
    metric for t >= t_learn (CG, per-sample fallback to Gamma_G). Not NFE-matched: CG adds JVP/VJPs."""
    return sample_tcflow_hybrid(model, con, x0, max(1, steps // 2), noise, t_learn=t_learn, rcond=rcond, integrator="heun")

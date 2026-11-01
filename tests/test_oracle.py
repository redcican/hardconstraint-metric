"""Tier-0 checks: the oracle model, exact laws and the theory statements they must reproduce."""
import pytest
import torch

from tcflow.core import (DTYPE, GaussianOracle, circulant_from_spectrum_1d, mass_constraint,
                         matern_spectrum_1d, point_sensors)
from tcflow.metrics.gaussian import gaussian_report
from tcflow.samplers import REGISTRY
from tcflow.theory.exact_law import exact_law

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N = 64


def setup(alpha=2.0, kind="points"):
    S = circulant_from_spectrum_1d(matern_spectrum_1d(N, alpha)).to(DEV)
    model = GaussianOracle(S)
    if kind == "points":
        con = point_sensors(N, [0, 16, 40], [1.5, -1.0, 0.8], device=DEV)
    else:
        con = mass_constraint(N, 0.3, device=DEV)
    m_c, S_c = con.conditional(model.mu, model.Sigma)
    return model, con, m_c, S_c


def test_tweedie_identity():
    model, *_ = setup()
    S = model.Sigma
    for t in [0.1, 0.5, 0.9]:
        C = (1 - t) ** 2 * torch.eye(N, dtype=DTYPE, device=DEV) + t**2 * S
        G_joint = S - t**2 * S @ torch.linalg.solve(C, S)  # Cov[x1 | x_t] from the joint Gaussian
        assert torch.allclose(model.gamma_matrix(t), G_joint, atol=1e-10)
        assert torch.allclose(model.gamma_matrix(t), ((1 - t) ** 2 / t) * model.jac_matrix(t), atol=1e-10)


def test_lemma1_posthoc_matches_closed_form():
    model, con, m_c, S_c = setup()
    m, Sg = exact_law(REGISTRY["posthoc"], model, con, N, 2000, DEV)
    A, S, y = con.A, model.Sigma, con.y
    P = torch.eye(N, dtype=DTYPE, device=DEV) - A.T @ torch.linalg.solve(A @ A.T, A)
    L = P @ S @ A.T @ torch.linalg.inv(A @ S @ A.T)
    r = y - A @ model.mu
    assert torch.allclose(m - m_c, -L @ r, atol=2e-2)
    assert torch.allclose(Sg - S_c, L @ (A @ S @ A.T) @ L.T, atol=5e-2)


@pytest.mark.parametrize("alpha", [0.0, 2.0, 3.0])
def test_tcflow_exact_and_feasible(alpha):
    model, con, m_c, S_c = setup(alpha)
    errs = []
    for steps in [50, 200]:
        m, S = exact_law(REGISTRY["tcflow"], model, con, N, steps, DEV)
        rep = gaussian_report(m, S, m_c, S_c, con)
        assert rep["mean_residual"] < 1e-9 and rep["cov_residual"] < 1e-9
        errs.append(rep["W2"])
    assert errs[1] < 0.5 * errs[0] + 1e-6  # O(dt) convergence
    assert errs[1] < 0.1


def test_mass_constraint_is_exact_for_projection():
    model, con, m_c, S_c = setup(2.0, "mass")
    m, S = exact_law(REGISTRY["posthoc"], model, con, N, 500, DEV)
    assert gaussian_report(m, S, m_c, S_c)["W2"] < 0.1


def test_pcfm_matches_pilot_blowup():
    model, con, m_c, S_c = setup(2.0)
    m, S = exact_law(REGISTRY["pcfm"], model, con, N, 100, DEV)
    assert gaussian_report(m, S, m_c, S_c)["var_ratio"] > 10  # pilot v2: 23x


def test_exact_law_matches_monte_carlo():
    model, con, *_ = setup(2.0)
    sampler = REGISTRY["proflow"]  # uses fresh noise every step
    m, S = exact_law(sampler, model, con, N, 20, DEV)
    g = torch.Generator(device=DEV).manual_seed(0)
    noise = lambda b, n: torch.randn(b, n, dtype=DTYPE, device=DEV, generator=g)
    X = sampler(model, con, noise(40000, N), 20, noise)
    assert torch.allclose(X.mean(0), m, atol=0.05)
    assert (torch.trace(torch.cov(X.T)) / torch.trace(S) - 1).abs() < 0.03


def test_tcflow_cg_matches_dense():
    model, con, m_c, S_c = setup(2.0)
    m1, S1 = exact_law(REGISTRY["tcflow"], model, con, N, 50, DEV)
    m2, S2 = exact_law(REGISTRY["tcflow_cg"], model, con, N, 50, DEV)
    assert torch.allclose(m1, m2, atol=1e-6) and torch.allclose(S1, S2, atol=1e-6)


def test_tcflow_hybrid_is_exact_on_oracle():
    """On the Gaussian oracle Gamma_G is the exact Gamma: the hybrid equals TC-Flow."""
    model, con, m_c, S_c = setup(2.0)
    hyb = lambda *a: REGISTRY["tcflow_hybrid"](*a, rcond=1e-12)
    m1, S1 = exact_law(REGISTRY["tcflow"], model, con, N, 50, DEV)
    m2, S2 = exact_law(hyb, model, con, N, 50, DEV)
    assert torch.allclose(m1, m2, atol=1e-6) and torch.allclose(S1, S2, atol=1e-6)


def test_get_sampler_spec():
    from tcflow.samplers.base import get_sampler
    f = get_sampler("pigdm@scale=0.5")
    assert f.keywords == {"scale": 0.5}
    assert get_sampler("eci_M1_R5").keywords == {"M": 1, "R": 5}


def test_tcflow_v2_oracle_close_to_exact():
    """v2 on the oracle (Gamma_G exact): Heun at 50 steps (NFE 99) beats Euler TC-Flow at 100 steps."""
    model, con, m_c, S_c = setup(2.0)
    v2 = lambda *a: REGISTRY["tcflow_v2"](*a, rcond=1e-12)
    m2, S2 = exact_law(v2, model, con, N, 100, DEV)
    m1, S1 = exact_law(REGISTRY["tcflow"], model, con, N, 100, DEV)
    assert gaussian_report(m2, S2, m_c, S_c)["W2"] < gaussian_report(m1, S1, m_c, S_c)["W2"]

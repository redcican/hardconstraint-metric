"""Hard-constraint flow samplers, all with one signature:

    sampler(model, con, x0, steps, noise, **kw) -> x1   (b, N)

- model: see tcflow.core (denoise, gamma_AT, jac_T_AT)
- con:   AffineConstraint (nonlinear constraints: tcflow.samplers.nonlinear, later)
- x0:    initial source noise (b, N); ECI/PCFM re-use it as u0
- noise: callable (b, N) -> fresh N(0, I) draws. Every random draw goes through it, so
         tcflow.theory.exact_law can recover each sampler's exact output law (all samplers
         are affine in the noise for a Gaussian oracle and affine constraints).

Baselines follow their papers' update rules (see research_notes/.../baseline_protocols.md).
Each docstring cites the algorithm it implements and states any deviation. The shared
rules: same model, same time grid, same final Euclidean projection for every hard-constraint
method (as in PCFM), and no method gets fewer steps or a different noise source.
"""
from .base import REGISTRY, get_sampler  # noqa: F401

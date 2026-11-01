# hardconstraint-metric

Code for **TC-Flow**, from "The Metric Matters: measure-correct hard constraints for flow-matching PDE samplers". TC-Flow samples physical fields from a flow-matching model under hard affine constraints. Its correction uses the time-dependent metric of the flow instead of the Euclidean one, so the samples follow the correct conditional distribution.

## Install

```
pip install -e .
```

Python ≥ 3.10 with PyTorch. The KS generator uses Exponax (JAX, CPU). `tcflow/parallel.py` spreads jobs over several GPUs.

## Layout

| Path | What |
|---|---|
| `tcflow/core.py` | Gaussian oracle model (denoiser, Tweedie covariance, Jacobian), affine constraints, Matérn spectra |
| `tcflow/samplers/base.py` | TC-Flow and baselines: post-hoc projection, PCFM (+relaxed), ECI, ProFlow, DDNM, ΠGDM, PPR, PBFM-stochastic, TMPD (full/diag), Euclidean ablation |
| `tcflow/theory/exact_law.py` | exact output law of any noise-affine sampler |
| `tcflow/metrics/` | exact Gaussian W2, bias/σ, variance ratio, covariance error; ensemble scores (CRPS, coverage) |
| `tcflow/models/` | FNO backbone, learned-model wrapper, potential parameterisation |
| `tcflow/tasks/` | Heat, Burgers, Kuramoto–Sivashinsky, reaction–diffusion, Darcy, Navier–Stokes generators and constraints |
| `tcflow/refs/bingham.py` | exact fixed-energy (Bingham) sampler |
| `tests/` | oracle theory checks |

## Usage

```
make test     # pytest (oracle theory checks)
```

## Baselines

Baselines implement their published update rules and get the same tuning budget as TC-Flow. Reference implementations are not vendored; clone them into `third_party/` if needed:
[ECI-sampling](https://github.com/amazon-science/ECI-sampling),
[PCFM](https://github.com/cpfpengfei/PCFM),
[HardFlow](https://github.com/azizanlab/HardFlow),
[PBFM](https://github.com/tum-pbs/PBFM),
[PPR](https://github.com/OmerRochman/ppr),
[DiEM](https://github.com/francois-rozet/diem),
[ProbConserv](https://github.com/amazon-science/probconserv).

# Probabilistic Machine Learning with JAX

A notebook series exploring probabilistic machine learning through mathematical
derivations, implementations, and visual experiments. Starting with multivariate
Gaussians, the series builds toward Bayesian regression, Gaussian processes,
state estimation, approximate Bayesian neural networks, and Monte Carlo sampling.

The shared implementation lives in [`src/gaussian.py`](src/gaussian.py); the
notebooks connect the code to the underlying mathematics.

## Reading order

| #   | Notebook                                                            | Topics                                                                                       |
| --- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 1   | [Gaussians](notebooks/1.%20Gaussians.ipynb)                         | Densities, sampling, precision, marginals, transformations, and conditioning                 |
| 2   | [Bayesian Regression](notebooks/2.%20Bayesian%20Regression.ipynb)   | Gaussian weight posteriors, online updates, feature maps, and Auto MPG regression            |
| 3   | [Gaussian Processes](notebooks/3.%20Gaussian%20Processes.ipynb)     | Mean functions, covariance kernels, and kernel composition                                   |
| 4   | [GP Regression](notebooks/4.%20GP%20Regression.ipynb)               | Posterior predictions, marginal likelihood, and atmospheric CO₂ regression                   |
| 5   | [Kalman Filters](notebooks/5.%20Kalman%20Filters.ipynb)             | State-space models, tracking, smoothing, continuous-time models, and extended Kalman filters |
| 6   | [Exponential Families](notebooks/6.%20Exponential%20Families.ipynb) | Sufficient statistics, conjugate priors, Bernoulli–Beta and Poisson–Gamma examples           |
| 7   | [Logistic Regression](notebooks/7.%20Logistic%20Regression.ipynb)   | Binary and multiclass classification, Newton's method, and Laplace approximation             |
| 8   | [Deep Learning](notebooks/8.%20Deep%20Learning.ipynb)               | Neural networks, Laplax curvature, Laplace approximation, and predictive uncertainty         |
| 9   | [Sampling](notebooks/9.%20Sampling.ipynb)                           | Direct, rejection, and importance sampling; MH, MALA, HMC, and MCMC diagnostics              |

Familiarity with Python, linear algebra, calculus, and basic probability is useful.
Read the notebooks in order to follow the progression of ideas.

## Setup

Use **Python 3.11**, matching the notebooks' recorded environment. From the
repository root, create and activate an environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m ipykernel install --user --name prob-ml --display-name "Python (prob-ml)"
python -m jupyterlab
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` in PowerShell.

Open a notebook from `notebooks/` and select **Python (prob-ml)**. Run its cells
from top to bottom. The setup cells use `Path.cwd().parent` as the project root,
so the kernel's working directory must be `notebooks/`. In an IDE, configure the
notebook working directory accordingly if imports or relative data paths fail.

`requirements.txt` contains the pinned development environment, including
PyTorch and NVIDIA CUDA packages. Installation can be large, and those pins
are geared toward Linux. The notebooks primarily use JAX, with Equinox, Optax,
and Laplax in notebook 8. Importing `src.gaussian` enables JAX 64-bit mode.

## Shared code

| Class                        | Purpose                                                                                                        |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `Gaussian`                   | Gaussian densities, sampling, scalar marginals, arithmetic, linear conditioning, and online regression updates |
| `GaussianProcess`            | Finite-dimensional Gaussian evaluations and conditioning                                                       |
| `ParametricGaussianProcess`  | Processes induced by feature maps and Gaussian weight priors                                                   |
| `ConditionalGaussianProcess` | Posterior mean and covariance after observing noisy data                                                       |
| `ExponentialFamily`          | Abstract interface for sufficient statistics, log densities, and conjugate updates                             |
| `ConjugateFamily`            | Conjugate-prior representation for an exponential-family likelihood                                            |

For example, run this from the repository root:

```python
import jax.numpy as jnp
from jax import random
from src.gaussian import Gaussian

prior = Gaussian(mu=jnp.zeros(2), Sigma=jnp.eye(2))
posterior = prior.condition(
    A=jnp.array([[1.0, 0.0]]),
    y=jnp.array([2.0]),
    R=jnp.array([[0.25]]),  # Observation noise covariance.
)
samples = posterior.sample(sample_size=100, key=random.key(0))
print(posterior.mu)
print(samples.shape)  # (100, 2)
```

These are educational implementations. Gaussian addition assumes independence;
multiplication returns the normalized Gaussian corresponding to a product of
densities. Multivariate CDF evaluation uses Monte Carlo, and sampling adds a
small covariance jitter for numerical stability.

## Data and execution

- `data/auto-mpg.csv` is read locally by notebook 2.
- Notebook 4 downloads monthly Mauna Loa CO₂ data from NOAA, so that example
  needs internet access and its results can change as the source is updated.
  A local `data/co2_mm_mlo.csv` is also present, but the notebook currently uses
  the remote data.
- Other examples use synthetic data or scikit-learn datasets such as Iris.
- Neural-network training and sampling experiments can take substantially
  longer than the introductory notebooks. No full-series runtime is claimed.

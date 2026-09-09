from __future__ import annotations

import abc
import functools
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int
from numpy.typing import ArrayLike
from typing_extensions import Self

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Gaussian:
    """Multivariate Gaussian distribution."""

    mu: Float[Array, "D"]
    Sigma: Float[Array, "D D"]

    def __post_init__(self):
        """Normalize and validate shapes."""

        mu = jnp.atleast_1d(self.mu)
        Sigma = jnp.atleast_2d(self.Sigma)

        if Sigma.shape != (mu.size, mu.size):
            raise ValueError("Sigma must have shape (D, D).")

        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "Sigma", Sigma)

    @functools.cached_property
    def cov_SVD(self):
        """Compute covariance SVD."""

        U, S, VH = jnp.linalg.svd(self.Sigma)
        return U, jnp.sqrt(S), VH

    @property
    def logdet(self):
        """Return log determinant."""

        return jnp.linalg.slogdet(self.Sigma)[1]

    @property
    def variance(self):
        """Return marginal variances."""

        return jnp.diag(self.Sigma)

    @property
    def std(self):
        """Return marginal standard deviations."""

        return jnp.sqrt(self.variance)

    @property
    def dim(self):
        """Return distribution dimension."""

        return self.mu.shape[-1]

    @functools.cached_property
    def cholesky(self) -> Float[Array, "D D"]:
        """Compute covariance Cholesky factor."""
        Sigma = (self.Sigma + self.Sigma.T) / 2

        jitter = 1e-6 * jnp.maximum(
            jnp.mean(jnp.diag(Sigma)),
            1.0,
        )

        return jnp.linalg.cholesky(Sigma + jitter * jnp.eye(self.dim))

    def sample(self, sample_size: Int, key: Array) -> Float[Array, "L D"]:
        """Draw Gaussian samples."""

        L = self.cholesky
        z = random.normal(shape=(sample_size, self.mu.shape[0]), key=key)

        return self.mu + z @ L.T

    def log_pdf(self, x: Float[Array, "... D"]) -> Float[Array, "..."]:
        """Evaluate log density."""

        x = jnp.asarray(x)
        diff = x - self.mu

        solve = jnp.linalg.solve(
            self.Sigma,
            diff[..., None],
        )[..., 0]

        quad = jnp.sum(diff * solve, axis=-1)

        return -0.5 * (self.dim * jnp.log(2 * jnp.pi) + self.logdet + quad)

    def pdf(self, x: Float[Array, "... D"]) -> Float[Array, "..."]:
        """Evaluate density."""

        return jnp.exp(self.log_pdf(x))

    def cdf(self, x: Float[Array, "D"], key: Array | None = None) -> Float:
        """Evaluate the CDF."""

        if key is None:
            key = random.key(47)

        if self.dim == 1:
            z = (x[0] - self.mu[0]) / (self.std[0] * jnp.sqrt(2))
            return 0.5 * (1 + jax.lax.erf(z))

        random_sample = self.sample(sample_size=100_000, key=key)
        return jnp.mean((random_sample <= x).all(axis=1))

    @functools.cached_property
    def precision(self):
        """Return precision matrix."""

        U, S, VH = self.cov_SVD
        return U @ jnp.diag(1 / S) ** 2 @ VH

    @functools.cached_property
    def mp(self):
        """Return precision-weighted mean."""

        return self.precision @ self.mu

    def prec_mult(self, other: Float[Array, "D"]) -> Float[Array, "D"]:
        """Multiply by the precision matrix."""

        return self.precision @ other

    def __getitem__(self, i: Int):
        """Return a marginal Gaussian."""

        return Gaussian(jnp.atleast_1d(self.mu[i]), jnp.atleast_2d(self.Sigma[i][i]))

    def __add__(self, other: Gaussian | Float[Array, "D"]) -> Self:
        """Add an offset or Gaussian."""

        if isinstance(other, Gaussian):
            return Gaussian(
                self.mu + other.mu,
                self.Sigma + other.Sigma,
            )

        return Gaussian(
            self.mu + other,
            self.Sigma,
        )

    def __mul__(self, other: Self) -> Self:
        """Multiply Gaussian densities."""

        A_inv = self.precision
        B_inv = other.precision
        C = jnp.linalg.inv(A_inv + B_inv)
        c = C @ (self.mp + other.mp)
        return Gaussian(c, C)

    def __rmul__(self, c: Float) -> Self:
        """Scale the Gaussian."""

        return Gaussian(c * self.mu, c**2 * self.Sigma)

    def __rmatmul__(self, A: Float[Array, "N D"]) -> Self:
        """Apply a linear transformation."""

        return Gaussian(A @ self.mu, A @ self.Sigma @ A.T)

    def condition(
        self,
        A: Float[Array, "N D"],
        y: Float[Array, "N"],
        R: Float[Array, "N N"],
        jitter: str | Float = 1e-6,
    ) -> Self:
        """Condition on a linear observation."""

        A = jnp.asarray(A, dtype=jnp.float64)

        Gram = A @ self.Sigma @ A.T + R

        if jitter == "auto":
            Gram += 1e-6 * jnp.mean(jnp.diag(Gram)) * jnp.eye(Gram.shape[0])
        else:
            Gram += jitter * jnp.eye(Gram.shape[0])

        L = jax.scipy.linalg.cho_factor(Gram, lower=True)

        mu = self.mu + self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(L, y - A @ self.mu)
        Sigma = self.Sigma - self.Sigma @ A.T @ jax.scipy.linalg.cho_solve(
            L, A @ self.Sigma
        )

        return Gaussian(mu, Sigma)

    def fit_online(self, x: Float[Array, "D"], y: Float, sigma: Float) -> Self:
        """Update with one observation."""

        Sigma = jnp.linalg.inv(self.precision + 1 / sigma**2 * jnp.outer(x, x))
        mu = Sigma @ (self.precision @ self.mu + 1 / sigma**2 * x * y)

        return Gaussian(mu=mu, Sigma=Sigma)


@dataclass
class GaussianProcess:
    """Gaussian process."""

    mu: Callable[[jnp.ndarray], jnp.ndarray]
    k: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __call__(self, x):
        """Evaluate the process."""

        return Gaussian(self.mu(x), self.k(x[:, None, :], x[None, :, :]))

    def condition(self, y, X, R):
        """Condition on observations."""

        return ConditionalGaussianProcess(self, y, X, Gaussian(jnp.zeros_like(y), R))


class ParametricGaussianProcess(GaussianProcess):
    """Parametric Gaussian process."""

    def __init__(self, phi: Callable[[jnp.ndarray], jnp.ndarray], prior: Gaussian):
        """Initialize from features and prior."""

        self.phi = phi
        self.prior = prior
        super().__init__(self._mean, self._covariance)

    def _mean(self, x):
        """Compute the mean."""

        x = jnp.asarray(x)
        return self.phi(x) @ self.prior.mu

    def _covariance(self, x1, x2):
        """Compute the covariance."""

        phi1 = self.phi(jnp.asarray(x1))
        phi2 = self.phi(jnp.asarray(x2))

        return jnp.einsum(
            "...i,ij,...j->...",
            phi1,
            self.prior.Sigma,
            phi2,
        )


class ConditionalGaussianProcess(GaussianProcess):
    """Conditional Gaussian process."""

    def __init__(self, prior, y, X, epsilon: Gaussian):
        """Initialize the conditioned process."""

        self.prior = prior
        self.y = jnp.atleast_1d(y)
        self.X = jnp.atleast_2d(X)
        self.epsilon = epsilon
        super().__init__(self._mean, self._covariance)

    @functools.cached_property
    def predictive_covariance(self):
        """Return predictive covariance."""

        return self.epsilon.Sigma + self.prior.k(self.X[:, None, :], self.X[None, :, :])

    @functools.cached_property
    def predictive_covariance_cho(self):
        """Factor predictive covariance."""

        return jax.scipy.linalg.cho_factor(self.predictive_covariance, lower=True)

    @functools.cached_property
    def representer_weights(self):
        """Compute representer weights."""

        return jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho,
            self.y - self.prior(self.X).mu,
        )

    def _mean(self, x):
        """Compute posterior mean."""

        x = jnp.asarray(x)
        return (
            self.prior(x).mu
            + self.prior.k(x[..., None, :], self.X[None, :, :])
            @ self.representer_weights
        )

    @functools.partial(jnp.vectorize, signature="(d),(d)->()", excluded={0})
    def _covariance(self, a, b):
        """Compute posterior covariance."""

        return self.prior.k(a, b) - self.prior.k(
            a, self.X
        ) @ jax.scipy.linalg.cho_solve(
            self.predictive_covariance_cho, self.prior.k(self.X, b)
        )


class ExponentialFamily(abc.ABC):
    """Exponential-family distribution."""

    @abc.abstractmethod
    def sufficient_statistics(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Compute sufficient statistics."""

    @abc.abstractmethod
    def log_base_measure(self, x: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Compute log base measure."""

    @abc.abstractmethod
    def log_partition(
        self, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Compute log partition function."""

    def log_pdf(
        self, x: ArrayLike | jnp.ndarray, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Evaluate log density."""

        x = jnp.asarray(x)
        natural_parameters = jnp.asarray(natural_parameters)

        linear_term = (
            self.sufficient_statistics(x)[..., None, :] @ natural_parameters[..., None]
        )[..., 0, 0]
        return (
            self.log_base_measure(x)
            + linear_term
            - self.log_partition(natural_parameters)
        )

    def conjugate_log_partition(
        self, alpha: ArrayLike | jnp.ndarray, nu: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Compute conjugate log partition."""

        raise NotImplementedError()

    def conjugate_prior(self) -> "ConjugateFamily":
        """Return the conjugate family."""

        return ConjugateFamily(self)

    def posterior_parameters(
        self,
        prior_natural_parameters: ArrayLike | jnp.ndarray,
        data: ArrayLike | jnp.ndarray,
        /,
    ):
        """Compute posterior parameters."""

        prior_natural_parameters = jnp.asarray(prior_natural_parameters)

        prior_alpha, prior_nu = (
            prior_natural_parameters[:-1],
            prior_natural_parameters[-1],
        )
        sufficient_statistics = self.sufficient_statistics(data)
        n = sufficient_statistics[..., 0].size

        expected_sufficient_statistics = jnp.sum(
            sufficient_statistics, axis=tuple(range(sufficient_statistics.ndim - 1))
        )

        return jnp.append(prior_alpha + expected_sufficient_statistics, prior_nu + n)


class ConjugateFamily(ExponentialFamily):
    """Conjugate exponential family."""

    def __init__(self, likelihood: ExponentialFamily) -> None:
        """Initialize from a likelihood."""

        self._likelihood = likelihood

    def sufficient_statistics(self, eta: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Compute sufficient statistics."""

        eta = jnp.asarray(eta)

        return jnp.concatenate(
            [
                eta,
                -self._likelihood.log_partition(eta)[..., None],
            ],
            axis=-1,
        )

    def log_base_measure(self, eta: ArrayLike | jnp.ndarray, /) -> jnp.ndarray:
        """Compute log base measure."""

        eta = jnp.asarray(eta)

        return jnp.zeros_like(eta[..., 0])

    def log_partition(
        self, natural_parameters: ArrayLike | jnp.ndarray, /
    ) -> jnp.ndarray:
        """Compute log partition function."""

        natural_parameters = jnp.asarray(natural_parameters)

        alpha, nu = natural_parameters[:-1], natural_parameters[-1]
        return self._likelihood.conjugate_log_partition(alpha, nu)

    def unnormalized_log_pdf(
        self,
        eta: ArrayLike | jnp.ndarray,
        natural_parameters: ArrayLike | jnp.ndarray,
        /,
    ) -> jnp.ndarray:
        """Evaluate unnormalized log density."""

        return self.sufficient_statistics(eta) @ jnp.asarray(natural_parameters)

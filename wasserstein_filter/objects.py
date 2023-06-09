from typing import NamedTuple, Callable
from jax import numpy as jnp

from jax.scipy.stats import multivariate_normal as mvn


class MVNStandard(NamedTuple):
    mean: jnp.array
    cov: jnp.array

    @property
    def dim(self):
        return self.mean.shape[-1]


class MVNSqrt(NamedTuple):
    mean: jnp.array
    cov_sqrt: jnp.array

    @property
    def dim(self):
        return self.mean.shape[-1]

    def logpdf(self, x):
        mu = self.mean
        sigma = self.cov_sqrt @ self.cov_sqrt.T
        return mvn.logpdf(x, mu, sigma)


class GMMSqrt(NamedTuple):
    mean: jnp.array
    cov_sqrt: jnp.array

    @property
    def dim(self):
        return self.mean.shape[-1]

    @property
    def size(self):
        return self.mean.shape[0]


class ConditionalMVN(NamedTuple):
    mean: Callable
    cov: Callable

    def logpdf(self, x: jnp.ndarray, y: jnp.ndarray):
        mu = self.mean(x)
        sigma = self.cov(x)
        return mvn.logpdf(y, mu, sigma)


class ConditionalMVNSqrt(NamedTuple):
    mean: Callable
    cov_sqrt: Callable

    def logpdf(self, x: jnp.ndarray, y: jnp.ndarray):
        mu = self.mean(x)
        sigma_sqrt = self.cov_sqrt(x)
        return mvn.logpdf(y, mu, sigma_sqrt @ sigma_sqrt.T)

from typing import NamedTuple, Callable
from jax import numpy as jnp

from jax.scipy.stats import multivariate_normal as mvn
from tensorflow_probability.substrates.jax.distributions import LogNormal as lognorm


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


class GMMSqrt(NamedTuple):
    mean: jnp.array
    cov_sqrt: jnp.array

    @property
    def dim(self):
        return self.mean.shape[-1]

    @property
    def size(self):
        return self.mean.shape[0]


class ConditionalLogNorm(NamedTuple):
    loc: Callable
    scale: Callable

    def logpdf(self, x, y):
        _loc = self.loc(x)
        _scale = self.scale(x)
        return lognorm(_loc, _scale).log_prob(y).squeeze()


class ConditionalMVN(NamedTuple):
    mean: Callable
    cov: Callable

    def logpdf(self, x, y):
        mu = self.mean(x)
        sigma = self.cov(x)
        return mvn.logpdf(y, mu, sigma)


class ConditionalMVNSqrt(NamedTuple):
    mean: Callable
    cov_sqrt: Callable

    def logpdf(self, x, y):
        mu = self.mean(x)
        sigma_sqrt = self.cov_sqrt(x)
        return mvn.logpdf(y, mu, sigma_sqrt @ sigma_sqrt.T)

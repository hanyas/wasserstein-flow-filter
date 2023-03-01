from typing import NamedTuple, Callable
from jax import numpy as jnp


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


class FunctionalModel(NamedTuple):
    function: Callable
    mvn: MVNStandard


class ConditionalModel(NamedTuple):
    mean: Callable
    cov: Callable


class ConditionalModelSqrt(NamedTuple):
    mean: Callable
    cov_sqrt: Callable

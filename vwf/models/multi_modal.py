import jax
from jax import numpy as jnp

from vwf.objects import ConditionalMVN
from vwf.objects import ConditionalLogNorm

from tensorflow_probability.substrates.jax.distributions import LogNormal as lognorm


def generate_data(key, x0, T, params):
    nx, ny = 1, 1

    s = params
    trns_mdl, obs_mdl = build_model(params)

    def transition_fcn(key, x):
        _mu = trns_mdl.mean(x)
        _sigma = trns_mdl.cov(x)

        key, sub_key = jax.random.split(key, 2)
        xn = _mu + jnp.linalg.cholesky(_sigma) @ jax.random.normal(sub_key, shape=(nx,))
        return key, xn

    def observation_fcn(key, x):
        _loc = obs_mdl.loc(x)
        _scale = obs_mdl.scale(x)

        key, sub_key = jax.random.split(key, 2)
        y = lognorm(_loc, _scale).sample(seed=sub_key)
        return key, y

    def body(carry, args):
        key, x = carry
        key, xn = transition_fcn(key, x)
        key, yn = observation_fcn(key, xn)
        return (key, xn), (xn, yn)

    key, sub_key = jax.random.split(key, 2)

    m0, P0 = x0
    x0 = m0 + jnp.linalg.cholesky(P0) @ jax.random.normal(sub_key, shape=(nx,))
    (key, _), (Xs, Ys) = jax.lax.scan(body, (key, x0), (), length=T)

    Xs = jnp.insert(Xs, 0, x0, 0)
    return Xs, Ys


def build_model(params):
    s = params

    def trns_mean(x):
        return x

    def trns_cov(x):
        return jnp.eye(1)

    def obs_loc(x):
        return jnp.abs(x) - jnp.exp(0.5 * s ** 2)

    def obs_scale(x):
        return s

    trns_mdl = ConditionalMVN(trns_mean, trns_cov)
    obs_mdl = ConditionalLogNorm(obs_loc, obs_scale)
    return trns_mdl, obs_mdl

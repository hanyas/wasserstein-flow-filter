import jax
from jax import numpy as jnp

from vwf.objects import ConditionalMVNSqrt


def generate_data(key, x0, T, params):
    nx, ny = 1, 1

    s = params
    trns_mdl, obs_mdl = build_model(params)

    def transition_fcn(key, x):
        _mu = trns_mdl.mean(x)
        _sigma_sqrt = trns_mdl.cov_sqrt(x)

        key, sub_key = jax.random.split(key, 2)
        xn = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(nx,))
        return key, xn

    def observation_fcn(key, x):
        _mu = obs_mdl.mean(x)
        _sigma_sqrt = obs_mdl.cov_sqrt(x)

        key, sub_key = jax.random.split(key, 2)
        y = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(ny,))
        return key, y

    def body(carry, args):
        key, x = carry
        key, xn = transition_fcn(key, x)
        key, yn = observation_fcn(key, xn)
        return (key, xn), (xn, yn)

    key, sub_key = jax.random.split(key, 2)

    m0, P0_sqrt = x0
    x0 = m0 + P0_sqrt @ jax.random.normal(sub_key, shape=(nx,))
    (key, _), (Xs, Ys) = jax.lax.scan(body, (key, x0), (), length=T)

    Xs = jnp.insert(Xs, 0, x0, 0)
    return Xs, Ys


def build_model(params):
    s = params

    def trns_mean(x):
        return x

    def trns_cov_sqrt(x):
        return jnp.eye(1)

    def obs_mean(x):
        return jnp.abs(x)

    def obs_cov_sqrt(x):
        return jnp.eye(1) * s

    trns_mdl = ConditionalMVNSqrt(trns_mean, trns_cov_sqrt)
    obs_mdl = ConditionalMVNSqrt(obs_mean, obs_cov_sqrt)
    return trns_mdl, obs_mdl

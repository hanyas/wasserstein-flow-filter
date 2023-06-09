import jax
from jax import numpy as jnp

from wasserstein_filter.objects import ConditionalMVNSqrt


def generate_data(key, prior, length, params):
    nx, ny = 1, 1

    s = params
    trans_mdl, obsrv_mdl = build_model(params)

    def transition_fcn(key, x):
        _mu = trans_mdl.mean(x)
        _sigma_sqrt = trans_mdl.cov_sqrt(x)

        key, sub_key = jax.random.split(key, 2)
        xn = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(nx,))
        return key, xn

    def observation_fcn(key, x):
        _mu = obsrv_mdl.mean(x)
        _sigma_sqrt = obsrv_mdl.cov_sqrt(x)

        key, sub_key = jax.random.split(key, 2)
        y = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(ny,))
        return key, y

    def body(carry, args):
        key, x = carry
        key, xn = transition_fcn(key, x)
        key, yn = observation_fcn(key, xn)
        return (key, xn), (xn, yn)

    key, sub_key = jax.random.split(key, 2)

    m0, P0_sqrt = prior
    # x0 = m0 + P0_sqrt @ jax.random.normal(sub_key, shape=(nx,))
    x0 = jnp.array([-3.0])
    (key, _), (Xs, Ys) = jax.lax.scan(body, (key, x0), (), length=length)

    Xs = jnp.insert(Xs, 0, x0, 0)
    return Xs, Ys


def build_model(params):
    s = params

    def trans_mean(x):
        return x

    def trans_cov_sqrt(x):
        return jnp.eye(1)

    def obsrv_mean(x):
        return jnp.abs(x)

    def obsrv_cov_sqrt(x):
        return jnp.eye(1) * s

    trans_mdl = ConditionalMVNSqrt(trans_mean, trans_cov_sqrt)
    obsrv_mdl = ConditionalMVNSqrt(obsrv_mean, obsrv_cov_sqrt)
    return trans_mdl, obsrv_mdl

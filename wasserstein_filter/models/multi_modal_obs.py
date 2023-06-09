import jax
from jax import numpy as jnp

from wasserstein_filter.objects import ConditionalMVN


def generate_data(key, xi, T, params):
    nx, ny = 1, 1

    s = params
    trans_mdl, obsrv_mdl = build_model(params)

    def transition_fcn(key, x):
        _mu = trans_mdl.mean(x)
        _sigma = trans_mdl.cov(x)
        _sigma_sqrt = jnp.linalg.cholesky(_sigma)

        key, sub_key = jax.random.split(key, 2)
        xn = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(nx,))
        return key, xn

    def observation_fcn(key, x):
        _mu = obsrv_mdl.mean(x)
        _sigma = obsrv_mdl.cov(x)
        _sigma_sqrt = jnp.linalg.cholesky(_sigma)

        key, sub_key = jax.random.split(key, 2)
        y = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(ny,))
        return key, y

    def body(carry, args):
        key, x = carry
        key, xn = transition_fcn(key, x)
        key, yn = observation_fcn(key, xn)
        return (key, xn), (xn, yn)

    key, sub_key = jax.random.split(key, 2)

    m0, P0 = xi
    P0_sqrt = jnp.linalg.cholesky(P0)
    # x0 = m0 + P0_sqrt @ jax.random.normal(sub_key, shape=(nx,))
    x0 = jnp.array([-3.0])
    (key, _), (Xs, Ys) = jax.lax.scan(body, (key, x0), (), length=T)

    Xs = jnp.insert(Xs, 0, x0, 0)
    return Xs, Ys


def build_model(params):
    s = params

    def trans_mean(x):
        return x

    def trans_cov(x):
        return jnp.eye(1)

    def obsrv_mean(x):
        return jnp.abs(x)

    def obsrv_cov(x):
        return jnp.eye(1) * s**2

    trans_mdl = ConditionalMVN(trans_mean, trans_cov)
    obsrv_mdl = ConditionalMVN(obsrv_mean, obsrv_cov)
    return trans_mdl, obsrv_mdl

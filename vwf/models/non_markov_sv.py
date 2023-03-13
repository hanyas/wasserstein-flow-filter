import jax
from jax import numpy as jnp

from vwf.objects import ConditionalMVN


def generate_data(key, x0, T, params):
    nx, ny = 1, 1

    mu, a, sig, rho = params
    trans_mdl, obsrv_mdl = build_model(params)

    def transition_fcn(key, x, y):
        _mu = trans_mdl.mean(x, y)
        _sigma = trans_mdl.cov(x, y)
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
        key, x, y = carry
        key, xn = transition_fcn(key, x, y)
        key, yn = observation_fcn(key, xn)
        return (key, xn, yn), (xn, yn)

    key, sub_key = jax.random.split(key, 2)

    m0, P0 = x0
    P0_sqrt = jnp.linalg.cholesky(P0)
    x0 = m0 + P0_sqrt @ jax.random.normal(sub_key, shape=(nx,))

    key, sub_key = jax.random.split(key, 2)
    _mu, _sigma = obsrv_mdl.mean(x0), obsrv_mdl.cov(x0)
    _sigma_sqrt = jnp.linalg.cholesky(_sigma)
    y0 = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(ny,))

    (key, _, _), (Xs, Ys) = jax.lax.scan(body, (key, x0, y0), (), length=T - 1)

    Xs = jnp.insert(Xs, 0, x0, 0)
    Ys = jnp.insert(Ys, 0, y0, 0)
    return Xs, Ys


def build_model(params):
    mu, a, sig, rho = params

    def trans_mean(x, y):
        xn = mu * (1.0 - a) + a * x + sig * rho * y * jnp.exp(-x / 2.0)
        return xn

    def trans_cov(x, y):
        _cov = jnp.array([sig**2 * (1.0 - rho**2)])
        return jnp.diag(_cov)

    def obsrv_mean(x):
        return jnp.array([0.0])

    def obsrv_cov(x):
        _cov = jnp.exp(x)
        return jnp.diag(_cov)

    trans_mdl = ConditionalMVN(trans_mean, trans_cov)
    obsrv_mdl = ConditionalMVN(obsrv_mean, obsrv_cov)
    return trans_mdl, obsrv_mdl

import jax
from jax import numpy as jnp

from vwf.objects import ConditionalMVN


def generate_data(key, x0, T, params):
    nx, ny = 1, 1

    mu, a, sig, rho = params
    trns_mdl, obs_mdl = build_model(params)

    def transition_fcn(key, x, y):
        _mu = trns_mdl.mean(x, y)
        _sigma = trns_mdl.cov(x, y)
        _sigma_sqrt = jnp.linalg.cholesky(_sigma)

        key, sub_key = jax.random.split(key, 2)
        xn = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(nx,))
        return key, xn

    def observation_fcn(key, x):
        _mu = obs_mdl.mean(x)
        _sigma = obs_mdl.cov(x)
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
    _mu, _cov = obs_mdl.mean(x0), obs_mdl.cov(x0)
    y0 = _mu + jnp.linalg.cholesky(_cov) @ jax.random.normal(
        sub_key, shape=(ny,)
    )

    (key, _, _), (Xs, Ys) = jax.lax.scan(body, (key, x0, y0), (), length=T - 1)

    Xs = jnp.insert(Xs, 0, x0, 0)
    Ys = jnp.insert(Ys, 0, y0, 0)
    return Xs, Ys


def build_model(params):
    mu, a, sig, rho = params

    def trns_mean(x, y):
        xn = mu * (1.0 - a) + a * x + sig * rho * y * jnp.exp(-x / 2.0)
        return xn

    def trns_cov(x, y):
        _cov = jnp.array([sig**2 * (1.0 - rho**2)])
        return jnp.diag(_cov)

    def obs_mean(x):
        return jnp.array([0.0])

    def obs_cov(x):
        _cov = jnp.exp(x)
        return jnp.diag(_cov)

    trns_mdl = ConditionalMVN(trns_mean, trns_cov)
    obs_mdl = ConditionalMVN(obs_mean, obs_cov)
    return trns_mdl, obs_mdl

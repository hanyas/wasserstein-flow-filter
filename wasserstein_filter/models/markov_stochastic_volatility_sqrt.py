import jax
from jax import numpy as jnp

from wasserstein_filter.objects import ConditionalMVNSqrt


def generate_data(key, prior, length, params):
    nz, ny = 2, 1

    mu, a, sig, rho = params
    trans_mdl, obsrv_mdl = build_model(params)

    def transition_fcn(key, z):
        _mu = trans_mdl.mean(z)
        _sigma_sqrt = trans_mdl.cov_sqrt(z)

        key, sub_key = jax.random.split(key, 2)
        zn = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(nz,))
        return key, zn

    def observation_fcn(key, z):
        _mu = obsrv_mdl.mean(z)
        _sigma_sqrt = obsrv_mdl.cov_sqrt(z)

        key, sub_key = jax.random.split(key, 2)
        y = _mu + _sigma_sqrt @ jax.random.normal(sub_key, shape=(ny,))
        return key, y

    def body(carry, args):
        key, z = carry
        key, zn = transition_fcn(key, z)
        key, yn = observation_fcn(key, zn)
        return (key, zn), (zn, yn)

    key, sub_key = jax.random.split(key, 2)

    m0, P0_sqrt = prior
    z0 = m0 + P0_sqrt @ jax.random.normal(sub_key, shape=(nz,))
    (key, _), (Zs, Ys) = jax.lax.scan(body, (key, z0), (), length=length)

    Zs = jnp.insert(Zs, 0, z0, 0)
    return Zs, Ys


def build_model(params):
    mu, a, sig, rho = params

    def trans_mean(z):
        x, eta = z
        F = jnp.array([[a, sig], [0.0, 0.0]])
        b = jnp.array([mu * (1.0 - a), 0.0])
        return F @ z + b

    def trans_cov_sqrt(z):
        x, eta = z
        _cov = jnp.array([1e-32, 1.0])
        return jnp.diag(jnp.sqrt(_cov))

    def obsrv_mean(z):
        x, eta = z
        return jnp.array([rho * eta * jnp.exp(x / 2.0)])

    def obsrv_cov_sqrt(z):
        x, eta = z
        _cov = jnp.array([(1.0 - rho**2) * jnp.exp(x)])
        return jnp.diag(jnp.sqrt(_cov))

    trans_mdl = ConditionalMVNSqrt(trans_mean, trans_cov_sqrt)
    obsrv_mdl = ConditionalMVNSqrt(obsrv_mean, obsrv_cov_sqrt)
    return trans_mdl, obsrv_mdl

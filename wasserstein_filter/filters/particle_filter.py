from typing import Callable

import jax
import jax.random

from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from jax.scipy.special import logsumexp

from wasserstein_filter.objects import MVNStandard
from wasserstein_filter.objects import ConditionalMVN


def _avg_n_nplusone(x):
    """returns x[0]/2, (x[0]+x[1])/2, ... (x[-2]+x[-1])/2, x[-1]/2"""
    hx = 0.5 * x
    y = jnp.pad(hx, [[0, 1]], constant_values=0.0, mode="constant")
    y = y.at[..., 1:].add(hx)
    return y


def continuous_resampling(x, w, u):
    _x, _unravel_fcn = ravel_pytree(x)
    idx = jnp.argsort(_x)
    _xs, ws = _x[idx], w[idx]
    cs = jnp.cumsum(_avg_n_nplusone(ws))
    cs = cs[:-1]
    return _unravel_fcn(jnp.interp(u, cs, _xs))


def stratified_resampling(x, w, u):
    N = w.shape[0]
    cs = jnp.cumsum(w)
    idx = jnp.searchsorted(cs, u, side="left")
    return x[jnp.clip(idx, 0, N - 1)]


def particle_filter(
    key: jax.random.PRNGKey,
    nb_particles: int,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
    resampling_scheme: Callable = stratified_resampling,
):
    N = nb_particles

    def _propagate(x, q, trans_mdl):
        xn = trans_mdl.mean(x) + jnp.linalg.cholesky(trans_mdl.cov(x)) @ q
        return xn

    def _log_weights(x, y, obsrv_mdl):
        return obsrv_mdl.logpdf(x, y)

    def body(carry, args):
        x, ell = carry
        y, q, u = args

        # propagate
        xn = jax.vmap(_propagate, in_axes=(0, 0, None))(x, q, transition_model)

        # weights
        log_wn = jax.vmap(_log_weights, in_axes=(0, None, None))(
            xn, y, observation_model
        )

        # normalize
        log_wn_norm = logsumexp(log_wn)
        wn = jnp.exp(log_wn - log_wn_norm)

        # ell
        ell += log_wn_norm - jnp.log(N)

        # resample
        xn = stratified_resampling(xn, wn, u)
        return (xn, ell), (xn, wn)

    m0, P0 = initial_dist
    d = initial_dist.dim

    y = observations
    T = y.shape[0]

    key, x_key, q_key, u_key = jax.random.split(key, 4)

    x0 = m0 + jnp.einsum(
        "ij,ni->nj",
        jnp.linalg.cholesky(P0),
        jax.random.normal(x_key, shape=(N, d)),
    )

    # random vectors
    q = jax.random.normal(q_key, shape=(T, N, d))

    # resampling uniforms
    u = jax.random.uniform(
        u_key,
        shape=(T, N),
        minval=jnp.arange(0, N) / N,
        maxval=jnp.arange(1, N + 1) / N,
    )

    (_, ell), (Xs, Ws) = jax.lax.scan(body, (x0, 0.0), (y, q, u))

    w0 = jnp.ones((N, )) / N
    Xs = jnp.insert(Xs, 0, x0, 0)
    Ws = jnp.insert(Ws, 0, w0, 0)
    return Xs, ell, Ws


def non_markov_particle_filter(
    key: jax.random.PRNGKey,
    nb_particles: int,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
    resampling_scheme: Callable = stratified_resampling,
):
    N = nb_particles

    def _propagate(x, y, q, trans_mdl):
        _mu = trans_mdl.mean(x, y)
        _sigma_sqrt = jnp.linalg.cholesky(trans_mdl.cov(x, y))
        xn = _mu + _sigma_sqrt @ q
        return xn

    def _log_weights(x, y, obsrv_mdl):
        return obsrv_mdl.logpdf(x, y)

    def body(carry, args):
        x, ell = carry
        y, q, u = args

        # weights
        log_w = jax.vmap(_log_weights, in_axes=(0, None, None))(
            x, y, observation_model
        )

        # normalize
        log_w_norm = logsumexp(log_w)
        w = jnp.exp(log_w - log_w_norm)

        # ell
        ell += log_w_norm - jnp.log(N)

        # resample
        x = resampling_scheme(x, w, u)

        # propagate
        xn = jax.vmap(_propagate, in_axes=(0, None, 0, None))(
            x, y, q, transition_model
        )
        return (xn, ell), (x, w)

    m0, P0 = initial_dist
    d = initial_dist.dim

    y = observations
    T = y.shape[0]

    key, x_key, q_key, u_key = jax.random.split(key, 4)

    x0 = m0 + jnp.einsum(
        "ij,ni->nj",
        jnp.linalg.cholesky(P0),
        jax.random.normal(x_key, shape=(N, d)),
    )

    # random vectors
    q = jax.random.normal(q_key, shape=(T, N, d))

    # resampling uniforms
    u = jax.random.uniform(
        u_key,
        shape=(T, N),
        minval=jnp.arange(0, N) / N,
        maxval=jnp.arange(1, N + 1) / N,
    )

    (_, ell), (Xs, Ws) = jax.lax.scan(body, (x0, 0.0), (y, q, u))
    return Xs, ell, Ws


def non_markov_stratified_particle_filter(
    key: jax.random.PRNGKey,
    nb_particles: int,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
):
    return non_markov_particle_filter(
        key,
        nb_particles,
        observations,
        initial_dist,
        transition_model,
        observation_model,
        stratified_resampling,
    )


def non_markov_diffable_particle_filter(
    key: jax.random.PRNGKey,
    nb_particles: int,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: ConditionalMVN,
    observation_model: ConditionalMVN,
):
    return non_markov_particle_filter(
        key,
        nb_particles,
        observations,
        initial_dist,
        transition_model,
        observation_model,
        continuous_resampling,
    )

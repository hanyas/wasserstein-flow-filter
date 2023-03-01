from typing import Callable

import jax
import jax.random

from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal as mvn

from vwf.objects import MVNStandard, ConditionalModel


def _stratified_resampling(x, w, u):
    N = w.shape[0]
    cs = jnp.cumsum(w)
    idx = jnp.searchsorted(cs, u, side='left')
    return x[jnp.clip(idx, 0, N - 1)]


def _avg_n_nplusone(x):
    """ returns x[0]/2, (x[0]+x[1])/2, ... (x[-2]+x[-1])/2, x[-1]/2 """
    hx = 0.5 * x
    y = jnp.pad(hx, [[0, 1]], constant_values=0.0, mode="constant")
    y = y.at[..., 1:].add(hx)
    return y


def _continuous_resampling(x, w, u):
    _x, _unravel_fcn = ravel_pytree(x)
    idx = jnp.argsort(_x)
    _xs, ws = _x[idx], w[idx]
    cs = jnp.cumsum(_avg_n_nplusone(ws))
    cs = cs[:-1]
    return _unravel_fcn(jnp.interp(u, cs, _xs))


def particle_filter(key: jax.random.PRNGKey,
                    nb_particles: int,
                    observations: jnp.ndarray,
                    initial_dist: MVNStandard,
                    transition_model: ConditionalModel,
                    observation_model: ConditionalModel,
                    resampling_scheme: Callable = _stratified_resampling):

    N = nb_particles

    def _propagate(x, q, trns_mdl):
        xn = trns_mdl.mean(x) \
             + jnp.linalg.cholesky(trns_mdl.cov(x)) @ q
        return xn

    def _log_weights(x, y, obs_mdl):
        mu_y = obs_mdl.mean(x)
        cov_y = obs_mdl.cov(x)
        log_w = mvn.logpdf(y, mu_y, cov_y)
        return log_w

    def body(carry, args):
        x, ell = carry
        y, q, u = args

        # propagate
        xn = jax.vmap(_propagate, in_axes=(0, 0, None))(x, q, transition_model)

        # weights
        log_wn = jax.vmap(_log_weights, in_axes=(0, None, None))(xn, y, observation_model)

        # normalize
        log_wn_norm = logsumexp(log_wn)
        wn = jnp.exp(log_wn - log_wn_norm)  # normalize

        # ell
        ell += log_wn_norm - jnp.log(N)

        # resample
        xn = _stratified_resampling(xn, wn, u)
        return (xn, ell), (xn, wn)

    m0, P0 = initial_dist
    d = initial_dist.dim

    y = observations
    T = y.shape[0]

    key, x_key, q_key, u_key = jax.random.split(key, 4)

    x0 = m0 + jnp.einsum('ij,ni->nj', jnp.linalg.cholesky(P0),
                         jax.random.normal(x_key, shape=(N, d)))

    # random vectors
    q = jax.random.normal(q_key, shape=(T, N, d))

    # resampling uniforms
    u = jax.random.uniform(u_key, shape=(T, N),
                           minval=jnp.arange(0, N) / N,
                           maxval=jnp.arange(1, N + 1) / N)

    (_, ell), (Xs, Ws) = jax.lax.scan(body, (x0, 0.0), (y, q, u))

    Xs = jnp.insert(Xs, 0, x0, 0)
    return Xs, ell, Ws


def non_markov_particle_filter(key: jax.random.PRNGKey,
                               nb_particles: int,
                               observations: jnp.ndarray,
                               initial_dist: MVNStandard,
                               transition_model: ConditionalModel,
                               observation_model: ConditionalModel,
                               resampling_scheme: Callable = _stratified_resampling):

    N = nb_particles

    def _propagate(x, y, q, trns_mdl):
        xn = trns_mdl.mean(x, y) \
             + jnp.linalg.cholesky(trns_mdl.cov(x, y)) @ q
        return xn

    def _log_weights(x, y, obs_mdl):
        mu_y = obs_mdl.mean(x)
        cov_y = obs_mdl.cov(x)
        log_w = mvn.logpdf(y, mu_y, cov_y)
        return log_w

    def body(carry, args):
        x, ell = carry
        y, q, u = args

        # weights
        log_w = jax.vmap(_log_weights, in_axes=(0, None, None))(x, y, observation_model)

        # normalize
        log_w_norm = logsumexp(log_w)
        w = jnp.exp(log_w - log_w_norm)  # normalize

        # ell
        ell += log_w_norm - jnp.log(N)

        # resample
        x = resampling_scheme(x, w, u)

        # propagate
        xn = jax.vmap(_propagate, in_axes=(0, None, 0, None))(x, y, q, transition_model)
        return (xn, ell), (x, w)

    m0, P0 = initial_dist
    d = initial_dist.dim

    y = observations
    T = y.shape[0]

    key, x_key, q_key, u_key = jax.random.split(key, 4)

    x0 = m0 + jnp.einsum('ij,ni->nj', jnp.linalg.cholesky(P0),
                         jax.random.normal(x_key, shape=(N, d)))

    # random vectors
    q = jax.random.normal(q_key, shape=(T, N, d))

    # resampling uniforms
    u = jax.random.uniform(u_key, shape=(T, N),
                           minval=jnp.arange(0, N) / N,
                           maxval=jnp.arange(1, N + 1) / N)

    (_, ell), (Xs, Ws) = jax.lax.scan(body, (x0, 0.0), (y, q, u))
    return Xs, ell, Ws


def non_markov_stratified_particle_filter(key: jax.random.PRNGKey,
                                          nb_particles: int,
                                          observations: jnp.ndarray,
                                          initial_dist: MVNStandard,
                                          transition_model: ConditionalModel,
                                          observation_model: ConditionalModel):

    return non_markov_particle_filter(key, nb_particles, observations,
                                      initial_dist, transition_model, observation_model,
                                      _stratified_resampling)


def non_markov_diffable_particle_filter(key: jax.random.PRNGKey,
                                        nb_particles: int,
                                        observations: jnp.ndarray,
                                        initial_dist: MVNStandard,
                                        transition_model: ConditionalModel,
                                        observation_model: ConditionalModel):

    return non_markov_particle_filter(key, nb_particles, observations,
                                      initial_dist, transition_model, observation_model,
                                      _continuous_resampling)
